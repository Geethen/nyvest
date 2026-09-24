"""On-disk cache for virtual-tiff COG manifests.

Parsing a COG header into a VirtualiZarr ``ManifestStore`` (``VirtualTIFF(...)``)
costs a network round-trip plus decode — ~1.6 s per AEF tile, measured. Nothing
about that manifest changes between sessions (it is derived purely from the COG's
immutable header), yet the stock reader re-parses every tile on every process.
For a wall-to-wall AOI (tens of tiles) that is the dominant cold-start cost.

This module serialises the manifest to a small JSON file keyed by the tile's
cloud path, and rebuilds the ``ManifestStore`` from JSON on subsequent opens —
turning a ~1.6 s header fetch+parse into a sub-millisecond local read. The cached
JSON holds only header-derived metadata (array shape, chunk grid, codecs, and the
per-chunk byte-range table pointing back into the remote object); no pixel data is
cached, so a cache entry is a few hundred KB regardless of tile size.

Correctness: the live ``ObjectStoreRegistry`` is re-attached at load time (it is a
process-local handle to the cloud store, not serialisable state), so a rebuilt
store reads the same remote bytes as a freshly parsed one. Cache keys include the
IFD index and a format version, so overviews and format changes never collide.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from virtualizarr.manifests import ChunkManifest, ManifestArray, ManifestGroup
from virtualizarr.manifests.store import ManifestStore
from zarr.core.metadata.v3 import ArrayV3Metadata

if TYPE_CHECKING:
    from virtualizarr.registry import ObjectStoreRegistry

logger = logging.getLogger(__name__)

# Bump when the on-disk JSON layout or the manifest semantics change, so stale
# entries from an older aef-loader are ignored rather than mis-read.
CACHE_FORMAT_VERSION = 1


def _cache_key(url: str, ifd: int) -> str:
    """Stable filename stem for a (tile url, ifd) pair.

    A hash keeps the filename fixed-length and filesystem-safe regardless of the
    URL, and folds in the format version and ifd so unrelated variants never map
    to the same file.
    """
    raw = f"v{CACHE_FORMAT_VERSION}|ifd{ifd}|{url}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def cache_path_for(cache_dir: Path, url: str, ifd: int) -> Path:
    """Absolute path of the cache file for a (url, ifd) under ``cache_dir``."""
    return Path(cache_dir) / f"manifest_{_cache_key(url, ifd)}.json"


def _manifest_to_jsonable(store: ManifestStore) -> dict:
    """Extract the JSON-serialisable state of a single-group ``ManifestStore``.

    Captures, per array: the Zarr V3 metadata dict, and the chunk manifest in a
    *columnar* form — flat ``paths``/``offsets``/``lengths`` lists plus the chunk
    grid ``shape``. Columnar beats the nested ``{key: {...}}`` dict on both counts
    that matter here: the JSON is markedly smaller, and reconstruction uses
    ``ChunkManifest.from_arrays`` (vectorised) instead of the per-entry
    ``ChunkManifest(entries=...)`` path, which is ~50 ms for a 4096-chunk tile.
    Chunk order is row-major over the grid so keys are implicit and need not be
    stored. Also captures group-level attributes. Deliberately does NOT capture
    the registry (see module docstring).
    """
    group = store._group
    arrays_out: dict[str, dict] = {}
    for name, marr in group.arrays.items():
        manifest = marr.manifest
        # These three parallel ndarrays are the manifest's native columnar layout,
        # shaped like the chunk grid. Row-major flatten keeps them aligned.
        paths = np.asarray(manifest._paths).reshape(-1)
        offsets = np.asarray(manifest._offsets).reshape(-1)
        lengths = np.asarray(manifest._lengths).reshape(-1)
        arrays_out[name] = {
            "metadata": marr.metadata.to_dict(),
            "chunk_grid_shape": list(manifest.shape_chunk_grid),
            "paths": paths.tolist(),
            "offsets": offsets.tolist(),
            "lengths": lengths.tolist(),
        }
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "group_attributes": dict(group.metadata.attributes),
        "arrays": arrays_out,
    }


def _jsonable_to_manifest(
    data: dict, registry: ObjectStoreRegistry
) -> ManifestStore:
    """Rebuild a ``ManifestStore`` from :func:`_manifest_to_jsonable` output.

    ``registry`` is the live, process-local object store to attach so the rebuilt
    store can fetch remote chunk bytes.
    """
    arrays: dict[str, ManifestArray] = {}
    for name, arr in data["arrays"].items():
        metadata = ArrayV3Metadata.from_dict(arr["metadata"])
        shape = tuple(arr["chunk_grid_shape"])
        # Rebuild the columnar arrays and hand them to from_arrays (vectorised).
        # validate_paths=False skips a per-path URL re-parse: paths came straight
        # from a manifest we serialised ourselves, so they are already well-formed.
        paths = np.array(arr["paths"], dtype=np.dtypes.StringDType()).reshape(shape)
        offsets = np.array(arr["offsets"], dtype=np.uint64).reshape(shape)
        lengths = np.array(arr["lengths"], dtype=np.uint64).reshape(shape)
        manifest = ChunkManifest.from_arrays(
            paths=paths, offsets=offsets, lengths=lengths, validate_paths=False
        )
        arrays[name] = ManifestArray(metadata=metadata, chunkmanifest=manifest)
    group = ManifestGroup(
        arrays=arrays, attributes=data.get("group_attributes", {})
    )
    return ManifestStore(group, registry=registry)


def load_cached_manifest(
    cache_dir: Path, url: str, ifd: int, registry: ObjectStoreRegistry
) -> ManifestStore | None:
    """Return a rebuilt ``ManifestStore`` from cache, or None on miss/stale/error.

    Never raises: any problem reading or parsing the cache is treated as a miss so
    the caller transparently falls back to a fresh parse.
    """
    path = cache_path_for(cache_dir, url, ifd)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("format_version") != CACHE_FORMAT_VERSION:
            logger.debug("manifest cache %s: stale format, ignoring", path.name)
            return None
        store = _jsonable_to_manifest(data, registry)
        logger.debug("manifest cache hit: %s", url)
        return store
    except Exception as exc:  # noqa: BLE001 — cache must never be fatal
        logger.warning("manifest cache %s unreadable (%s); reparsing", path.name, exc)
        return None


def save_manifest(
    cache_dir: Path, url: str, ifd: int, store: ManifestStore
) -> None:
    """Serialise ``store`` to the cache for (url, ifd). Best-effort; never raises.

    Writes atomically (temp file + rename) so a crash mid-write can't leave a
    half-written JSON that a later run would treat as a corrupt hit.
    """
    path = cache_path_for(cache_dir, url, ifd)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _manifest_to_jsonable(store)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, default=_json_default))
        tmp.replace(path)
        logger.debug("manifest cached: %s -> %s", url, path.name)
    except Exception as exc:  # noqa: BLE001 — caching is an optimisation, not a contract
        logger.warning("could not cache manifest for %s (%s)", url, exc)


def _json_default(obj):
    """Coerce the few non-JSON-native types that appear in manifest metadata.

    ChunkManifest.dict() offsets/lengths come back as numpy integers; array
    metadata may carry numpy scalars or tuples. Tuples already serialise as
    lists; this handles numpy scalars/arrays.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON-serialisable: {type(obj)!r}")
