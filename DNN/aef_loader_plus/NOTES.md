# aef_loader_plus — local fork with nyvest improvements

A local fork of [jakenotjay/aef-loader](https://github.com/jakenotjay/aef-loader)
carrying six changes ported from this repo's own AEF inference pipeline
(`DNN/fetch_aef_sourcecoop.py`, `DNN/prep_aef_tiles.py`, `DNN/build_vrt.py`).
Built and tested against the pixi-managed `geo` env (Python 3.12), which already
had every dependency except `virtual-tiff` (installed via `geo-pip`).

Forked from upstream commit `feff13e` (see `UPSTREAM_COMMIT.txt`). The package is
still importable as `aef_loader` (a `.pth` in the geo env's site-packages points
here), so it drops in wherever the upstream package would be used, and every edit
is exactly the diff we intend to PR.

## How to use it in the geo env

```python
# geo env: /home/geethen.singh/.pixi/envs/geo/bin/python
from aef_loader import AEFIndex, VirtualTiffReader, DataSource, aoi_geobox
from aef_loader.utils import reproject_datatree
```

The `.pth` file that wires it in:
`/home/geethen.singh/.pixi/envs/geo/lib/python3.12/site-packages/aef_loader_plus.pth`
(a single line containing this directory). Delete it to fall back to any
pip-installed `aef_loader`.

## Changes and measured results

All speed numbers were measured in the geo env against live source.coop tiles
over a small Norwegian AOI (zone 31N/32N, Rogaland + Bergen/Voss), 2024.

### A. `chunks=None` no longer materialises whole tiles  ← biggest win, and a latent upstream bug
`reader.py`. `open_tiles_by_zone` must add a length-1 `time` dimension
(`expand_dims`). On dask-backed arrays that's a free graph op, but on the
**numpy-backed arrays that `open_zarr(chunks=None)` returns it forces a full read
of the entire 8192² × 64 int8 tile (~100 s/tile measured)**. The upstream
docstring actively recommends `chunks=None` ("useful to pass None to stop dask
task explosions"), so this path is a footgun: it silently turns a metadata op
into a ~100 s/tile download. `concat` and `set_dims` have the same effect; only
plain windowed indexing stays lazy.

Fix: when `chunks is None`, open at the COG's **native block size** (one dask
chunk per stored 1024×1024 block, derived from the manifest via
`_native_block_chunks`). `expand_dims`/`concat` become graph ops again, and a
windowed read still fetches only the blocks it overlaps. Explicit `chunks`
values (`int`/`dict`/`"auto"`) pass through unchanged.

| `open_tiles_by_zone(chunks=None)`, 2 tiles | before | after |
|---|---:|---:|
| open + build cube | ~200 s | **1.97 s** |
| windowed read afterwards | (already read) | 0.60 s |

Data verified bit-identical to the `chunks="auto"` path.

### B. LUT dequantization (`dequantize_aef`)
`utils.py`. The plain-ndarray path now gathers through a precomputed 256-entry
int8→float32 LUT instead of recomputing `(v/127.5)²·sign(v)` per pixel, and
folds the `-128 → NaN` nodata step into the same gather (no separate mask/
temporaries). The dask-backed `DataArray` path is unchanged (stays lazy).

- **1.71× faster** on a 2048²×64 window (2702 ms → 1582 ms).
- Bit-exact vs the reference formula for all 256 int8 values (int8 and int16 in).

### C. Manifest caching (`cache.py`, new module; wired into `reader.py`)
Parsing a COG header into a VirtualiZarr `ManifestStore` costs ~1.6 s/tile and
never changes between sessions. `VirtualTiffReader(manifest_cache_dir=...)` now
serialises each tile's manifest to a small JSON (columnar paths/offsets/lengths
+ array metadata; no pixel data) and reloads it on later opens. Opt-in; default
behaviour (no cache) is unchanged.

- **103× faster** manifest rebuild (1.65 s parse → 15.9 ms load), cache file ~0.5 MB/tile.
- Reads verified bit-identical between fresh-parse and cache-load stores.
- Note: this removes the ~1.6 s/tile *parse*; item A removes the ~100 s/tile
  *materialisation*. They are independent — A dominates, C is the repeat-run win.

### D. Densified projected-bbox query (`AEFIndex.query(bbox_crs=...)`)
`index.py`. `query` gains a `bbox_crs` arg. When it's a projected CRS, the bbox
is reprojected to WGS84 with `Transformer.transform_bounds(densify_pts=21)`
before intersecting the index, so the envelope covers the projected rectangle's
outward-bowed edges instead of just its four corners (which under-covers at high
latitude and can drop boundary tiles). Default `"EPSG:4326"` is byte-identical
to the old behaviour.

- On a bowed 71°N strip, densify extends the WGS84 envelope 0.081° further north;
  on a fine synthetic grid it catches 2 edge tiles the corner-only bbox misses.
- On AEF's coarse ~110 km grid the extra coverage rarely tips a whole tile, so
  this is correctness insurance, not a guaranteed tile gain.

### E. Lattice-snapped GeoBox helper (`aoi_geobox`, new in `utils.py`)
`GeoBox.from_bbox` anchors the grid at the AOI corner, so two AOIs reprojected
independently land on different pixel grids and can't be mosaicked losslessly.
`aoi_geobox(bbox, crs, resolution, snap=True)` snaps the origin to the global
`resolution` lattice (anchored at 0,0) and grows the extent to cover the AOI —
the odc-geo analogue of `prep_aef_tiles.py`'s lattice snap.

- Two off-lattice, overlapping AOIs → both origins on the 10 m lattice, integer
  pixel offset between them, and 5000×3000 identical pixel coordinates in the
  overlap.

### F. Resampling guard on quantized data (`reproject_datatree`)
`utils.py`. Interpolating resamplers (`bilinear`/`cubic`/…) on raw int8 blend the
−128 sentinel into valid pixels and interpolate along the nonlinear quantization
curve — silently corrupt embeddings. `reproject_datatree` now raises on
`resampling != "nearest"` when the tree is integer-typed, with a message telling
you to dequantize first, and an `allow_lossy_resampling=True` escape hatch.

- Verified: int8+bilinear raises; int8+nearest, float+bilinear, and the override
  all pass.

### G. `combine_by_coords` keeps int8 (`_combine_tiles_single_zone`)
`reader.py`. The intra-zone `combine_by_coords(..., join="outer")` had no
`fill_value`, so when a zone's tiles don't fully tile the union rectangle the
NaN gap-fill promotes int8 → float64 (8× the dask-graph memory of the whole
zone, and the −128 sentinel is lost). Now passes `fill_value` in the data's own
dtype (−128 for int, NaN for float) via `_nodata_fill_for`.

- Verified: the combined 2-tile zone stays int8 (16384×8192) with nodata/
  _FillValue = −128; stock code promotes it to float32/float64.

## Test status
- Offline regression (9 checks, no cloud): all pass — run the snippet in
  "Consolidated offline regression" from the session, or see the individual
  tests below.
- Live-cloud checks (A–G above): all pass against source.coop.
- No upstream test file was modified; these are additive changes. Before the PR,
  port each check into `tests/` in the upstream layout (pytest, `asyncio_mode=auto`).

## Nothing parked
Every intended change landed and is tested. The one item that looked like it
might need parking — the `chunks=None` slowdown — turned out to be the
`expand_dims`-materialisation bug (item A) and is fixed, not deferred.

## Next step (per plan): open the PR
Suggested PR order, smallest/safest first: B (LUT) and D (bbox_crs) as
self-contained first PRs; then F (guard) and E (`aoi_geobox`); then C (manifest
cache); then A (the `chunks=None` fix) and G (fill_value) as the correctness/
performance fixes, each with a regression test. A is worth calling out to the
maintainer as a bug report on its own — the recommended `chunks=None` path is
~100 s/tile today.
