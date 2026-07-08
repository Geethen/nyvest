"""Run one Stage-3 ablation: reuse stage3 main logic, tag the output JSON.

Usage: ablation_one.py <tag>   (config comes from env vars, set by run_ablation.sh)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage3_robust_mlp as s3  # noqa: E402
from dnn_paths import result_path  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "untagged"
s3.OUT_JSON = result_path(f"ablation_{tag}.json")
s3.main()
