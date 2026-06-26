# nyvest

Geospatial land-cover modelling on the nyvest FSCS feature space (AlphaEarth embeddings + LiDAR). See [CLAUDE.md](CLAUDE.md) for environment + project conventions.

## Reports

- [reports/benchmark_cv_blocked_report.html](reports/benchmark_cv_blocked_report.html) — HTML summary of the 3-fold spatially-blocked CV runs (leaderboard, capped vs. uncapped foundation models, per-fold stability, takeaways).

### Open the blocked-CV report in a browser

```bash
# Linux (VDI)
xdg-open reports/benchmark_cv_blocked_report.html

# macOS
open reports/benchmark_cv_blocked_report.html

# Windows (Git Bash / PowerShell)
start reports/benchmark_cv_blocked_report.html
```

Or serve the `reports/` directory locally and browse to it:

```bash
~/myprojects/recover/.venv/bin/python -m http.server --directory reports 8000
# then open http://localhost:8000/benchmark_cv_blocked_report.html
```