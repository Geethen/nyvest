"""Shared paths for DNN scripts and generated artifacts."""

from pathlib import Path


DNN_DIR = Path(__file__).resolve().parent
REPO_DIR = DNN_DIR.parent

REPORTS_DIR = DNN_DIR / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
LOGS_DIR = REPORTS_DIR / "logs"
FIGURES_DIR = REPORTS_DIR / "figures"
HTML_DIR = REPORTS_DIR / "html"


def artifact_path(*parts: str | Path) -> Path:
    path = REPORTS_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def result_path(filename: str) -> Path:
    return artifact_path("results", filename)


def figure_path(filename: str) -> Path:
    return artifact_path("figures", filename)


def html_path(filename: str) -> Path:
    return artifact_path("html", filename)