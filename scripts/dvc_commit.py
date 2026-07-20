"""DVC helper: track a given path with DVC and push to the configured
remote. Used as an Airflow task after the Spark preprocessing step so
each new version of the processed dataset is captured by DVC.
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.returncode != 0:
        logger.error(result.stderr.strip())
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Track paths with DVC and push to remote")
    parser.add_argument("paths", nargs="+", help="File paths to `dvc add`")
    args = parser.parse_args()

    run(["dvc", "add", *args.paths])
    run(["dvc", "push"])
    logger.info("DVC tracked and pushed: %s", args.paths)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logger.error("dvc_commit failed: %s", exc)
        sys.exit(1)
