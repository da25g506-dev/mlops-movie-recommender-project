"""Download and validate the MovieLens ml-1m dataset.

Downloads the official ml-1m archive from GroupLens, verifies its MD5
checksum, and extracts ratings.dat / movies.dat / users.dat into
data/raw/.
"""
import hashlib
import logging
import sys
import zipfile
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
EXPECTED_MD5 = "c4d9eecfca2ab87c1945afe126590906"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "ml-1m.zip"
EXPECTED_FILES = ["ratings.dat", "movies.dat", "users.dat"]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_zip(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and not force:
        logger.info("Archive already present at %s, skipping download", ZIP_PATH)
        return ZIP_PATH

    logger.info("Downloading %s", DATASET_URL)
    response = requests.get(DATASET_URL, stream=True, timeout=60)
    response.raise_for_status()
    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    logger.info("Saved archive to %s (%d bytes)", ZIP_PATH, ZIP_PATH.stat().st_size)
    return ZIP_PATH


def verify_checksum(zip_path: Path) -> None:
    actual = _md5(zip_path)
    if actual != EXPECTED_MD5:
        raise ValueError(
            f"Checksum mismatch for {zip_path}: expected {EXPECTED_MD5}, got {actual}"
        )
    logger.info("Checksum verified: %s", actual)


def extract(zip_path: Path) -> None:
    logger.info("Extracting %s to %s", zip_path, RAW_DIR)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            filename = Path(member).name
            if filename in EXPECTED_FILES:
                target = RAW_DIR / filename
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                logger.info("Extracted %s (%d bytes)", target, target.stat().st_size)


def validate_extracted() -> None:
    missing = [f for f in EXPECTED_FILES if not (RAW_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected raw files after extraction: {missing}")
    for f in EXPECTED_FILES:
        size = (RAW_DIR / f).stat().st_size
        if size == 0:
            raise ValueError(f"Extracted file {f} is empty")
        logger.info("Validated %s (%d bytes)", f, size)


def main() -> None:
    zip_path = download_zip()
    verify_checksum(zip_path)
    extract(zip_path)
    validate_extracted()
    logger.info("Dataset ready in %s", RAW_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logger.error("Data download failed: %s", exc)
        sys.exit(1)
