"""Подготовить проверенные входы FAR-Trans из неизменяемого публичного архива."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
from .features import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    archive = args.archive or root / "data/raw/FAR-Trans.zip"
    manifest = json.loads((args.manifest or root / "data/manifest.json").read_text())
    if sha256(archive) != manifest["downloaded_archive_sha256"]:
        raise ValueError("Archive SHA-256 mismatch")
    if args.out.exists():
        raise FileExistsError("Choose a new data directory")
    args.out.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for filename, expected in manifest["source_file_sha256"].items():
            matches = [n for n in z.namelist() if n.endswith("/" + filename)]
            if len(matches) != 1:
                raise ValueError(f"Expected one archive member for {filename}")
            target = args.out / filename
            with z.open(matches[0]) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if sha256(target) != expected:
                raise ValueError(f"Input SHA-256 mismatch: {filename}")
    print("Verified FAR-Trans inputs prepared")


if __name__ == "__main__":
    main()
