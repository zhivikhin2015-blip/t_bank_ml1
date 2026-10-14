"""Извлечение публичных входов из сохранённого архива без загрузки сети."""
import argparse
from pathlib import Path
import shutil
import zipfile
from .features import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / "reports/finance_ml_audit/source_inputs/work/data"
    if args.out.exists():
        raise FileExistsError("Choose a new data directory")
    args.out.mkdir(parents=True)
    archive = source.parents[1] / "FAR-Trans.zip"
    assert sha256(archive) == "d0d6cfe027732bab45fdcf9af80eff4fefddc789aedae9bdce0d3d6cfd05cfb6"
    with zipfile.ZipFile(archive) as z:
        matches = [n for n in z.namelist() if n.endswith("transactions.csv")]
        assert len(matches) == 1
        with z.open(matches[0]) as src, (args.out / "transactions.csv").open("wb") as dst:
            shutil.copyfileobj(src, dst)
    assert sha256(args.out / "transactions.csv") == "358871fa1311a802799df86e56c3d633513566965b01412d058e0f8fbe6cd54b"
    shutil.copy2(source / "FAR-Trans/asset_information.csv", args.out)
    print("Verified FAR-Trans inputs prepared")


if __name__ == "__main__":
    main()
