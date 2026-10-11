"""Verify release model files against the repository SHA-256 manifest."""
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    failed = False
    for line in (ROOT / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = args.models_dir / name
        if not path.is_file():
            print(f"MISSING: {name}")
            failed = True
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        ok = digest.hexdigest() == expected
        print(f"{'OK' if ok else 'MISMATCH'}: {name}")
        failed |= not ok
    raise SystemExit(1 if failed else 0)

if __name__ == "__main__":
    main()
