#!/usr/bin/env python3
"""Automated developer test runner for Halı Saha Video Analysis.

Automatically detects active or installed Python virtual environments (.venv-boxmot,
.venv-train, .venv) and executes the unittest suite with proper PYTHONPATH setup.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_best_python() -> Path:
    candidates = [
        ROOT / ".venv-boxmot/bin/python",
        ROOT / ".venv-train/bin/python",
        ROOT / ".venv/bin/python",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return Path(sys.executable)


def main():
    parser = argparse.ArgumentParser(description="Halı saha test süitini çalıştır")
    parser.add_argument("-v", "--verbose", action="store_true", help="Detaylı test çıktısı göster")
    parser.add_argument("-p", "--pattern", default="test_*.py", help="Çalıştırılacak test dosya deseni (ör. test_calibration*.py)")
    parser.add_argument("--python", type=Path, help="Özel Python yorumlayıcı yolu")
    args = parser.parse_args()

    python_bin = args.python.resolve() if args.python else find_best_python()
    print(f"[run_tests] Python ortamı: {python_bin}")

    env = dict(os.environ)
    src_dir = str(ROOT / "src")
    env["PYTHONPATH"] = f"{src_dir}:{env.get('PYTHONPATH', '')}"

    cmd = [
        str(python_bin),
        "-m",
        "unittest",
        "discover",
        "-s",
        str(ROOT / "tests"),
        "-p",
        args.pattern,
    ]
    if args.verbose:
        cmd.append("-v")

    print(f"[run_tests] Çalıştırılıyor: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
