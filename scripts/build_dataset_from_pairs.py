#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Organize paired captures into training layout")
    parser.add_argument("--source", required=True, help="Folder containing class/individual subfolders with ply files")
    parser.add_argument("--output", required=True, help="Output dataset root")
    return parser.parse_args()


def main():
    args = parse_args()
    src = Path(args.source)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for class_dir in src.iterdir():
        if not class_dir.is_dir():
            continue
        for indiv_dir in class_dir.iterdir():
            if not indiv_dir.is_dir():
                continue
            dest_samples = out / class_dir.name / indiv_dir.name / "samples"
            dest_samples.mkdir(parents=True, exist_ok=True)
            for ply in indiv_dir.glob("*.ply"):
                shutil.copy(ply, dest_samples / ply.name)
    print(f"Dataset organized under {out}")


if __name__ == "__main__":
    main()

