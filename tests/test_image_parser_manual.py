#!/usr/bin/env python3
"""Manual test script — runs the image parser against real example data.

Usage:
    python -m tests.test_image_parser_manual

Scans data/examples/ and parses:
  - train_lora/: only .txt files  (training captions)
  - output/:     only .png/.jpg   (generated images with embedded prompts)

Prints exactly what was or wasn't extracted from each file.
"""

import os
import sys

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.image_parser import parse_file

EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "examples"
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def scan_directory(directory: str, dir_type: str) -> list[str]:
    """Find parseable files depending on directory type."""
    files = []
    for root, _dirs, filenames in os.walk(directory):
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if dir_type == "training" and ext == ".txt":
                files.append(os.path.join(root, fname))
            elif dir_type == "output" and ext in IMAGE_EXTENSIONS:
                files.append(os.path.join(root, fname))
    return files


def main():
    if not os.path.isdir(EXAMPLES_DIR):
        print(f"ERROR: Examples directory not found: {EXAMPLES_DIR}")
        sys.exit(1)

    training_dir = os.path.join(EXAMPLES_DIR, "train_lora")
    output_dir = os.path.join(EXAMPLES_DIR, "output")

    training_files = scan_directory(training_dir, "training") if os.path.isdir(training_dir) else []
    output_files = scan_directory(output_dir, "output") if os.path.isdir(output_dir) else []

    all_files = [("training", f) for f in training_files] + [("output", f) for f in output_files]

    print(f"Found {len(training_files)} training .txt files")
    print(f"Found {len(output_files)} output image files")
    print(f"Total: {len(all_files)} files to parse\n")
    print("=" * 80)

    success = 0
    fail = 0
    counts = {"txt": [0, 0], "png": [0, 0], "jpg": [0, 0]}  # [ok, total]

    for dir_type, filepath in all_files:
        rel = os.path.relpath(filepath, EXAMPLES_DIR)
        ext = os.path.splitext(filepath)[1].lower().lstrip(".")
        if ext == "jpeg":
            ext = "jpg"

        counts.setdefault(ext, [0, 0])
        counts[ext][1] += 1

        print(f"\n--- [{ext.upper()}] [{dir_type}] {rel} ---")
        result = parse_file(filepath)

        if result is None:
            print("  ❌ FAILED: No data extracted")
            fail += 1
        else:
            success += 1
            counts[ext][0] += 1
            prompt_preview = result.prompt[:120]
            print(f"  ✅ Prompt: {prompt_preview}{'...' if len(result.prompt) > 120 else ''}")
            print(f"  📦 Base Model: {result.base_model or '(none)'}")
            print(f"  🔗 LoRAs: {', '.join(result.loras) if result.loras else '(none)'}")
            if result.negative_prompt:
                print(f"  🚫 Negative: {result.negative_prompt[:80]}{'...' if len(result.negative_prompt) > 80 else ''}")
            else:
                print(f"  🚫 Negative: (none)")

    print("\n" + "=" * 80)
    print(f"\nSUMMARY:")
    print(f"  Total files:  {len(all_files)}")
    print(f"  Successful:   {success}")
    print(f"  Failed:       {fail}")
    for ext, (ok, total) in sorted(counts.items()):
        if total:
            print(f"  {ext.upper()} files: {ok}/{total} parsed")

    if fail:
        print(f"\n⚠️  {fail} file(s) could not be parsed.")
    else:
        print(f"\n🎉 All files parsed successfully!")


if __name__ == "__main__":
    main()
