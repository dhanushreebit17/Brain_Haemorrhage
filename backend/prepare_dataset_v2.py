import argparse
import hashlib
import os
import random
import shutil
from collections import defaultdict

VALID_EXTS = (".png", ".jpg", ".jpeg")


def file_hash(path, block_size=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True,
                         help="Path to the folder containing one subfolder per class "
                              "(e.g. '...\\BE DATASET\\train')")
    parser.add_argument("--out_dir", type=str, default="dataset")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    class_folders = [d for d in os.listdir(args.raw_dir)
                      if os.path.isdir(os.path.join(args.raw_dir, d))]
    if not class_folders:
        raise FileNotFoundError(f"No subfolders found under {args.raw_dir}")
    print(f"[prepare_v2] Found class folders: {class_folders}")

    random.seed(args.seed)
    counts = defaultdict(int)
    total_dupes = 0

    for class_name in class_folders:
        src_dir = os.path.join(args.raw_dir, class_name)
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(VALID_EXTS)]

        # Deduplicate by content hash (catches "xxx.png" vs "xxx - Copy.png")
        seen_hashes = {}
        unique_files = []
        for f in files:
            h = file_hash(os.path.join(src_dir, f))
            if h in seen_hashes:
                total_dupes += 1
                continue
            seen_hashes[h] = f
            unique_files.append(f)

        random.shuffle(unique_files)
        n_val = max(1, round(len(unique_files) * args.val_fraction))
        val_files = set(unique_files[:n_val])

        for f in unique_files:
            split = "val" if f in val_files else "train"
            dest_dir = os.path.join(args.out_dir, split, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copyfile(os.path.join(src_dir, f), os.path.join(dest_dir, f))
            counts[(split, class_name)] += 1

        print(f"[prepare_v2] {class_name}: {len(files)} files -> "
              f"{len(unique_files)} unique ({len(files) - len(unique_files)} duplicates removed)")

    print(f"\n[prepare_v2] Total duplicate files removed: {total_dupes}")
    print("[prepare_v2] Done. Class distribution:")
    for (split, cls), n in sorted(counts.items()):
        print(f"    {split}/{cls}: {n} images")
    print(f"[prepare_v2] Dataset written to: {args.out_dir}")
    print(f"[prepare_v2] Now run: python train.py --data_dir {args.out_dir} --epochs 15 --batch_size 16")


if __name__ == "__main__":
    main()