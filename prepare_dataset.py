from pathlib import Path
from PIL import Image
import argparse

"""prepare_dataset.py

Standardize images for training:
- converts to RGB
- resizes to a fixed size (default 224x224)
- saves as JPEG with consistent filenames
- produces output in `dataset/train/<class>` and `dataset/val/<class>`

Usage examples:
python prepare_dataset.py --source path/to/raw_images --out dataset --size 224

Source layout supported:
- source/train/<class>/*.jpg|png|...  (keeps split)
- source/val/<class>/*.*
- OR source/<class>/*.*  (will create train/ and val/ using 80/20 split)
"""

DEFAULT_CLASSES = ["cola", "orange_juice", "water"]


def standardize_image(in_path: Path, out_path: Path, size: int = 224, quality: int = 95) -> bool:
    try:
        with Image.open(in_path) as im:
            im = im.convert("RGB")
            im = im.resize((size, size), Image.LANCZOS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_file = out_path.with_suffix('.jpg')
            im.save(out_file, format='JPEG', quality=quality, optimize=True)
        return True
    except Exception as e:
        print(f"[ERROR] {in_path} -> {e}")
        return False


def process_split(source_dir: Path, dest_dir: Path, classes, size: int, ext: str = '.jpg'):
    counts = {}
    for cls in classes:
        # try both layouts: source/train/cls or source/cls
        src1 = source_dir / 'train' / cls
        src2 = source_dir / cls
        src = src1 if src1.exists() else src2
        files = list(src.glob('*')) if src.exists() else []
        counts[cls] = 0
        for idx, f in enumerate(sorted(files), start=1):
            out_name = f"{cls}_{idx}{ext}"
            out_path = dest_dir / 'train' / cls / out_name
            ok = standardize_image(f, out_path, size=size)
            if ok:
                counts[cls] += 1
    return counts


def split_and_process_flat(source_dir: Path, dest_dir: Path, classes, size: int, train_ratio: float = 0.8):
    # when source has source/<class>/*, split into train/ and val/
    counts = {"train": {}, "val": {}}
    for cls in classes:
        src = source_dir / cls
        files = sorted(list(src.glob('*'))) if src.exists() else []
        n = len(files)
        n_train = int(n * train_ratio)
        for i, f in enumerate(files):
            split = 'train' if i < n_train else 'val'
            idx = (i + 1) if split == 'train' else (i - n_train + 1)
            out_name = f"{cls}_{idx}.jpg"
            out_path = dest_dir / split / cls / out_name
            ok = standardize_image(f, out_path, size=size)
            if ok:
                counts[split].setdefault(cls, 0)
                counts[split][cls] += 1
    return counts


def main():
    p = argparse.ArgumentParser(description='Standardize images for training')
    p.add_argument('--source', type=str, required=True, help='Path to source images')
    p.add_argument('--out', type=str, default='juice_standarized', help='Output dataset folder')
    p.add_argument('--size', type=int, default=224, help='Output image size (square)')
    p.add_argument('--classes', type=str, default=','.join(DEFAULT_CLASSES), help='Comma-separated class names')
    p.add_argument('--train-ratio', type=float, default=0.8, help='Train/val split when needed')
    p.add_argument('--no-split', action='store_true', help='Do not split into train/val; save all images into out/<class>/')
    args = p.parse_args()

    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    classes = [c.strip() for c in args.classes.split(',') if c.strip()]

    # determine layout
    # if user requests no_split, put all images into out/<class>/ (no train/val)
    if args.no_split:
        print('Saving all standardized images into out/<class>/ (no split)')
        # look for source/<class> or source/train/<class>
        total_counts = {}
        for cls in classes:
            src_a = source / cls
            src_b = source / 'train' / cls
            src = src_a if src_a.exists() else src_b if src_b.exists() else None
            files = sorted(list(src.glob('*'))) if src else []
            cnt = 0
            for idx, f in enumerate(files, start=1):
                out_name = f"{cls}_{idx}.jpg"
                out_path = out / cls / out_name
                ok = standardize_image(f, out_path, size=args.size)
                if ok:
                    cnt += 1
            total_counts[cls] = cnt
            print(f"{cls}: {cnt}")
        print('Done.')
    else:
        # if source contains train/ and val/ use those; otherwise expect source/<class>/*
        if (source / 'train').exists() and (source / 'val').exists():
            print('Detected split folders under source. Processing train/ and val/...')
            train_counts = process_split(source, out, classes, size=args.size)
            # now handle val similarly
            for cls in classes:
                src = source / 'val' / cls
                files = list(src.glob('*')) if src.exists() else []
                cnt = 0
                for idx, f in enumerate(sorted(files), start=1):
                    out_name = f"{cls}_{idx}.jpg"
                    out_path = out / 'val' / cls / out_name
                    ok = standardize_image(f, out_path, size=args.size)
                    if ok:
                        cnt += 1
                print(f"val {cls}: {cnt}")
            for k, v in train_counts.items():
                print(f"train {k}: {v}")
        else:
            print('No train/val split detected under source, assuming source/<class>/* — splitting into train/ and val/')
            counts = split_and_process_flat(source, out, classes, size=args.size, train_ratio=args.train_ratio)
            print('Counts by split:')
            print(counts)


if __name__ == '__main__':
    main()
