import argparse
import os

import numpy as np
import pydicom
from PIL import Image


def apply_window(img: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(img, img_min, img_max)
    windowed = (windowed - img_min) / (img_max - img_min) * 255.0
    return windowed.astype(np.uint8)


def dicom_to_png(dicom_path: str, out_path: str,
                  window_center: int = 40, window_width: int = 80):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)

    # Convert to Hounsfield Units using rescale slope/intercept from the DICOM header
    slope = getattr(ds, "RescaleSlope", 1)
    intercept = getattr(ds, "RescaleIntercept", 0)
    hu = img * slope + intercept

    windowed = apply_window(hu, window_center, window_width)
    Image.fromarray(windowed).convert("RGB").save(out_path)


def convert_directory(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".dcm")]
    print(f"[dicom_to_png] Found {len(files)} DICOM files.")

    for i, fname in enumerate(files):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname.replace(".dcm", ".png"))
        try:
            dicom_to_png(in_path, out_path)
        except Exception as e:
            print(f"  [skip] {fname}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  ...converted {i+1}/{len(files)}")

    print("[dicom_to_png] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    convert_directory(args.input_dir, args.output_dir)
