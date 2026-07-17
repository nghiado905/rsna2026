"""Enhance vessel intensity inside a mask and normalize NIfTI volumes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Multiply voxel intensities inside a mask, clip the result, and apply "
            "z-score normalization to each NIfTI volume."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing input .nii or .nii.gz images.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Directory containing the corresponding masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which processed images will be saved.",
    )
    parser.add_argument(
        "--multiply-factor",
        type=float,
        default=1.15,
        help="Intensity multiplier inside the mask (default: 1.15).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files that already exist.",
    )
    return parser.parse_args()


def find_mask(case_id, mask_dir):
    candidates = [
        f"{case_id}.nii.gz",
        f"{case_id}.nii",
        f"{case_id}_0000.nii.gz",
        f"{case_id}_0000.nii",
    ]
    for filename in candidates:
        path = mask_dir / filename
        if path.exists():
            return path
    return None


def get_case_id(filename):
    name = filename.replace(".nii.gz", "").replace(".nii", "")
    if name.endswith("_0000"):
        return name[:-5]
    return name


def normalize(data):
    mean = np.mean(data)
    std = np.std(data) + 1e-8
    return (data - mean) / std


def process_one(img_path, mask_dir, output_dir, multiply_factor, overwrite):
    case_id = get_case_id(img_path.name)
    out_path = output_dir / f"{case_id}_0000.nii"

    if out_path.exists() and not overwrite:
        return "skipped", None

    try:
        img_nii = nib.load(str(img_path))
        img_data = img_nii.get_fdata().astype(np.float32)

        # Keep CT values within the expected Hounsfield unit range.
        img_data = np.clip(img_data, -1024, 3000)
        mask_path = find_mask(case_id, mask_dir)

        if mask_path is None:
            # Normalize and save the image even when no corresponding mask exists.
            output_data = normalize(img_data)
            new_img = nib.Nifti1Image(
                output_data.astype(np.float32), img_nii.affine, img_nii.header
            )
            nib.save(new_img, str(out_path))
            return "no_mask", None

        mask_data = nib.load(str(mask_path)).get_fdata()
        if img_data.shape != mask_data.shape:
            nib.save(img_nii, str(out_path))
            message = f"Shape mismatch for {case_id}: {img_data.shape} != {mask_data.shape}"
            return "error", message

        # Treat every positive mask voxel as part of the vessel region.
        mask_binary = mask_data > 0
        enhanced = img_data.copy()
        enhanced[mask_binary] *= multiply_factor

        # Clip again after multiplication to prevent values outside the CT range.
        enhanced = np.clip(enhanced, -1024, 3000)
        output_data = normalize(enhanced)

        new_img = nib.Nifti1Image(
            output_data.astype(np.float32), img_nii.affine, img_nii.header
        )
        nib.save(new_img, str(out_path))
        return "success", None
    except Exception as exc:
        return "error", f"Critical error for {img_path.name}: {exc}"


def validate_args(args):
    if not args.images_dir.is_dir():
        raise ValueError(f"Images directory does not exist: {args.images_dir}")
    if not args.mask_dir.is_dir():
        raise ValueError(f"Mask directory does not exist: {args.mask_dir}")
    if args.multiply_factor <= 0:
        raise ValueError("--multiply-factor must be greater than 0.")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1.")


def main():
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(args.images_dir.glob("*.nii.gz")) + list(args.images_dir.glob("*.nii"))
    )
    print(f"Found {len(image_files)} input images.")

    counts = {"success": 0, "no_mask": 0, "skipped": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_one,
                image_path,
                args.mask_dir,
                args.output_dir,
                args.multiply_factor,
                args.overwrite,
            ): image_path
            for image_path in image_files
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Mask-guided intensity multiplication",
        ):
            status, message = future.result()
            counts[status] += 1
            if message:
                print(f"\n[ERROR] {message}")

    print("\n" + "=" * 50)
    print("Completed")
    print(f"Total images: {len(image_files)}")
    print(f"Processed with a mask: {counts['success']}")
    print(f"Processed without a mask: {counts['no_mask']}")
    print(f"Skipped existing outputs: {counts['skipped']}")
    print(f"Errors: {counts['error']}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
