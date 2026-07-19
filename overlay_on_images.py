"""Create single-channel vessel-guided NIfTI images from images and masks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a vessel-guidance transform to each image/mask pair."
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
        help="Directory containing corresponding vessel masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where processed .nii files will be saved.",
    )
    parser.add_argument(
        "--method",
        choices=("multiply", "add", "replace", "mask", "distance", "frangi"),
        default="multiply",
        help=(
            "Vessel feature method. multiply/add modify intensities inside mask; "
            "replace sets mask voxels to a value; mask saves the binary mask; "
            "distance saves a normalized distance map; frangi saves vesselness."
        ),
    )
    parser.add_argument(
        "--multiply-factor",
        type=float,
        default=1.15,
        help="Multiplier for method=multiply.",
    )
    parser.add_argument(
        "--add-value",
        type=float,
        default=200.0,
        help="Value added inside mask for method=add.",
    )
    parser.add_argument(
        "--replace-value",
        type=float,
        default=1.0,
        help="Value written inside mask for method=replace.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.0,
        help="Mask values greater than this threshold are treated as vessel.",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=-1024.0,
        help="Minimum image intensity before/after intensity transforms.",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=3000.0,
        help="Maximum image intensity before/after intensity transforms.",
    )
    parser.add_argument(
        "--normalize",
        choices=("zscore", "minmax", "none"),
        default="zscore",
        help="Output normalization.",
    )
    parser.add_argument(
        "--frangi-sigmas",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 3.0),
        help="Sigma values for method=frangi.",
    )
    parser.add_argument(
        "--frangi-alpha",
        type=float,
        default=0.5,
        help="Frangi alpha parameter.",
    )
    parser.add_argument(
        "--frangi-beta",
        type=float,
        default=0.5,
        help="Frangi beta parameter.",
    )
    parser.add_argument(
        "--frangi-gamma",
        type=float,
        default=None,
        help="Frangi gamma parameter. Default lets scikit-image choose.",
    )
    parser.add_argument(
        "--frangi-black-ridges",
        action="store_true",
        help="Detect dark ridges instead of bright ridges for method=frangi.",
    )
    parser.add_argument(
        "--frangi-mask-output",
        action="store_true",
        help="Zero Frangi output outside the provided vessel mask.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files that already exist.",
    )
    return parser.parse_args()


def find_mask(case_id: str, mask_dir: Path) -> Path | None:
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


def get_case_id(filename: str) -> str:
    name = filename.replace(".nii.gz", "").replace(".nii", "")
    if name.endswith("_0000"):
        return name[:-5]
    return name


def normalize(data: np.ndarray, mode: str) -> np.ndarray:
    data = data.astype(np.float32, copy=False)
    if mode == "none":
        return data
    if mode == "zscore":
        return (data - float(np.mean(data))) / (float(np.std(data)) + 1e-8)
    if mode == "minmax":
        min_value = float(np.min(data))
        max_value = float(np.max(data))
        return (data - min_value) / (max_value - min_value + 1e-8)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def normalized_distance(mask_binary: np.ndarray) -> np.ndarray:
    distance = distance_transform_edt(mask_binary).astype(np.float32)
    max_value = float(distance.max())
    if max_value > 0:
        distance /= max_value
    return distance


def frangi_vesselness(
    image: np.ndarray,
    mask_binary: np.ndarray,
    sigmas: tuple[float, ...],
    alpha: float,
    beta: float,
    gamma: float | None,
    black_ridges: bool,
    mask_output: bool,
) -> np.ndarray:
    from skimage.filters import frangi

    image_min = float(np.min(image))
    image_max = float(np.max(image))
    scaled = (image - image_min) / (image_max - image_min + 1e-8)
    vesselness = frangi(
        scaled,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
    ).astype(np.float32)
    if mask_output:
        vesselness *= mask_binary.astype(np.float32)
    return vesselness


def apply_method(
    image: np.ndarray,
    mask_binary: np.ndarray,
    args,
) -> np.ndarray:
    image = np.clip(image.astype(np.float32), args.clip_min, args.clip_max)

    if args.method == "multiply":
        output = image.copy()
        output[mask_binary] *= args.multiply_factor
        return np.clip(output, args.clip_min, args.clip_max)

    if args.method == "add":
        output = image.copy()
        output[mask_binary] += args.add_value
        return np.clip(output, args.clip_min, args.clip_max)

    if args.method == "replace":
        output = image.copy()
        output[mask_binary] = args.replace_value
        return output

    if args.method == "mask":
        return mask_binary.astype(np.float32)

    if args.method == "distance":
        return normalized_distance(mask_binary)

    if args.method == "frangi":
        return frangi_vesselness(
            image=image,
            mask_binary=mask_binary,
            sigmas=tuple(args.frangi_sigmas),
            alpha=args.frangi_alpha,
            beta=args.frangi_beta,
            gamma=args.frangi_gamma,
            black_ridges=args.frangi_black_ridges,
            mask_output=args.frangi_mask_output,
        )

    raise ValueError(f"Unsupported method: {args.method}")


def process_one(img_path: Path, args):
    case_id = get_case_id(img_path.name)
    out_path = args.output_dir / f"{case_id}_0000.nii"

    if out_path.exists() and not args.overwrite:
        return "skipped", None

    try:
        img_nii = nib.load(str(img_path))
        img_data = img_nii.get_fdata().astype(np.float32)
        mask_path = find_mask(case_id, args.mask_dir)

        if mask_path is None:
            output_data = normalize(
                np.clip(img_data, args.clip_min, args.clip_max), args.normalize
            )
            nib.save(
                nib.Nifti1Image(output_data.astype(np.float32), img_nii.affine, img_nii.header),
                str(out_path),
            )
            return "no_mask", None

        mask_data = nib.load(str(mask_path)).get_fdata()
        if img_data.shape != mask_data.shape:
            message = f"Shape mismatch for {case_id}: {img_data.shape} != {mask_data.shape}"
            return "error", message

        mask_binary = mask_data > args.mask_threshold
        output_data = apply_method(img_data, mask_binary, args)
        output_data = normalize(output_data, args.normalize)

        nib.save(
            nib.Nifti1Image(output_data.astype(np.float32), img_nii.affine, img_nii.header),
            str(out_path),
        )
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
    if args.clip_max <= args.clip_min:
        raise ValueError("--clip-max must be greater than --clip-min.")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1.")
    if args.method == "frangi":
        if len(args.frangi_sigmas) < 1:
            raise ValueError("--frangi-sigmas must contain at least one value.")
        if any(sigma <= 0 for sigma in args.frangi_sigmas):
            raise ValueError("--frangi-sigmas values must be greater than 0.")


def iter_image_files(images_dir: Path) -> list[Path]:
    return sorted(list(images_dir.glob("*.nii.gz")) + list(images_dir.glob("*.nii")))


def main():
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_files = iter_image_files(args.images_dir)
    print(f"Found {len(image_files)} input images.")
    print(f"Method: {args.method}")
    print(f"Normalization: {args.normalize}")

    counts = {"success": 0, "no_mask": 0, "skipped": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_one, image_path, args): image_path for image_path in image_files}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Vessel feature: {args.method}",
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
