#!/usr/bin/env python3
"""
Batch calibrate using the images captured by camera_calibration.exe.

The script scans calibration/images/ for chessboard photos (calib_*.jpg by default),
detects the 9x6 inner corners, and runs cv.calibrateCamera without requiring the
interactive capture flow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


BOARD_SIZE: Tuple[int, int] = (9, 6)  # inner corners (columns, rows)
SQUARE_SIZE_MM: float = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate using saved chessboard photos instead of live capture."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "images",
        help="Directory that contains calib_*.jpg images (default: calibration/images)",
    )
    parser.add_argument(
        "--pattern",
        default="calib_*.jpg",
        help="Glob pattern for calibration images (default: calib_*.jpg)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "camera_calibration_from_images.yaml",
        help="Path of the YAML file to store calibration results.",
    )
    return parser.parse_args()


def prepare_object_points() -> np.ndarray:
    """Pre-compute the chessboard coordinates in millimeters."""
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : BOARD_SIZE[0], 0 : BOARD_SIZE[1]].T.reshape(-1, 2)
        * SQUARE_SIZE_MM
    )
    return objp


def load_image_paths(image_dir: Path, pattern: str) -> List[Path]:
    paths = sorted(image_dir.glob(pattern))
    return [p for p in paths if p.is_file()]


def main() -> int:
    args = parse_args()
    image_paths = load_image_paths(args.image_dir, args.pattern)

    if not image_paths:
        print(f"No images found in {args.image_dir} matching '{args.pattern}'.")
        return 1

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    objp = prepare_object_points()
    image_size: Tuple[int, int] | None = None

    print(f"Found {len(image_paths)} candidate images, scanning for chessboards...")
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  [SKIP] Unable to read {path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            BOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print(f"  [FAIL] {path.name}: chessboard not detected")
            continue

        cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
        )

        obj_points.append(objp.copy())
        img_points.append(corners)
        image_size = (gray.shape[1], gray.shape[0])
        print(f"  [OK]   {path.name}")

    if len(img_points) < 10:
        print(f"Need at least 10 valid images, got {len(img_points)}.")
        return 1

    assert image_size is not None
    print(f"\nRunning calibration with {len(img_points)} images...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
        flags=cv2.CALIB_FIX_PRINCIPAL_POINT,
    )

    print("\n=== Calibration Results ===")
    print(f"Image size: {image_size[0]} x {image_size[1]}")
    print(f"RMS reprojection error: {rms:.4f}")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs.ravel())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(args.output), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("board_width", BOARD_SIZE[0])
    fs.write("board_height", BOARD_SIZE[1])
    fs.write("square_size_mm", SQUARE_SIZE_MM)
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("rms_reprojection_error", rms)
    fs.release()

    print(f"\nSaved calibration to {args.output}")
    print("Update config/camera_config.yaml with:")
    print(f"  fx={camera_matrix[0,0]:.6f}, fy={camera_matrix[1,1]:.6f}")
    print(f"  cx={camera_matrix[0,2]:.6f}, cy={camera_matrix[1,2]:.6f}")

    coeffs = dist_coeffs.ravel()
    padded = np.zeros(5, dtype=np.float64)
    padded[: min(5, coeffs.size)] = coeffs[: min(5, coeffs.size)]
    print(
        "  k1={:.6f}, k2={:.6f}, p1={:.6f}, p2={:.6f}, k3={:.6f}".format(
            padded[0], padded[1], padded[2], padded[3], padded[4]
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
