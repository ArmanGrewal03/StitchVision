import cv2
import numpy as np
import time
import csv
import argparse
import os
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

ORIG_IMG_DIR = PROJECT_ROOT / "original_images"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_IMG1 = ORIG_IMG_DIR / "src_left.jpg"
DEFAULT_IMG2 = ORIG_IMG_DIR / "src_right.jpg"
DEFAULT_OUTPUT_DIR = RESULTS_DIR

os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

import stitch_orb_bf as orb_bf_module
import stitch_orb_flann_lsh as orb_flann_module
import stitch_sift_bf as sift_bf_module
import stitch_sift_flann as sift_flann_module


def run_orb_bf(left, right, nfeatures=4000, ratio_threshold=0.75):
    start_time = time.perf_counter()

    feat_start = time.perf_counter()
    kp_left, des_left = orb_bf_module.detect_features(left, nfeatures=nfeatures)
    kp_right, des_right = orb_bf_module.detect_features(right, nfeatures=nfeatures)
    feat_time = (time.perf_counter() - feat_start) * 1000

    if des_left is None or des_right is None:
        return None

    num_kp1 = len(kp_left)
    num_kp2 = len(kp_right)

    match_start = time.perf_counter()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn_matches = bf.knnMatch(des_left, des_right, k=2)
        num_matches = len(knn_matches)
    except Exception:
        num_matches = 0

    good_matches = orb_bf_module.match_features(
        des_left,
        des_right,
        ratio_threshold=ratio_threshold,
        min_matches=10,
    )
    match_time = (time.perf_counter() - match_start) * 1000

    num_good_matches = len(good_matches)

    H, mask, inlier_count = orb_bf_module.compute_homography(
        kp_left, kp_right, good_matches, ransac_threshold=5.0
    )

    if H is None:
        return None

    num_inliers = inlier_count
    inlier_ratio = num_inliers / num_good_matches if num_good_matches > 0 else 0.0

    result = orb_bf_module.warp_and_compose(left, right, H, blend=True)
    cropped = orb_bf_module.auto_crop(result)

    matches_vis = orb_bf_module.draw_matches(
        left, kp_left, right, kp_right, good_matches, max_matches=50
    )

    total_time = (time.perf_counter() - start_time) * 1000

    return {
        "name": "ORB+BF",
        "pano": cropped,
        "match_vis": matches_vis,
        "num_kp1": num_kp1,
        "num_kp2": num_kp2,
        "num_matches": num_matches,
        "num_good_matches": num_good_matches,
        "num_inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
        "total_time_ms": total_time,
        "feature_time_ms": feat_time,
        "match_time_ms": match_time,
    }


def run_orb_flann(left, right, nfeatures=4000, ratio_threshold=0.75):
    start_time = time.perf_counter()

    feat_start = time.perf_counter()
    kp_left, des_left = orb_flann_module.detect_orb(left, nfeatures=nfeatures)
    kp_right, des_right = orb_flann_module.detect_orb(right, nfeatures=nfeatures)
    feat_time = (time.perf_counter() - feat_start) * 1000

    if des_left is None or des_right is None:
        return None

    num_kp1 = len(kp_left)
    num_kp2 = len(kp_right)

    match_start = time.perf_counter()

    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=12,
        key_size=20,
        multi_probe_level=2,
    )
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        knn_matches = flann.knnMatch(des_left, des_right, k=2)
        num_matches = len(knn_matches)
    except Exception:
        num_matches = 0

    good_matches = orb_flann_module.match_flann_lsh(
        des_left,
        des_right,
        ratio_threshold=ratio_threshold,
        min_matches=10,
    )
    match_time = (time.perf_counter() - match_start) * 1000

    num_good_matches = len(good_matches)

    H, mask, inlier_count = orb_flann_module.compute_homography(
        kp_left, kp_right, good_matches, ransac_threshold=5.0
    )

    if H is None:
        return None

    num_inliers = inlier_count
    inlier_ratio = num_inliers / num_good_matches if num_good_matches > 0 else 0.0

    result = orb_flann_module.warp_and_compose(left, right, H, blend=True)
    cropped = orb_flann_module.auto_crop(result)

    matches_vis = orb_flann_module.draw_matches(
        left, kp_left, right, kp_right, good_matches, max_matches=50
    )

    total_time = (time.perf_counter() - start_time) * 1000

    return {
        "name": "ORB+FLANN",
        "pano": cropped,
        "match_vis": matches_vis,
        "num_kp1": num_kp1,
        "num_kp2": num_kp2,
        "num_matches": num_matches,
        "num_good_matches": num_good_matches,
        "num_inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
        "total_time_ms": total_time,
        "feature_time_ms": feat_time,
        "match_time_ms": match_time,
    }


def run_sift_bf(left, right, nfeatures=4000, ratio_threshold=0.75):
    start_time = time.perf_counter()

    feat_start = time.perf_counter()
    kp_left, des_left = sift_bf_module.detect_sift_features(left, nfeatures=nfeatures)
    kp_right, des_right = sift_bf_module.detect_sift_features(right, nfeatures=nfeatures)
    feat_time = (time.perf_counter() - feat_start) * 1000

    if des_left is None or des_right is None:
        return None

    num_kp1 = len(kp_left)
    num_kp2 = len(kp_right)

    match_start = time.perf_counter()
    good_matches, knn_matches = sift_bf_module.match_features_bf(
        des_left, des_right, ratio_threshold=ratio_threshold
    )
    match_time = (time.perf_counter() - match_start) * 1000

    num_good_matches = len(good_matches)
    num_matches = len(knn_matches)

    H, mask, inlier_count = sift_bf_module.compute_homography(
        kp_left, kp_right, good_matches, ransac_threshold=5.0
    )

    if H is None:
        return None

    num_inliers = inlier_count
    inlier_ratio = num_inliers / num_good_matches if num_good_matches > 0 else 0.0

    result = sift_bf_module.warp_and_compose(left, right, H, blend=True)
    cropped = sift_bf_module.auto_crop(result)

    matches_vis = sift_bf_module.draw_matches_simple(
        left, kp_left, right, kp_right, good_matches, max_matches=50
    )

    total_time = (time.perf_counter() - start_time) * 1000

    return {
        "name": "SIFT+BF",
        "pano": cropped,
        "match_vis": matches_vis,
        "num_kp1": num_kp1,
        "num_kp2": num_kp2,
        "num_matches": num_matches,
        "num_good_matches": num_good_matches,
        "num_inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
        "total_time_ms": total_time,
        "feature_time_ms": feat_time,
        "match_time_ms": match_time,
    }


def run_sift_flann(left, right, nfeatures=4000, ratio_threshold=0.75):
    start_time = time.perf_counter()

    feat_start = time.perf_counter()
    kp_left, des_left = sift_flann_module.detect_sift_features(
        left, nfeatures=nfeatures
    )
    kp_right, des_right = sift_flann_module.detect_sift_features(
        right, nfeatures=nfeatures
    )
    feat_time = (time.perf_counter() - feat_start) * 1000

    if des_left is None or des_right is None:
        return None

    num_kp1 = len(kp_left)
    num_kp2 = len(kp_right)

    match_start = time.perf_counter()
    good_matches, knn_matches = sift_flann_module.match_features_flann(
        des_left, des_right, ratio_threshold=ratio_threshold
    )
    match_time = (time.perf_counter() - match_start) * 1000

    num_good_matches = len(good_matches)
    num_matches = len(knn_matches)

    H, mask, inlier_count = sift_flann_module.compute_homography(
        kp_left, kp_right, good_matches, ransac_threshold=5.0
    )

    if H is None:
        return None

    num_inliers = inlier_count
    inlier_ratio = num_inliers / num_good_matches if num_good_matches > 0 else 0.0

    result = sift_flann_module.warp_and_compose(left, right, H, blend=True)
    cropped = sift_flann_module.auto_crop(result)

    matches_vis = sift_flann_module.draw_matches_visualization(
        left, kp_left, right, kp_right, good_matches, max_matches=50
    )

    total_time = (time.perf_counter() - start_time) * 1000

    return {
        "name": "SIFT+FLANN",
        "pano": cropped,
        "match_vis": matches_vis,
        "num_kp1": num_kp1,
        "num_kp2": num_kp2,
        "num_matches": num_matches,
        "num_good_matches": num_good_matches,
        "num_inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
        "total_time_ms": total_time,
        "feature_time_ms": feat_time,
        "match_time_ms": match_time,
    }


def save_results_csv(results, filename="results_comparison.csv"):
    with open(filename, "w", newline="") as csvfile:
        fieldnames = [
            "algorithm",
            "total_ms",
            "feature_ms",
            "match_ms",
            "num_kp1",
            "num_kp2",
            "num_matches",
            "num_good_matches",
            "num_inliers",
            "inlier_ratio",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            if result is not None:
                writer.writerow(
                    {
                        "algorithm": result["name"],
                        "total_ms": result.get("total_time_ms", ""),
                        "feature_ms": result.get("feature_time_ms", ""),
                        "match_ms": result.get("match_time_ms", ""),
                        "num_kp1": result.get("num_kp1", ""),
                        "num_kp2": result.get("num_kp2", ""),
                        "num_matches": result.get("num_matches", ""),
                        "num_good_matches": result.get("num_good_matches", ""),
                        "num_inliers": result.get("num_inliers", ""),
                        "inlier_ratio": result.get("inlier_ratio", ""),
                    }
                )


def print_comparison_table(results):
    print("\n" + "=" * 120)
    print("=== Algorithm Comparison (ORB+BF, ORB+FLANN, SIFT+BF, SIFT+FLANN) ===")
    print("=" * 120)

    header = f"{'Algorithm':<15} {'total_ms':<12} {'kp1':<8} {'kp2':<8} {'matches':<10} "
    header += f"{'good_matches':<15} {'inliers':<10} {'inlier_ratio':<15}"
    print(header)
    print("-" * 120)

    for result in results:
        if result is not None:
            row = f"{result['name']:<15} "
            row += f"{result.get('total_time_ms', 0):>10.2f}  "
            row += f"{result.get('num_kp1', 0):<8} "
            row += f"{result.get('num_kp2', 0):<8} "
            row += f"{result.get('num_matches', 0):<10} "
            row += f"{result.get('num_good_matches', 0):<15} "
            row += f"{result.get('num_inliers', 0):<10} "
            row += f"{result.get('inlier_ratio', 0):>13.4f}"
            print(row)
        else:
            print(
                f"{'FAILED':<15} {'N/A':<12} {'N/A':<8} {'N/A':<8} "
                f"{'N/A':<10} {'N/A':<15} {'N/A':<10} {'N/A':<15}"
            )

    print("=" * 120)


def create_comparison_image(results, output_path="pano_comparison_all.png"):
    valid_results = [r for r in results if r is not None and r.get("pano") is not None]

    if len(valid_results) == 0:
        print("Warning: No valid panoramas to create comparison image.")
        return

    target_height = 400
    resized_panos = []

    for result in valid_results:
        pano = result["pano"]
        h, w = pano.shape[:2]
        aspect = w / h
        new_width = int(target_height * aspect)
        resized = cv2.resize(pano, (new_width, target_height))
        resized_panos.append((resized, result["name"]))

    if len(resized_panos) > 0:
        combined = np.hstack([pano for pano, _ in resized_panos])

        current_x = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 255, 255)
        bg_color = (0, 0, 0)

        for pano, name in resized_panos:
            (text_width, text_height), baseline = cv2.getTextSize(
                name, font, font_scale, thickness
            )

            cv2.rectangle(
                combined,
                (current_x + 10, 10),
                (current_x + text_width + 20, text_height + baseline + 20),
                bg_color,
                -1,
            )

            cv2.putText(
                combined,
                name,
                (current_x + 15, text_height + 15),
                font,
                font_scale,
                color,
                thickness,
            )

            current_x += pano.shape[1]

        cv2.imwrite(output_path, combined)
        print(f"\nCombined comparison image saved to: {output_path}")


def print_summary(results):
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    valid_results = [r for r in results if r is not None]

    if len(valid_results) == 0:
        print("No algorithms completed successfully.")
        return

    fastest = min(valid_results, key=lambda x: x.get("total_time_ms", float("inf")))
    print(
        f"Fastest algorithm: {fastest['name']} "
        f"({fastest.get('total_time_ms', 0):.2f} ms)"
    )

    most_matches = max(valid_results, key=lambda x: x.get("num_good_matches", 0))
    print(
        f"Most good matches: {most_matches['name']} "
        f"({most_matches.get('num_good_matches', 0)} matches)"
    )

    highest_inlier = max(valid_results, key=lambda x: x.get("inlier_ratio", 0))
    print(
        f"Highest inlier ratio: {highest_inlier['name']} "
        f"({highest_inlier.get('inlier_ratio', 0):.4f})"
    )

    print("\nDetailed observations:")
    for result in valid_results:
        name = result["name"]
        total_ms = result.get("total_time_ms", 0)
        good_matches = result.get("num_good_matches", 0)
        inlier_ratio = result.get("inlier_ratio", 0)

        if total_ms < 1000:
            speed_desc = "fast"
        elif total_ms < 3000:
            speed_desc = "moderate"
        else:
            speed_desc = "slow"

        if good_matches > 100:
            match_desc = "many"
        elif good_matches > 50:
            match_desc = "moderate"
        else:
            match_desc = "few"

        if inlier_ratio > 0.7:
            inlier_desc = "high"
        elif inlier_ratio > 0.5:
            inlier_desc = "moderate"
        else:
            inlier_desc = "low"

        print(
            f"  - {name}: {speed_desc} ({total_ms:.2f} ms), "
            f"{match_desc} good matches ({good_matches}), "
            f"{inlier_desc} inlier ratio ({inlier_ratio:.4f})"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare panorama stitching algorithms")
    parser.add_argument(
        "--img1",
        type=str,
        default=str(DEFAULT_IMG1),
        help=(
            "Path to left image (first image). "
            "Default: original_images/src_left.jpg relative to project root."
        ),
    )
    parser.add_argument(
        "--img2",
        type=str,
        default=str(DEFAULT_IMG2),
        help=(
            "Path to right image (second image). "
            "Default: original_images/src_right.jpg relative to project root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for results (default: top-level 'results' folder).",
    )

    args = parser.parse_args()

    img1_path = Path(args.img1)
    img2_path = Path(args.img2)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading images...")
    left = cv2.imread(str(img1_path))
    right = cv2.imread(str(img2_path))

    if left is None:
        print(f"Error: Could not load {img1_path}")
        sys.exit(1)

    if right is None:
        print(f"Error: Could not load {img2_path}")
        sys.exit(1)

    print(f"Left image shape: {left.shape}")
    print(f"Right image shape: {right.shape}")

    print("\n" + "=" * 120)
    print("Running algorithms...")
    print("=" * 120)

    results = []

    print("\n[1/4] Running ORB + BFMatcher...")
    try:
        result = run_orb_bf(left, right)
        if result:
            results.append(result)
            cv2.imwrite(str(output_dir / "pano_orb_bf.png"), result["pano"])
            cv2.imwrite(str(output_dir / "matches_orb_bf.png"), result["match_vis"])
            print(f"  ✓ Completed: {result.get('total_time_ms', 0):.2f} ms")
        else:
            results.append(None)
            print("  ✗ Failed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results.append(None)

    print("\n[2/4] Running ORB + FLANN...")
    try:
        result = run_orb_flann(left, right)
        if result:
            results.append(result)
            cv2.imwrite(str(output_dir / "pano_orb_flann.png"), result["pano"])
            cv2.imwrite(str(output_dir / "matches_orb_flann.png"), result["match_vis"])
            print(f"  ✓ Completed: {result.get('total_time_ms', 0):.2f} ms")
        else:
            results.append(None)
            print("  ✗ Failed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results.append(None)

    print("\n[3/4] Running SIFT + BFMatcher...")
    try:
        result = run_sift_bf(left, right)
        if result:
            results.append(result)
            cv2.imwrite(str(output_dir / "pano_sift_bf.png"), result["pano"])
            cv2.imwrite(str(output_dir / "matches_sift_bf.png"), result["match_vis"])
            print(f"  ✓ Completed: {result.get('total_time_ms', 0):.2f} ms")
        else:
            results.append(None)
            print("  ✗ Failed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results.append(None)

    print("\n[4/4] Running SIFT + FLANN...")
    try:
        result = run_sift_flann(left, right)
        if result:
            results.append(result)
            cv2.imwrite(str(output_dir / "pano_sift_flann.png"), result["pano"])
            cv2.imwrite(str(output_dir / "matches_sift_flann.png"), result["match_vis"])
            print(f"  ✓ Completed: {result.get('total_time_ms', 0):.2f} ms")
        else:
            results.append(None)
            print("  ✗ Failed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results.append(None)

    csv_path = output_dir / "results_comparison2.csv"
    save_results_csv(results, str(csv_path))
    print(f"\nMetrics saved to: {csv_path}")

    comparison_path = output_dir / "pano_comparison_all.png"
    create_comparison_image(results, str(comparison_path))

    print_comparison_table(results)
    print_summary(results)

    print("\n" + "=" * 120)
    print("Comparison complete!")
    print("=" * 120)


if __name__ == "__main__":
    main()
