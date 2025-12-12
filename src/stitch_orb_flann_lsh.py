import cv2
import numpy as np
import os
import sys


def detect_orb(image, nfeatures=4000):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(image, None)
    if des is None:
        print(f"Warning: No features detected in image")
        return kp, None
    if des.dtype != np.uint8:
        print(f"Warning: Descriptors are {des.dtype}, converting to uint8")
        des = des.astype(np.uint8)
    return kp, des


def match_flann_lsh(des_left, des_right, ratio_threshold=0.75, min_matches=10):
    if des_left.dtype != np.uint8:
        des_left = des_left.astype(np.uint8)
    if des_right.dtype != np.uint8:
        des_right = des_right.astype(np.uint8)
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        knn_matches = flann.knnMatch(des_left, des_right, k=2)
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        if len(good_matches) >= min_matches:
            print(f"Found {len(good_matches)} good matches using FLANN (LSH) + ratio test (threshold={ratio_threshold})")
            return good_matches
        print(f"Only {len(good_matches)} matches found with ratio={ratio_threshold}. Trying relaxed threshold...")
        if ratio_threshold < 0.85:
            relaxed_threshold = min(0.85, ratio_threshold + 0.05)
            good_matches = []
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < relaxed_threshold * n.distance:
                        good_matches.append(m)
            if len(good_matches) >= min_matches:
                print(f"Found {len(good_matches)} matches using relaxed ratio threshold={relaxed_threshold}")
                return good_matches
        if len(good_matches) < min_matches:
            print(f"Still only {len(good_matches)} matches. Trying increased search checks...")
            search_params_increased = dict(checks=100)
            flann_increased = cv2.FlannBasedMatcher(index_params, search_params_increased)
            knn_matches = flann_increased.knnMatch(des_left, des_right, k=2)
            good_matches = []
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
            print(f"After increased checks: Found {len(good_matches)} matches")
            return good_matches
    except cv2.error as e:
        print(f"FLANN matching failed: {e}")
        raise
    return good_matches


def compute_homography(kp_left, kp_right, matches, ransac_threshold=5.0):
    if len(matches) < 4:
        print(f"Error: Need at least 4 matches to compute homography, got {len(matches)}")
        return None, None, 0
    left_pts = np.float32([kp_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    right_pts = np.float32([kp_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(right_pts, left_pts, cv2.RANSAC, ransac_threshold)
    if H is None:
        print("Error: Homography computation failed")
        return None, None, 0
    inlier_count = np.sum(mask)
    print(f"Homography computed with {inlier_count}/{len(matches)} inliers (RANSAC threshold={ransac_threshold})")
    if inlier_count < 10:
        print(f"Warning: Very few inliers ({inlier_count}). Homography may be unreliable.")
    return H, mask, inlier_count


def warp_and_compose(left, right, H, blend=True):
    h_left, w_left = left.shape[:2]
    h_right, w_right = right.shape[:2]
    canvas_w = w_left + w_right
    canvas_h = max(h_left, h_right)
    warped_right = cv2.warpPerspective(right, H, (canvas_w, canvas_h))
    result = warped_right.copy()
    left_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    left_mask[0:h_left, 0:w_left] = 255
    warped_gray = cv2.cvtColor(warped_right, cv2.COLOR_BGR2GRAY) if len(warped_right.shape) == 3 else warped_right
    warped_mask = (warped_gray > 10).astype(np.uint8) * 255
    overlap = cv2.bitwise_and(left_mask, warped_mask)
    if blend and np.any(overlap):
        result_float = result.astype(np.float32)
        left_padded = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        left_padded[0:h_left, 0:w_left] = left.astype(np.float32)
        overlap_mask_3d = overlap[:, :, np.newaxis] / 255.0
        result_float = result_float * (1 - overlap_mask_3d * 0.5) + left_padded * (overlap_mask_3d * 0.5)
        non_overlap_left = cv2.bitwise_and(left_mask, cv2.bitwise_not(overlap))
        non_overlap_mask_3d = non_overlap_left[:, :, np.newaxis] / 255.0
        result_float = result_float * (1 - non_overlap_mask_3d) + left_padded * non_overlap_mask_3d
        result = result_float.astype(np.uint8)
    else:
        result[0:h_left, 0:w_left] = left
    return result


def auto_crop(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    non_black = np.where(gray > 10)
    if len(non_black[0]) == 0:
        print("Warning: Image appears to be all black, returning original")
        return image
    y_min, y_max = non_black[0].min(), non_black[0].max()
    x_min, x_max = non_black[1].min(), non_black[1].max()
    padding = 5
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    cropped = image[y_min:y_max, x_min:x_max]
    print(f"Auto-cropped: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    return cropped


def draw_keypoints(image, keypoints):
    vis = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return vis


def draw_matches(left, kp_left, right, kp_right, matches, max_matches=50):
    matches_to_draw = matches[:min(len(matches), max_matches)]
    vis = cv2.drawMatches(left, kp_left, right, kp_right, matches_to_draw, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return vis


def save_visualizations(left, right, kp_left, kp_right, matches, warped, result, cropped, output_dir):
    left_kp = draw_keypoints(left, kp_left)
    cv2.imwrite(os.path.join(output_dir, "01_left_keypoints.png"), left_kp)
    print(f"Saved: 01_left_keypoints.png ({len(kp_left)} keypoints)")
    right_kp = draw_keypoints(right, kp_right)
    cv2.imwrite(os.path.join(output_dir, "02_right_keypoints.png"), right_kp)
    print(f"Saved: 02_right_keypoints.png ({len(kp_right)} keypoints)")
    matches_vis = draw_matches(left, kp_left, right, kp_right, matches, max_matches=50)
    cv2.imwrite(os.path.join(output_dir, "03_matches.png"), matches_vis)
    print(f"Saved: 03_matches.png ({len(matches)} good matches)")
    cv2.imwrite(os.path.join(output_dir, "04_warped_right.png"), warped)
    print("Saved: 04_warped_right.png")
    cv2.imwrite(os.path.join(output_dir, "05_stitched_before_crop.png"), result)
    print("Saved: 05_stitched_before_crop.png")
    cv2.imwrite(os.path.join(output_dir, "06_final_panorama.png"), cropped)
    print("Saved: 06_final_panorama.png")


def main():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results/stitch_orb_flann_lsh"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")
    left_path = "../original_images/src_left.jpg"
    right_path = "../original_images/src_right.jpg"
    if not os.path.exists(left_path):
        print(f"Error: {left_path} not found!")
        sys.exit(1)
    if not os.path.exists(right_path):
        print(f"Error: {right_path} not found!")
        sys.exit(1)
    print("Loading images...")
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if left is None:
        print(f"Error: Could not load {left_path}")
        sys.exit(1)
    if right is None:
        print(f"Error: Could not load {right_path}")
        sys.exit(1)
    print(f"Left image shape: {left.shape}")
    print(f"Right image shape: {right.shape}")
    print("\nDetecting ORB features...")
    kp_left, des_left = detect_orb(left, nfeatures=4000)
    kp_right, des_right = detect_orb(right, nfeatures=4000)
    if des_left is None or des_right is None:
        print("Error: Failed to detect features in one or both images")
        sys.exit(1)
    print(f"Left: {len(kp_left)} keypoints, descriptors shape: {des_left.shape}, dtype: {des_left.dtype}")
    print(f"Right: {len(kp_right)} keypoints, descriptors shape: {des_right.shape}, dtype: {des_right.dtype}")
    print("\nMatching features using FLANN (LSH)...")
    matches = match_flann_lsh(des_left, des_right, ratio_threshold=0.75, min_matches=10)
    if len(matches) < 4:
        print(f"Error: Too few matches ({len(matches)}). Need at least 4 for homography.")
        sys.exit(1)
    print("\nComputing homography with RANSAC...")
    H, mask, inlier_count = compute_homography(kp_left, kp_right, matches, ransac_threshold=5.0)
    if H is None:
        print("Error: Homography computation failed")
        sys.exit(1)
    print(f"Homography matrix:\n{H}")
    print("\nWarping and compositing images...")
    h_right, w_right = right.shape[:2]
    canvas_w = left.shape[1] + w_right
    canvas_h = max(left.shape[0], right.shape[0])
    warped = cv2.warpPerspective(right, H, (canvas_w, canvas_h))
    result = warp_and_compose(left, right, H, blend=True)
    print("\nAuto-cropping black borders...")
    cropped = auto_crop(result)
    output_path = os.path.join(output_dir, "panorama_result_flann.jpg")
    cv2.imwrite(output_path, cropped)
    print(f"\nPanorama saved to {output_path}")
    print("\nSaving visualizations as separate PNG files...")
    save_visualizations(left, right, kp_left, kp_right, matches, warped, result, cropped, output_dir)
    print("\nPanorama stitching completed successfully!")
    print("\nSummary:")
    print(f"  - Matcher: FLANN with LSH index (vs Brute-Force in BF version)")
    print(f"  - Keypoints detected: Left={len(kp_left)}, Right={len(kp_right)}")
    print(f"  - Matches found: {len(matches)}")
    print(f"  - Inliers: {inlier_count}")


if __name__ == "__main__":
    main()
