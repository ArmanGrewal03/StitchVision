"""
Panorama Stitching using ORB Feature Detection and FLANN-based Matching with LSH
=================================================================================
This script stitches two images (src_left.jpg and src_right.jpg) into a panorama
using ORB feature detection, FLANN matching with LSH (Locality-Sensitive Hashing)
for binary descriptors, and homography-based warping.

KEY DIFFERENCE from BF version: Uses FLANN with LSH index instead of Brute-Force
matcher. FLANN is faster for large descriptor sets due to approximate nearest
neighbor search using hashing.
"""

import cv2
import numpy as np
import os
import sys


def detect_orb(image, nfeatures=4000):
    """
    Detect and compute ORB features from an image.
    (Same as BF version - no changes)
    
    Args:
        image: Input image (grayscale or BGR)
        nfeatures: Maximum number of features to detect (default: 4000)
        
    Returns:
        kp: List of keypoints
        des: Descriptor array (None if no features found)
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(image, None)
    
    if des is None:
        print(f"Warning: No features detected in image")
        return kp, None
    
    # Ensure descriptors are uint8 (required for FLANN LSH)
    # ORB already returns uint8, but we assert/cast to be safe
    if des.dtype != np.uint8:
        print(f"Warning: Descriptors are {des.dtype}, converting to uint8")
        des = des.astype(np.uint8)
    
    return kp, des


def match_flann_lsh(des_left, des_right, ratio_threshold=0.75, min_matches=10):
    """
    Match descriptors using FLANN matcher with LSH (Locality-Sensitive Hashing).
    
    KEY CHANGE from BF version: Uses cv2.FlannBasedMatcher with LSH index params
    instead of cv2.BFMatcher. FLANN uses approximate nearest neighbor search,
    which is faster for large descriptor sets.
    
    Args:
        des_left: Descriptors from left image (must be uint8)
        des_right: Descriptors from right image (must be uint8)
        ratio_threshold: Lowe's ratio test threshold (default: 0.75)
        min_matches: Minimum number of matches required (default: 10)
        
    Returns:
        good_matches: List of good matches (DMatch objects) after ratio test
    """
    # Ensure descriptors are uint8 (required for FLANN LSH)
    if des_left.dtype != np.uint8:
        des_left = des_left.astype(np.uint8)
    if des_right.dtype != np.uint8:
        des_right = des_right.astype(np.uint8)
    
    # FLANN parameters for ORB binary descriptors (LSH index)
    # FLANN_INDEX_LSH = 6 for binary descriptors like ORB
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=12,      # Number of hash tables (typical range: 6-20)
        key_size=20,          # Size of hash key (typical range: 10-30)
        multi_probe_level=2   # Multi-probe level for better accuracy (range: 1-2)
    )
    search_params = dict(checks=50)  # Number of times tree is traversed (higher = more accurate but slower)
    
    # Initialize FLANN matcher with LSH parameters
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        # Use knnMatch with k=2 for Lowe's ratio test
        knn_matches = flann.knnMatch(des_left, des_right, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Keep match if distance is significantly smaller than second best
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        # If we have enough matches, return them
        if len(good_matches) >= min_matches:
            print(f"Found {len(good_matches)} good matches using FLANN (LSH) + ratio test (threshold={ratio_threshold})")
            return good_matches
        
        print(f"Only {len(good_matches)} matches found with ratio={ratio_threshold}. Trying relaxed threshold...")
        
        # Fallback: Try slightly higher ratio threshold (0.8-0.85) if too few matches
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
        
        # If still too few, try increasing search checks
        if len(good_matches) < min_matches:
            print(f"Still only {len(good_matches)} matches. Trying increased search checks...")
            search_params_increased = dict(checks=100)  # Increase from 50 to 100
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
    """
    Compute homography matrix using RANSAC.
    (Same as BF version - no changes)
    
    Args:
        kp_left: Keypoints from left image
        kp_right: Keypoints from right image
        matches: List of good matches
        ransac_threshold: RANSAC reprojection threshold (default: 5.0 pixels)
        
    Returns:
        H: Homography matrix (3x3) or None if failed
        mask: Inlier mask (1D array where 1 indicates inlier)
        inlier_count: Number of inliers
    """
    if len(matches) < 4:
        print(f"Error: Need at least 4 matches to compute homography, got {len(matches)}")
        return None, None, 0
    
    # Extract matching points from keypoints using match indices
    left_pts = np.float32([kp_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    right_pts = np.float32([kp_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute homography using RANSAC for robust estimation
    # Maps right_pts -> left_pts (warping right image to left image's frame)
    H, mask = cv2.findHomography(
        right_pts, left_pts, 
        cv2.RANSAC, 
        ransac_threshold
    )
    
    if H is None:
        print("Error: Homography computation failed")
        return None, None, 0
    
    inlier_count = np.sum(mask)
    print(f"Homography computed with {inlier_count}/{len(matches)} inliers (RANSAC threshold={ransac_threshold})")
    
    # Validate homography quality
    if inlier_count < 10:
        print(f"Warning: Very few inliers ({inlier_count}). Homography may be unreliable.")
    
    return H, mask, inlier_count


def warp_and_compose(left, right, H, blend=True):
    """
    Warp the right image and compose with the left image.
    (Same as BF version - no changes)
    
    Args:
        left: Left image
        right: Right image
        H: Homography matrix (3x3) mapping right -> left
        blend: Whether to apply feathering blend in overlap region
        
    Returns:
        result: Stitched panorama image
    """
    h_left, w_left = left.shape[:2]
    h_right, w_right = right.shape[:2]
    
    # Calculate canvas size: width = left.width + right.width, height = max of heights
    canvas_w = w_left + w_right
    canvas_h = max(h_left, h_right)
    
    # Warp the right image into the left image's frame using homography
    warped_right = cv2.warpPerspective(right, H, (canvas_w, canvas_h))
    
    # Create result canvas - start with warped right image
    result = warped_right.copy()
    
    # Create mask for left image region
    left_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    left_mask[0:h_left, 0:w_left] = 255
    
    # Create mask for warped right image (non-black pixels)
    warped_gray = cv2.cvtColor(warped_right, cv2.COLOR_BGR2GRAY) if len(warped_right.shape) == 3 else warped_right
    warped_mask = (warped_gray > 10).astype(np.uint8) * 255  # Threshold to ignore near-black
    
    # Find overlap region where both images have content
    overlap = cv2.bitwise_and(left_mask, warped_mask)
    
    if blend and np.any(overlap):
        # Simple blending: equal weights (50/50) in overlap region
        result_float = result.astype(np.float32)
        left_padded = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        left_padded[0:h_left, 0:w_left] = left.astype(np.float32)
        
        # Blend only in overlap region
        overlap_mask_3d = overlap[:, :, np.newaxis] / 255.0
        result_float = result_float * (1 - overlap_mask_3d * 0.5) + left_padded * (overlap_mask_3d * 0.5)
        
        # Non-overlap regions: left image takes precedence in its region
        non_overlap_left = cv2.bitwise_and(left_mask, cv2.bitwise_not(overlap))
        non_overlap_mask_3d = non_overlap_left[:, :, np.newaxis] / 255.0
        result_float = result_float * (1 - non_overlap_mask_3d) + left_padded * non_overlap_mask_3d
        
        result = result_float.astype(np.uint8)
    else:
        # No blending: simple paste (left image takes precedence in its region)
        result[0:h_left, 0:w_left] = left
    
    return result


def auto_crop(image):
    """
    Automatically crop black borders from the stitched image.
    (Same as BF version - no changes)
    
    Finds the bounding box of non-black pixels and crops to that region.
    
    Args:
        image: Input image with potential black borders
        
    Returns:
        cropped: Cropped image with black borders removed
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Find non-black pixels (threshold > 10 to account for near-black pixels)
    non_black = np.where(gray > 10)
    
    if len(non_black[0]) == 0:
        print("Warning: Image appears to be all black, returning original")
        return image
    
    # Get bounding box of non-black region
    y_min, y_max = non_black[0].min(), non_black[0].max()
    x_min, x_max = non_black[1].min(), non_black[1].max()
    
    # Crop with small padding to avoid cutting off edges
    padding = 5
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    
    cropped = image[y_min:y_max, x_min:x_max]
    
    print(f"Auto-cropped: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    return cropped


def draw_keypoints(image, keypoints):
    """
    Draw keypoints on an image for visualization.
    (Same as BF version - no changes)
    
    Args:
        image: Input image
        keypoints: List of keypoints
        
    Returns:
        vis: Visualization image with keypoints drawn
    """
    vis = cv2.drawKeypoints(
        image, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return vis


def draw_matches(left, kp_left, right, kp_right, matches, max_matches=50):
    """
    Draw matches between two images for visualization.
    (Same as BF version - no changes)
    
    Args:
        left: Left image
        kp_left: Left keypoints
        right: Right image
        kp_right: Right keypoints
        matches: List of matches (DMatch objects)
        max_matches: Maximum number of matches to draw (default: 50)
        
    Returns:
        vis: Visualization image with matches drawn
    """
    # Draw top N matches
    matches_to_draw = matches[:min(len(matches), max_matches)]
    
    vis = cv2.drawMatches(
        left, kp_left,
        right, kp_right,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis


def save_visualizations(left, right, kp_left, kp_right, matches, warped, result, cropped):
    """
    Save intermediate and final results as separate PNG files.
    (Same as BF version - saves as separate PNGs instead of matplotlib figure)
    
    Args:
        left: Left input image
        right: Right input image
        kp_left: Left keypoints
        kp_right: Right keypoints
        matches: Good matches
        warped: Warped right image
        result: Stitched result before cropping
        cropped: Final cropped result
    """
    # Left image with keypoints
    left_kp = draw_keypoints(left, kp_left)
    cv2.imwrite("01_left_keypoints.png", left_kp)
    print(f"Saved: 01_left_keypoints.png ({len(kp_left)} keypoints)")
    
    # Right image with keypoints
    right_kp = draw_keypoints(right, kp_right)
    cv2.imwrite("02_right_keypoints.png", right_kp)
    print(f"Saved: 02_right_keypoints.png ({len(kp_right)} keypoints)")
    
    # Matches
    matches_vis = draw_matches(left, kp_left, right, kp_right, matches, max_matches=50)
    cv2.imwrite("03_matches.png", matches_vis)
    print(f"Saved: 03_matches.png ({len(matches)} good matches)")
    
    # Warped right image
    cv2.imwrite("04_warped_right.png", warped)
    print("Saved: 04_warped_right.png")
    
    # Stitched result (before cropping)
    cv2.imwrite("05_stitched_before_crop.png", result)
    print("Saved: 05_stitched_before_crop.png")
    
    # Final cropped result
    cv2.imwrite("06_final_panorama.png", cropped)
    print("Saved: 06_final_panorama.png")


def main():
    """
    Main function to execute panorama stitching pipeline using FLANN (LSH) matching.
    
    Pipeline:
    1. Load images (src_left.jpg, src_right.jpg)
    2. Detect ORB features
    3. Match features using FLANN with LSH
    4. Compute homography with RANSAC
    5. Warp and compose images
    6. Auto-crop black borders
    7. Save results and visualizations
    """
    # Image file paths
    left_path = "src_left.jpg"
    right_path = "src_right.jpg"
    
    # Check if images exist
    if not os.path.exists(left_path):
        print(f"Error: {left_path} not found!")
        sys.exit(1)
    
    if not os.path.exists(right_path):
        print(f"Error: {right_path} not found!")
        sys.exit(1)
    
    # Load images
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
    
    # Detect ORB features (same as BF version)
    print("\nDetecting ORB features...")
    kp_left, des_left = detect_orb(left, nfeatures=4000)
    kp_right, des_right = detect_orb(right, nfeatures=4000)
    
    if des_left is None or des_right is None:
        print("Error: Failed to detect features in one or both images")
        sys.exit(1)
    
    print(f"Left: {len(kp_left)} keypoints, descriptors shape: {des_left.shape}, dtype: {des_left.dtype}")
    print(f"Right: {len(kp_right)} keypoints, descriptors shape: {des_right.shape}, dtype: {des_right.dtype}")
    
    # Match features using FLANN with LSH (KEY DIFFERENCE from BF version)
    print("\nMatching features using FLANN (LSH)...")
    matches = match_flann_lsh(des_left, des_right, ratio_threshold=0.75, min_matches=10)
    
    if len(matches) < 4:
        print(f"Error: Too few matches ({len(matches)}). Need at least 4 for homography.")
        sys.exit(1)
    
    # Compute homography with RANSAC (same as BF version)
    print("\nComputing homography with RANSAC...")
    H, mask, inlier_count = compute_homography(kp_left, kp_right, matches, ransac_threshold=5.0)
    
    if H is None:
        print("Error: Homography computation failed")
        sys.exit(1)
    
    print(f"Homography matrix:\n{H}")
    
    # Warp and compose images (same as BF version)
    print("\nWarping and compositing images...")
    h_right, w_right = right.shape[:2]
    canvas_w = left.shape[1] + w_right
    canvas_h = max(left.shape[0], right.shape[0])
    
    warped = cv2.warpPerspective(right, H, (canvas_w, canvas_h))
    result = warp_and_compose(left, right, H, blend=True)
    
    # Auto-crop black borders (same as BF version)
    print("\nAuto-cropping black borders...")
    cropped = auto_crop(result)
    
    # Save final result
    output_path = "panorama_result_flann.jpg"
    cv2.imwrite(output_path, cropped)
    print(f"\nPanorama saved to {output_path}")
    
    # Save visualizations as separate PNG files (same as BF version)
    print("\nSaving visualizations as separate PNG files...")
    save_visualizations(left, right, kp_left, kp_right, matches, warped, result, cropped)
    
    print("\nPanorama stitching completed successfully!")
    print("\nSummary:")
    print(f"  - Matcher: FLANN with LSH index (vs Brute-Force in BF version)")
    print(f"  - Keypoints detected: Left={len(kp_left)}, Right={len(kp_right)}")
    print(f"  - Matches found: {len(matches)}")
    print(f"  - Inliers: {inlier_count}")


if __name__ == "__main__":
    main()

