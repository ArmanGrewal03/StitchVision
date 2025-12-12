"""
Panorama Stitching using ORB Feature Detection and Brute-Force Matching
=======================================================================
This script stitches two images (src_left.jpg and src_right.jpg) into a panorama
using ORB feature detection, Brute-Force matching with Hamming distance, and
homography-based warping.
"""

import cv2
import numpy as np
import os
import sys


def detect_features(image, nfeatures=4000):
    """
    Detect and compute ORB features from an image.
    
    Args:
        image: Input image (grayscale or BGR)
        nfeatures: Maximum number of features to detect
        
    Returns:
        kp: List of keypoints
        des: Descriptor array (None if no features found)
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(image, None)
    
    if des is None:
        print(f"Warning: No features detected in image")
        return kp, None
    
    return kp, des


def match_features(des_left, des_right, ratio_threshold=0.75, min_matches=50):
    """
    Match descriptors using Brute-Force matcher with Hamming distance.
    First tries knnMatch with Lowe's ratio test, falls back to crossCheck
    if too few matches are found.
    
    Args:
        des_left: Descriptors from left image
        des_right: Descriptors from right image
        ratio_threshold: Lowe's ratio test threshold (default: 0.75)
        min_matches: Minimum number of matches required (default: 50)
        
    Returns:
        good_matches: List of good matches (DMatch objects)
    """
    # Initialize BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Try knnMatch with ratio test first
    try:
        knn_matches = bf.knnMatch(des_left, des_right, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        # If we have enough matches, return them
        if len(good_matches) >= min_matches:
            print(f"Found {len(good_matches)} good matches using knnMatch + ratio test")
            return good_matches
        
        print(f"Only {len(good_matches)} matches found with knnMatch. Trying fallback...")
        
    except cv2.error as e:
        print(f"knnMatch failed: {e}. Trying fallback...")
    
    # Fallback: Use crossCheck matching
    bf_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf_cross.match(des_left, des_right)
    
    # Sort matches by distance and take top matches
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:min(len(matches), 500)]  # Take top 500 or all if fewer
    
    print(f"Fallback: Found {len(good_matches)} matches using crossCheck")
    return good_matches


def compute_homography(kp_left, kp_right, matches, ransac_threshold=5.0):
    """
    Compute homography matrix using RANSAC.
    
    Args:
        kp_left: Keypoints from left image
        kp_right: Keypoints from right image
        matches: List of good matches
        ransac_threshold: RANSAC reprojection threshold (default: 5.0)
        
    Returns:
        H: Homography matrix (3x3) or None if failed
        mask: Inlier mask
        inlier_count: Number of inliers
    """
    if len(matches) < 4:
        print(f"Error: Need at least 4 matches to compute homography, got {len(matches)}")
        return None, None, 0
    
    # Extract matching points
    left_pts = np.float32([kp_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    right_pts = np.float32([kp_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute homography using RANSAC
    H, mask = cv2.findHomography(
        right_pts, left_pts, 
        cv2.RANSAC, 
        ransac_threshold
    )
    
    if H is None:
        print("Error: Homography computation failed")
        return None, None, 0
    
    inlier_count = np.sum(mask)
    print(f"Homography computed with {inlier_count}/{len(matches)} inliers")
    
    # Validate homography quality
    if inlier_count < 10:
        print(f"Warning: Very few inliers ({inlier_count}). Homography may be unreliable.")
    
    return H, mask, inlier_count


def warp_and_compose(left, right, H, blend=True):
    """
    Warp the right image and compose with the left image.
    
    Args:
        left: Left image
        right: Right image
        H: Homography matrix (3x3)
        blend: Whether to apply feathering blend in overlap region
        
    Returns:
        result: Stitched panorama image
    """
    h_left, w_left = left.shape[:2]
    h_right, w_right = right.shape[:2]
    
    # Calculate canvas size
    canvas_w = w_left + w_right
    canvas_h = max(h_left, h_right)
    
    # Warp the right image
    warped_right = cv2.warpPerspective(right, H, (canvas_w, canvas_h))
    
    # Create result canvas - start with warped right image
    result = warped_right.copy()
    
    # Create mask for left image region
    left_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    left_mask[0:h_left, 0:w_left] = 255
    
    # Create mask for warped right image (non-black pixels)
    warped_gray = cv2.cvtColor(warped_right, cv2.COLOR_BGR2GRAY) if len(warped_right.shape) == 3 else warped_right
    warped_mask = (warped_gray > 10).astype(np.uint8) * 255  # Threshold to ignore near-black
    
    # Find overlap region
    overlap = cv2.bitwise_and(left_mask, warped_mask)
    
    if blend and np.any(overlap):
        # Simple blending: equal weights in overlap region
        result_float = result.astype(np.float32)
        left_padded = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        left_padded[0:h_left, 0:w_left] = left.astype(np.float32)
        
        # Blend only in overlap region
        overlap_mask_3d = overlap[:, :, np.newaxis] / 255.0
        result_float = result_float * (1 - overlap_mask_3d * 0.5) + left_padded * (overlap_mask_3d * 0.5)
        
        # Non-overlap regions
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
    
    # Find non-black pixels
    non_black = np.where(gray > 10)  # Threshold to account for near-black pixels
    
    if len(non_black[0]) == 0:
        print("Warning: Image appears to be all black, returning original")
        return image
    
    # Get bounding box
    y_min, y_max = non_black[0].min(), non_black[0].max()
    x_min, x_max = non_black[1].min(), non_black[1].max()
    
    # Crop with small padding
    padding = 5
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    
    cropped = image[y_min:y_max, x_min:x_max]
    
    print(f"Auto-cropped: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    return cropped


def draw_keypoints(image, keypoints, title="Keypoints"):
    """
    Draw keypoints on an image for visualization.
    
    Args:
        image: Input image
        keypoints: List of keypoints
        title: Window title
        
    Returns:
        vis: Visualization image
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
    
    Args:
        left: Left image
        kp_left: Left keypoints
        right: Right image
        kp_right: Right keypoints
        matches: List of matches
        max_matches: Maximum number of matches to draw
        
    Returns:
        vis: Visualization image
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
    Main function to execute panorama stitching pipeline.
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
    
    # Detect features
    print("\nDetecting features...")
    kp_left, des_left = detect_features(left, nfeatures=4000)
    kp_right, des_right = detect_features(right, nfeatures=4000)
    
    if des_left is None or des_right is None:
        print("Error: Failed to detect features in one or both images")
        sys.exit(1)
    
    print(f"Left: {len(kp_left)} keypoints, descriptors shape: {des_left.shape}")
    print(f"Right: {len(kp_right)} keypoints, descriptors shape: {des_right.shape}")
    
    # Match features
    print("\nMatching features...")
    matches = match_features(des_left, des_right, ratio_threshold=0.75, min_matches=50)
    
    if len(matches) < 4:
        print(f"Error: Too few matches ({len(matches)}). Need at least 4 for homography.")
        sys.exit(1)
    
    # Compute homography
    print("\nComputing homography...")
    H, mask, inlier_count = compute_homography(kp_left, kp_right, matches, ransac_threshold=5.0)
    
    if H is None:
        print("Error: Homography computation failed")
        sys.exit(1)
    
    print(f"Homography matrix:\n{H}")
    
    # Warp and compose
    print("\nWarping and compositing images...")
    h_right, w_right = right.shape[:2]
    canvas_w = left.shape[1] + w_right
    canvas_h = max(left.shape[0], right.shape[0])
    
    warped = cv2.warpPerspective(right, H, (canvas_w, canvas_h))
    result = warp_and_compose(left, right, H, blend=True)
    
    # Auto-crop
    print("\nAuto-cropping black borders...")
    cropped = auto_crop(result)
    
    # Save result
    output_path = "panorama_result.jpg"
    cv2.imwrite(output_path, cropped)
    print(f"\nPanorama saved to {output_path}")
    
    # Save visualizations as separate PNG files
    print("\nSaving visualizations as separate PNG files...")
    save_visualizations(left, right, kp_left, kp_right, matches, warped, result, cropped)
    
    print("\nPanorama stitching completed successfully!")


if __name__ == "__main__":
    main()

