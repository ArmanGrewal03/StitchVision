# StitchVision

StitchVision is a simple Python + OpenCV project for **panorama stitching**.

It stitches two images together using different feature detectors and matchers so you can see **what works better and why**.

## What It Does

1. Load two overlapping images  
2. Find feature points  
3. Match features  
4. Compute a homography (RANSAC)  
5. Warp and blend images  
6. Crop black borders  
7. Save results  

## Files Overview

### `stitch_orb_bf.py`
- ORB features  
- Brute-Force matcher (Hamming)  
- Fast and lightweight  

### `stitch_orb_flann_lsh.py`
- ORB features  
- FLANN matcher (LSH)  
- Faster matching for many features  

### `stitch_sift_bf.py`
- SIFT features  
- Brute-Force matcher (L2)  
- More accurate, slower  

### `stitch_sift_flann.py`
- SIFT features  
- FLANN matcher (KD-tree)  
- Faster than BF for SIFT  

## Output

Each script saves:
- Keypoints image  
- Matches image  
- Warped image  
- Final panorama  

Results are saved in the `results/` folder.

## Requirements

- Python 3  
- OpenCV (contrib build for SIFT)  
- NumPy  

## Goal

Learn how panorama stitching works and compare:
- ORB vs SIFT  
- Brute-Force vs FLANN  
- Speed vs quality  
