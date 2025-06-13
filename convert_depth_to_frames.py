#!/usr/bin/env python3
"""
Convert NvGesture depth .npy files to individual frame images
to match the RGB data structure for FusionNet training.

This script reads depth data from dataset-original/Nvidia/Processed/
and creates individual depth frame images in dataset/dataset/Nvidia/Processed_frames/depth/
"""

import os
import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm

def normalize_depth_for_visualization(depth_array):
    """
    Normalize depth array to 0-255 range for saving as image
    """
    # Remove invalid depth values (typically 0 or very large values)
    valid_depth = depth_array[depth_array > 0]
    if len(valid_depth) == 0:
        return np.zeros_like(depth_array, dtype=np.uint8)
    
    # Normalize to 0-255 range
    min_depth = np.percentile(valid_depth, 1)  # Use 1st percentile to avoid outliers
    max_depth = np.percentile(valid_depth, 99)  # Use 99th percentile to avoid outliers
    
    # Clip and normalize
    depth_normalized = np.clip(depth_array, min_depth, max_depth)
    depth_normalized = ((depth_normalized - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    
    return depth_normalized

def process_depth_npy_to_frames(npy_path, output_dir, sample_id):
    """
    Process a single depth .npy file and save individual frames
    """
    try:
        # Load depth data
        depth_data = np.load(npy_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each frame
        num_frames = depth_data.shape[0]
        for frame_idx in range(num_frames):
            frame = depth_data[frame_idx]
            
            # Handle different depth data formats
            if len(frame.shape) == 3:
                # If 3D, take first channel (assuming depth is in first channel)
                depth_frame = frame[:, :, 0]
            else:
                depth_frame = frame
            
            # Normalize depth for visualization
            depth_normalized = normalize_depth_for_visualization(depth_frame)
            
            # Save as grayscale image (matching RGB naming convention)
            frame_filename = f"img_{frame_idx+1:05d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, depth_normalized)
        
        return num_frames
        
    except Exception as e:
        print(f"Error processing {npy_path}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Convert NvGesture depth .npy files to frame images')
    parser.add_argument('--source', default='dataset-original/Nvidia/Processed', 
                       help='Source directory containing depth .npy files')
    parser.add_argument('--target', default='dataset/dataset/Nvidia/Processed_frames/depth',
                       help='Target directory for depth frame images')
    parser.add_argument('--splits', default='data/dataset_splits/NvGesture/rgb',
                       help='Directory containing train.txt and test.txt splits')
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    target_dir = Path(args.target)
    splits_dir = Path(args.splits)
    
    # Read train and valid splits to know which samples to process
    train_split_file = splits_dir / 'train.txt'
    valid_split_file = splits_dir / 'valid.txt'
    
    if not train_split_file.exists() or not valid_split_file.exists():
        print(f"Split files not found in {splits_dir}")
        return
    
    # Read RGB splits to get the sample structure
    with open(train_split_file, 'r') as f:
        train_samples = [line.strip().split() for line in f.readlines()]
    
    with open(valid_split_file, 'r') as f:
        test_samples = [line.strip().split() for line in f.readlines()]
    
    all_samples = train_samples + test_samples
    
    print(f"Processing {len(all_samples)} samples...")
    
    processed_count = 0
    total_frames = 0
    
    for sample_info in tqdm(all_samples, desc="Converting depth files"):
        rgb_path = sample_info[0]  # RGB path from split file
        class_label = sample_info[1]
        
        # Parse RGB path to get class and subject info
        # RGB path format: train/class_XX/subjectX_rX or test/class_XX/subjectX_rX
        path_parts = rgb_path.split('/')
        split_name = path_parts[0]  # train or test
        class_name = path_parts[1]  # class_XX
        subject_name = path_parts[2]  # subjectX_rX
        
        # Find corresponding depth .npy file
        depth_npy_pattern = source_dir / split_name / class_name / subject_name / "sk_depth.avi"
        
        if not depth_npy_pattern.exists():
            print(f"Depth directory not found: {depth_npy_pattern}")
            continue
        
        # Find the .npy file in the sk_depth.avi directory
        npy_files = list(depth_npy_pattern.glob("*_depth_label_*.npy"))
        if not npy_files:
            print(f"No depth .npy files found in {depth_npy_pattern}")
            continue
        
        # Use the first .npy file (there should typically be one per sample)
        npy_file = npy_files[0]
        
        # Create output directory structure matching RGB
        output_dir = target_dir / split_name / class_name / subject_name
        
        # Process the depth file
        frames_processed = process_depth_npy_to_frames(npy_file, output_dir, processed_count)
        
        if frames_processed > 0:
            processed_count += 1
            total_frames += frames_processed
    
    print(f"\nProcessing complete!")
    print(f"Processed {processed_count} samples")
    print(f"Total frames created: {total_frames}")
    print(f"Depth frames saved to: {target_dir}")
    
    # Create depth splits files
    depth_splits_dir = Path('data/dataset_splits/NvGesture/depth')
    depth_splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy RGB splits but change paths to depth
    with open(train_split_file, 'r') as f:
        train_lines = f.readlines()
    
    with open(valid_split_file, 'r') as f:
        valid_lines = f.readlines()
    
    # Create depth train split
    with open(depth_splits_dir / 'train.txt', 'w') as f:
        for line in train_lines:
            parts = line.strip().split()
            rgb_path = parts[0]
            label = parts[1]
            # Convert RGB path to depth path
            depth_path = rgb_path  # Same structure, just different modality
            f.write(f"{depth_path} {label}\n")
    
    # Create depth valid split
    with open(depth_splits_dir / 'valid.txt', 'w') as f:
        for line in valid_lines:
            parts = line.strip().split()
            rgb_path = parts[0]
            label = parts[1]
            # Convert RGB path to depth path
            depth_path = rgb_path  # Same structure, just different modality
            f.write(f"{depth_path} {label}\n")
    
    print(f"Depth split files created in: {depth_splits_dir}")

if __name__ == "__main__":
    main() 