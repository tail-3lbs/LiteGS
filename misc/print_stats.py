#!/usr/bin/env python3

import os
import re
from pathlib import Path

def extract_gaussian_count(file_path):
    """Extract gaussian count from gaussian_count.txt"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        match = re.search(r'Final Gaussian count: (\d+)', content)
        return int(match.group(1)) if match else None
    except FileNotFoundError:
        return None

def extract_psnr(file_path):
    """Extract PSNR from metrics_testset.txt"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        match = re.search(r'PSNR:\s+([0-9.]+)', content)
        return float(match.group(1)) if match else None
    except FileNotFoundError:
        return None

def extract_training_time(file_path):
    """Extract training time in seconds from training_time.txt"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        match = re.search(r'Training wall time: ([0-9.]+) seconds', content)
        return float(match.group(1)) if match else None
    except FileNotFoundError:
        return None

def get_scene_data(base_path):
    """Extract scene data from a directory"""
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        print(f"Error: Directory {base_path} does not exist")
        return None, None
    
    scenes = []
    data = {}
    
    # Get all scene directories
    for scene_dir in base_dir.iterdir():
        if scene_dir.is_dir():
            scene_name = scene_dir.name
            scenes.append(scene_name)
            
            # Extract data for each scene
            gaussian_file = scene_dir / 'gaussian_count.txt'
            psnr_file = scene_dir / 'metrics_testset.txt'
            time_file = scene_dir / 'training_time.txt'
            
            gaussian_count = extract_gaussian_count(gaussian_file)
            psnr = extract_psnr(psnr_file)
            training_time = extract_training_time(time_file)
            
            data[scene_name] = {
                'gaussian_count': gaussian_count,
                'psnr': psnr,
                'training_time': training_time
            }
    
    scenes.sort()
    return scenes, data

def print_table(base_path, scenes, data, label=""):
    """Print table with scene data"""
    # Print the path
    if label:
        print(f"Data from {label}: {base_path}")
    else:
        print(f"Data from: {base_path}")
    print()
    
    # Calculate individual column widths for each scene + buffer
    scene_col_widths = {scene: max(12, len(scene) + 2) for scene in scenes}
    
    # Print table header
    print(f"{'Metric':<20}|", end='')
    for scene in scenes:
        print(f"{scene:>{scene_col_widths[scene]}}|", end='')
    print()
    
    # Print separator
    total_width = 21 + sum(scene_col_widths[scene] + 1 for scene in scenes)
    print('-' * total_width)
    
    # Print Gaussian count row
    print(f"{'Gaussian Number':<20}|", end='')
    for scene in scenes:
        count = data[scene]['gaussian_count']
        if count is not None:
            print(f"{count:>{scene_col_widths[scene]},}|", end='')
        else:
            print(f"{'N/A':>{scene_col_widths[scene]}}|", end='')
    print()
    
    # Print Testing PSNR row
    print(f"{'Testing PSNR':<20}|", end='')
    for scene in scenes:
        psnr = data[scene]['psnr']
        if psnr is not None:
            print(f"{psnr:>{scene_col_widths[scene]}.4f}|", end='')
        else:
            print(f"{'N/A':>{scene_col_widths[scene]}}|", end='')
    print()
    
    # Print Training time row
    print(f"{'Training Time (s)':<20}|", end='')
    for scene in scenes:
        time_sec = data[scene]['training_time']
        if time_sec is not None:
            print(f"{time_sec:>{scene_col_widths[scene]}.2f}|", end='')
        else:
            print(f"{'N/A':>{scene_col_widths[scene]}}|", end='')
    print()

def compare_training_times(scenes_a, data_a, scenes_b, data_b):
    """Compare training times between two datasets"""
    print("\n" + "="*60)
    print("TRAINING TIME COMPARISON (A/B ratios)")
    print("="*60)
    
    # Find common scenes
    common_scenes = set(scenes_a) & set(scenes_b)
    common_scenes = sorted(list(common_scenes))
    
    if not common_scenes:
        print("No common scenes found for comparison")
        return
    
    # Calculate column width for scene names (minimum 12) + buffer
    scene_col_width = max(12, max(len(scene) + 2 for scene in common_scenes))
    
    ratios = []
    
    print(f"{'Scene':<{scene_col_width}}|{'Time A (s)':<12}|{'Time B (s)':<12}|{'Ratio (A/B)':<12}|")
    print('-' * (scene_col_width + 41))
    
    for scene in common_scenes:
        time_a = data_a[scene]['training_time']
        time_b = data_b[scene]['training_time']
        
        if time_a is not None and time_b is not None and time_b != 0:
            ratio = time_a / time_b
            ratios.append(ratio)
            print(f"{scene:<{scene_col_width}}|{time_a:<12.2f}|{time_b:<12.2f}|{ratio:<12.4f}|")
        else:
            print(f"{scene:<{scene_col_width}}|{'N/A':<12}|{'N/A':<12}|{'N/A':<12}|")
    
    if ratios:
        mean_ratio = sum(ratios) / len(ratios)
        print('-' * (scene_col_width + 41))
        print(f"Mean ratio (A/B): {mean_ratio:.4f}")
        print(f"Number of scenes compared: {len(ratios)}")

def build_table(path_a, path_b=None):
    """Build table(s) with scene data"""
    scenes_a, data_a = get_scene_data(path_a)
    if scenes_a is None:
        return
    
    if path_b is None:
        # Single path mode
        print_table(path_a, scenes_a, data_a)
    else:
        # Comparison mode
        scenes_b, data_b = get_scene_data(path_b)
        if scenes_b is None:
            return
        
        print_table(path_a, scenes_a, data_a, "A")
        print()
        print_table(path_b, scenes_b, data_b, "B")
        
        compare_training_times(scenes_a, data_a, scenes_b, data_b)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build a table of scene metrics from output directory')
    parser.add_argument('path_a', help='Path to the first output directory (A)')
    parser.add_argument('path_b', nargs='?', help='Path to the second output directory (B) for comparison')
    
    args = parser.parse_args()
    build_table(args.path_a, args.path_b)