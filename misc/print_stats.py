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
            full_name = scene_dir.name
            scene_name = full_name.split('-')[0]
            scenes.append(scene_name)
            
            # Extract data for each scene
            gaussian_file = scene_dir / 'gaussian_count.txt'
            psnr_file = scene_dir / 'metrics_testset.txt'
            time_file = scene_dir / 'training_time.txt'
            
            gaussian_count = extract_gaussian_count(gaussian_file)
            psnr = extract_psnr(psnr_file)
            training_time = extract_training_time(time_file)
            
            if scene_name not in data:
                data[scene_name] = []
            
            data[scene_name].append({
                'gaussian_count': gaussian_count,
                'psnr': psnr,
                'training_time': training_time
            })
    
    scenes = list(set(scenes))
    scenes.sort()
    return scenes, data

def get_mean_metrics(data):
    """Calculate mean metrics for each scene"""
    mean_data = {}
    for scene, experiments in data.items():
        gaussian_counts = [exp['gaussian_count'] for exp in experiments if exp['gaussian_count'] is not None]
        psnrs = [exp['psnr'] for exp in experiments if exp['psnr'] is not None]
        training_times = [exp['training_time'] for exp in experiments if exp['training_time'] is not None]
        
        mean_data[scene] = {
            'gaussian_count': sum(gaussian_counts) / len(gaussian_counts) if gaussian_counts else None,
            'psnr': sum(psnrs) / len(psnrs) if psnrs else None,
            'training_time': sum(training_times) / len(training_times) if training_times else None
        }
    
    return mean_data

def print_table(base_path, scenes, data, label=""):
    """Print table with scene data"""
    # Calculate mean metrics
    mean_data = get_mean_metrics(data)
    
    # Print the path
    if label:
        print(f"Data from {label}: {base_path}")
    else:
        print(f"Data from: {base_path}")
    
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
        count = mean_data[scene]['gaussian_count']
        if count is not None:
            print(f"{count:>{scene_col_widths[scene]},.0f}|", end='')
        else:
            print(f"{'N/A':>{scene_col_widths[scene]}}|", end='')
    print()
    
    # Print Testing PSNR row
    print(f"{'Testing PSNR':<20}|", end='')
    for scene in scenes:
        psnr = mean_data[scene]['psnr']
        if psnr is not None:
            print(f"{psnr:>{scene_col_widths[scene]}.4f}|", end='')
        else:
            print(f"{'N/A':>{scene_col_widths[scene]}}|", end='')
    print()
    
    # Print Training time row
    print(f"{'Training Time (s)':<20}|", end='')
    for scene in scenes:
        time_sec = mean_data[scene]['training_time']
        if time_sec is not None:
            print(f"{time_sec:>{scene_col_widths[scene]}.2f}|", end='')
        else:
            print(f"{'N/A':>{scene_col_widths[scene]}}|", end='')
    print()

def compare_metrics(scenes_a, data_a, scenes_b, data_b):
    """Compare all metrics between two datasets"""
    # Calculate mean metrics for both datasets
    mean_data_a = get_mean_metrics(data_a)
    mean_data_b = get_mean_metrics(data_b)
    
    # Find common scenes
    common_scenes = set(scenes_a) & set(scenes_b)
    common_scenes = sorted(list(common_scenes))
    
    if not common_scenes:
        print("No common scenes found for comparison")
        return
    
    print("\n" + "="*80)
    print("METHOD COMPARISON (B/A ratios)")
    print("="*80)
    
    # Calculate column width for scene names (minimum 12) + buffer
    scene_col_width = max(12, max(len(scene) + 2 for scene in common_scenes))
    
    gaussian_ratios = []
    psnr_ratios = []
    time_ratios = []
    
    print(f"{'Scene':<{scene_col_width}}|{'Gaussian A':<12}|{'Gaussian B':<12}|{'Ratio (B/A)':<12}|{'PSNR A':<10}|{'PSNR B':<10}|{'Ratio (B/A)':<12}|{'Time A (s)':<12}|{'Time B (s)':<12}|{'Ratio (B/A)':<12}|{'Speedup':<8}|")
    print('-' * (scene_col_width + 123))
    
    for scene in common_scenes:
        gaussian_a = mean_data_a[scene]['gaussian_count']
        gaussian_b = mean_data_b[scene]['gaussian_count']
        psnr_a = mean_data_a[scene]['psnr']
        psnr_b = mean_data_b[scene]['psnr']
        time_a = mean_data_a[scene]['training_time']
        time_b = mean_data_b[scene]['training_time']
        
        # Calculate ratios
        gaussian_ratio = "N/A"
        psnr_ratio = "N/A"
        time_ratio = "N/A"
        
        if gaussian_a is not None and gaussian_b is not None and gaussian_a != 0:
            gaussian_ratio = gaussian_b / gaussian_a
            gaussian_ratios.append(gaussian_ratio)
        
        if psnr_a is not None and psnr_b is not None and psnr_a != 0:
            psnr_ratio = psnr_b / psnr_a
            psnr_ratios.append(psnr_ratio)
        
        if time_a is not None and time_b is not None and time_a != 0:
            time_ratio = time_b / time_a
            time_ratios.append(time_ratio)
        
        # Print row
        gaussian_a_str = f"{gaussian_a:,.0f}" if gaussian_a is not None else "N/A"
        gaussian_b_str = f"{gaussian_b:,.0f}" if gaussian_b is not None else "N/A"
        gaussian_ratio_str = f"{gaussian_ratio:.4f}" if isinstance(gaussian_ratio, (int, float)) else gaussian_ratio
        
        psnr_a_str = f"{psnr_a:.4f}" if psnr_a is not None else "N/A"
        psnr_b_str = f"{psnr_b:.4f}" if psnr_b is not None else "N/A"
        psnr_ratio_str = f"{psnr_ratio:.4f}" if isinstance(psnr_ratio, (int, float)) else psnr_ratio
        
        time_a_str = f"{time_a:.2f}" if time_a is not None else "N/A"
        time_b_str = f"{time_b:.2f}" if time_b is not None else "N/A"
        time_ratio_str = f"{time_ratio:.4f}" if isinstance(time_ratio, (int, float)) else time_ratio
        
        speedup_str = f"{1/time_ratio:.2f}x" if isinstance(time_ratio, (int, float)) else "N/A"
        print(f"{scene:<{scene_col_width}}|{gaussian_a_str:<12}|{gaussian_b_str:<12}|{gaussian_ratio_str:<12}|{psnr_a_str:<10}|{psnr_b_str:<10}|{psnr_ratio_str:<12}|{time_a_str:<12}|{time_b_str:<12}|{time_ratio_str:<12}|{speedup_str:<8}|")
    
    # Add mean row
    print('-' * (scene_col_width + 123))
    
    # Calculate mean values for the mean row
    gaussian_a_values = [mean_data_a[scene]['gaussian_count'] for scene in common_scenes 
                       if mean_data_a[scene]['gaussian_count'] is not None and 
                          mean_data_b[scene]['gaussian_count'] is not None]
    gaussian_b_values = [mean_data_b[scene]['gaussian_count'] for scene in common_scenes 
                       if mean_data_a[scene]['gaussian_count'] is not None and 
                          mean_data_b[scene]['gaussian_count'] is not None]
    
    psnr_a_values = [mean_data_a[scene]['psnr'] for scene in common_scenes 
                    if mean_data_a[scene]['psnr'] is not None and 
                       mean_data_b[scene]['psnr'] is not None]
    psnr_b_values = [mean_data_b[scene]['psnr'] for scene in common_scenes 
                    if mean_data_a[scene]['psnr'] is not None and 
                       mean_data_b[scene]['psnr'] is not None]
    
    time_a_values = [mean_data_a[scene]['training_time'] for scene in common_scenes 
                    if mean_data_a[scene]['training_time'] is not None and 
                       mean_data_b[scene]['training_time'] is not None]
    time_b_values = [mean_data_b[scene]['training_time'] for scene in common_scenes 
                    if mean_data_a[scene]['training_time'] is not None and 
                       mean_data_b[scene]['training_time'] is not None]
    
    # Calculate means
    mean_gaussian_a = sum(gaussian_a_values) / len(gaussian_a_values) if gaussian_a_values else None
    mean_gaussian_b = sum(gaussian_b_values) / len(gaussian_b_values) if gaussian_b_values else None
    mean_psnr_a = sum(psnr_a_values) / len(psnr_a_values) if psnr_a_values else None
    mean_psnr_b = sum(psnr_b_values) / len(psnr_b_values) if psnr_b_values else None
    mean_time_a = sum(time_a_values) / len(time_a_values) if time_a_values else None
    mean_time_b = sum(time_b_values) / len(time_b_values) if time_b_values else None
    
    # Calculate ratios for mean row
    mean_gaussian_ratio_val = mean_gaussian_b / mean_gaussian_a if mean_gaussian_a and mean_gaussian_b else None
    mean_psnr_ratio_val = mean_psnr_b / mean_psnr_a if mean_psnr_a and mean_psnr_b else None
    mean_time_ratio_val = mean_time_b / mean_time_a if mean_time_a and mean_time_b else None
    
    # Format values for mean row
    mean_gaussian_a_str = f"{mean_gaussian_a:,.0f}" if mean_gaussian_a is not None else "N/A"
    mean_gaussian_b_str = f"{mean_gaussian_b:,.0f}" if mean_gaussian_b is not None else "N/A"
    mean_gaussian_ratio_str = f"{mean_gaussian_ratio_val:.4f}" if mean_gaussian_ratio_val is not None else "N/A"
    
    mean_psnr_a_str = f"{mean_psnr_a:.4f}" if mean_psnr_a is not None else "N/A"
    mean_psnr_b_str = f"{mean_psnr_b:.4f}" if mean_psnr_b is not None else "N/A"
    mean_psnr_ratio_str = f"{mean_psnr_ratio_val:.4f}" if mean_psnr_ratio_val is not None else "N/A"
    
    mean_time_a_str = f"{mean_time_a:.2f}" if mean_time_a is not None else "N/A"
    mean_time_b_str = f"{mean_time_b:.2f}" if mean_time_b is not None else "N/A"
    mean_time_ratio_str = f"{mean_time_ratio_val:.4f}" if mean_time_ratio_val is not None else "N/A"
    
    mean_speedup_str = f"{1/mean_time_ratio_val:.2f}x" if mean_time_ratio_val is not None else "N/A"
    
    print(f"{'Mean':<{scene_col_width}}|{mean_gaussian_a_str:<12}|{mean_gaussian_b_str:<12}|{mean_gaussian_ratio_str:<12}|{mean_psnr_a_str:<10}|{mean_psnr_b_str:<10}|{mean_psnr_ratio_str:<12}|{mean_time_a_str:<12}|{mean_time_b_str:<12}|{mean_time_ratio_str:<12}|{mean_speedup_str:<8}|")

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
        
        compare_metrics(scenes_a, data_a, scenes_b, data_b)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build a table of scene metrics from output directory')
    parser.add_argument('path_a', help='Path to the first output directory (A)')
    parser.add_argument('path_b', nargs='?', help='Path to the second output directory (B) for comparison')
    
    args = parser.parse_args()
    build_table(args.path_a, args.path_b)