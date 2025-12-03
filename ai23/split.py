import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from config import GlobalConfig

# Split percentages
TRAIN_RATIO = 0.6
TEST_RATIO = 0.2
VAL_RATIO = 0.2

# Weather and route name configuration
WEATHER = "sunny"
ROUTE_NAME = "2025-12-01_route00"

# Current directory (where this script is located)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the exact folder structure
FOLDER_STRUCTURE = [
    'camera/depth/cld',
    'camera/depth/cld2',
    'camera/depth/img',
    'camera/depth/map',
    'camera/histogram',
    'camera/optical_flow',
    'camera/rgb',
    'camera/seg/img',
    'camera/seg/map',
    'join_img/all_img',
    'lidar/cld',
    'lidar/img/bev_dep',
    'lidar/img/bev_seg',
    'lidar/img/front_dep',
    'lidar/img/front_seg',
    'lidar/img/rear_dep',
    'lidar/img/rear_seg',
    'lidar/seg',
    'meta',
]

def get_all_samples(source_dir):
    """Get all unique sample identifiers from the meta folder."""
    meta_dir = os.path.join(source_dir, 'meta')
    if os.path.exists(meta_dir):
        samples = [os.path.splitext(f)[0] for f in os.listdir(meta_dir) if os.path.isfile(os.path.join(meta_dir, f))]
    else:
        # Fallback: get samples from camera/rgb
        rgb_dir = os.path.join(source_dir, 'camera', 'rgb')
        if os.path.exists(rgb_dir):
            samples = [os.path.splitext(f)[0] for f in os.listdir(rgb_dir) if os.path.isfile(os.path.join(rgb_dir, f))]
        else:
            raise FileNotFoundError("Could not find meta or camera/rgb folder to get sample list")
    return sorted(list(set(samples)))

def split_samples(samples, train_ratio, test_ratio, val_ratio):
    """Split samples into train, test, val sets."""
    random.seed(42)  # For reproducibility
    random.shuffle(samples)
    
    n = len(samples)
    train_end = int(n * train_ratio)
    test_end = train_end + int(n * test_ratio)
    
    train_samples = samples[:train_end]
    test_samples = samples[train_end:test_end]
    val_samples = samples[test_end:]
    
    return train_samples, test_samples, val_samples

def get_all_subdirs(source_dir):
    """Recursively get all subdirectories that contain files."""
    subdirs = []
    for root, dirs, files in os.walk(source_dir):
        if files:  # Only include directories that have files
            rel_path = os.path.relpath(root, source_dir)
            subdirs.append(rel_path)
    return subdirs

def copy_files_for_split(source_dir, dest_dir, samples, weather, route_name):
    """Copy files belonging to samples to destination directory."""
    for subdir in tqdm(FOLDER_STRUCTURE, desc="  Processing folders"):
        src_subdir = os.path.join(source_dir, subdir)
        # New structure: dest_dir/weather/route_name/subdir
        dst_subdir = os.path.join(dest_dir, weather, route_name, subdir)
        
        # Always create the directory structure
        os.makedirs(dst_subdir, exist_ok=True)
        
        if not os.path.exists(src_subdir):
            continue
        
        files = [f for f in os.listdir(src_subdir) if os.path.isfile(os.path.join(src_subdir, f))]
        for filename in files:
            src_file = os.path.join(src_subdir, filename)
                
            # Check if file belongs to any sample
            file_base = os.path.splitext(filename)[0]
            if file_base in samples:
                dst_file = os.path.join(dst_subdir, filename)
                shutil.copy2(src_file, dst_file)

def main():
    source_dir = GlobalConfig.root_dir
    
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {CURRENT_DIR}")
    print(f"Weather: {WEATHER}")
    print(f"Route name: {ROUTE_NAME}")
    
    # Get all samples
    samples = get_all_samples(source_dir)
    print(f"Total samples found: {len(samples)}")
    
    # Split samples
    train_samples, test_samples, val_samples = split_samples(
        samples, TRAIN_RATIO, TEST_RATIO, VAL_RATIO
    )
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    # Create split directories in current directory
    splits = {
        'train': train_samples,
        'test': test_samples,
        'val': val_samples
    }
    
    for split_name, split_samples_list in splits.items():
        split_dir = os.path.join(CURRENT_DIR, split_name)
        print(f"\nCreating {split_name} split at: {split_dir}")
        
        if os.path.exists(split_dir):
            print(f"  Removing existing directory...")
            shutil.rmtree(split_dir)
        
        os.makedirs(split_dir, exist_ok=True)
        
        samples_set = set(split_samples_list)
        copy_files_for_split(source_dir, split_dir, samples_set, WEATHER, ROUTE_NAME)
        print(f"  Done copying {split_name} files.")
    
    print("\nDataset split complete!")
    print(f"\nFolder structure: <split>/{WEATHER}/{ROUTE_NAME}/<data_folders>")

if __name__ == "__main__":
    main()