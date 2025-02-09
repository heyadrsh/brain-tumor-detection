import os
import shutil
import random
from tqdm import tqdm

def create_directory_structure():
    """Create train and validation directories"""
    base_dir = os.path.join('data', 'CT')
    directories = [
        os.path.join(base_dir, 'train/aneurysm'), 
        os.path.join(base_dir, 'train/cancer'), 
        os.path.join(base_dir, 'train/tumor'),
        os.path.join(base_dir, 'train/normal'),
        os.path.join(base_dir, 'val/aneurysm'), 
        os.path.join(base_dir, 'val/cancer'), 
        os.path.join(base_dir, 'val/tumor'),
        os.path.join(base_dir, 'val/normal')
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
    """Split data into train and validation sets"""
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm'))]
    random.shuffle(files)
    
    split_idx = int(len(files) * split_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    # Copy files to train directory
    for file in tqdm(train_files, desc=f'Copying train files from {os.path.basename(source_dir)}'):
        src = os.path.join(source_dir, file)
        dst = os.path.join(train_dir, file)
        shutil.copy2(src, dst)
    
    # Copy files to validation directory
    for file in tqdm(val_files, desc=f'Copying validation files from {os.path.basename(source_dir)}'):
        src = os.path.join(source_dir, file)
        dst = os.path.join(val_dir, file)
        shutil.copy2(src, dst)
    
    return len(train_files), len(val_files)

def main():
    # Create directory structure
    create_directory_structure()
    
    # Process each category
    categories = ['aneurysm', 'cancer', 'tumor', 'normal']
    base_dir = os.path.join('data', 'CT')
    
    for category in categories:
        source_dir = os.path.join(base_dir, category)
        train_dir = os.path.join(base_dir, 'train', category)
        val_dir = os.path.join(base_dir, 'val', category)
        
        if os.path.exists(source_dir):
            print(f'\nProcessing {category} images...')
            train_count, val_count = split_data(source_dir, train_dir, val_dir)
            print(f'Split {category}: {train_count} training, {val_count} validation images')

if __name__ == '__main__':
    main() 