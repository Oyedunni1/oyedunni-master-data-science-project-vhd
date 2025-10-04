"""
Data acquisition and management for VHD detection.
Handles PhysioNet dataset download and organization.
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import wfdb
import soundfile as sf
from pathlib import Path
import warnings
import requests
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')

class PhysioNetDataManager:
    """
    Manages PhysioNet CinC Challenge 2016 dataset acquisition and organization.
    """
    
    def __init__(self, data_dir: str = "data", base_url: str = "https://archive.physionet.org/challenge/2016/"):
        self.data_dir = Path(data_dir)
        self.base_url = base_url
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.labels_file = self.data_dir / "labels.csv"
        
        # PhysioNet CinC Challenge 2016 specific URLs
        self.dataset_urls = {
            'training_a': 'https://archive.physionet.org/challenge/2016/training-a.zip',
            'training_b': 'https://archive.physionet.org/challenge/2016/training-b.zip', 
            'training_c': 'https://archive.physionet.org/challenge/2016/training-c.zip',
            'training_d': 'https://archive.physionet.org/challenge/2016/training-d.zip',
            'training_e': 'https://archive.physionet.org/challenge/2016/training-e.zip',
            'training_f': 'https://archive.physionet.org/challenge/2016/training-f.zip',
            'validation': 'https://archive.physionet.org/challenge/2016/validation.zip',
            'test': 'https://archive.physionet.org/challenge/2016/test.zip'
        }
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
    
    def download_dataset(self, download_all: bool = False) -> bool:
        """
        Download PhysioNet CinC Challenge 2016 dataset.
        
        Args:
            download_all: If True, download all training sets (a-f). 
                         If False, download only training-a and validation.
        """
        try:
            print("Downloading PhysioNet CinC Challenge 2016 dataset...")
            
            # Determine which datasets to download
            if download_all:
                datasets_to_download = list(self.dataset_urls.keys())
            else:
                datasets_to_download = ['training_a', 'validation']
            
            downloaded_files = []
            
            for dataset_name in datasets_to_download:
                url = self.dataset_urls[dataset_name]
                zip_filename = f"{dataset_name.replace('_', '-')}.zip"
                zip_path = self.data_dir / zip_filename
                
                if not zip_path.exists():
                    print(f"Downloading {dataset_name}...")
                    success = self._download_with_progress(url, zip_path)
                    if success:
                        downloaded_files.append((zip_path, dataset_name))
                        print(f"{dataset_name} downloaded successfully.")
                    else:
                        print(f"Failed to download {dataset_name}")
                else:
                    print(f"{dataset_name} already exists, skipping...")
                    downloaded_files.append((zip_path, dataset_name))
            
            # Extract downloaded files
            for zip_path, dataset_name in downloaded_files:
                extract_dir = self.raw_dir / dataset_name.replace('_', '-')
                print(f"Extracting {dataset_name}...")
                self._extract_zip(zip_path, extract_dir)
                print(f"{dataset_name} extracted successfully.")
            
            print("Dataset download and extraction completed.")
            return True
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    def _download_with_progress(self, url: str, filepath: Path) -> bool:
        """
        Download file with progress bar.
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=f"Downloading {filepath.name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def _extract_zip(self, zip_path: Path, extract_dir: Path):
        """Extract zip file to directory."""
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    def create_labels_dataframe(self) -> pd.DataFrame:
        """
        Create comprehensive labels dataframe from all training and validation directories.
        """
        labels_data = []
        
        # Process all training directories
        training_dirs = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']
        for dir_name in training_dirs:
            training_dir = self.raw_dir / dir_name
            if training_dir.exists():
                ref_file = training_dir / 'REFERENCE.csv'
                if ref_file.exists():
                    # Use REFERENCE.csv for accurate labels
                    ref_df = pd.read_csv(ref_file, header=None, names=['filename', 'label'])
                    for _, row in ref_df.iterrows():
                        filename = row['filename']
                        label = 1 if row['label'] == 1 else 0  # Convert -1 to 0, 1 to 1
                        filepath = training_dir / f"{filename}.wav"
                        if filepath.exists():
                            labels_data.append({
                                'filename': filename,
                                'filepath': str(filepath),
                                'split': 'training',
                                'label': label
                            })
                else:
                    # Fallback to filename-based labeling
                    for file_path in training_dir.glob("*.wav"):
                        filename = file_path.stem
                        labels_data.append({
                            'filename': filename,
                            'filepath': str(file_path),
                            'split': 'training',
                            'label': self._determine_label_from_filename(filename)
                        })
        
        # Process validation data
        validation_dir = self.raw_dir / "validation"
        if validation_dir.exists():
            ref_file = validation_dir / 'REFERENCE.csv'
            if ref_file.exists():
                # Use REFERENCE.csv for accurate labels
                ref_df = pd.read_csv(ref_file, header=None, names=['filename', 'label'])
                for _, row in ref_df.iterrows():
                    filename = row['filename']
                    label = 1 if row['label'] == 1 else 0  # Convert -1 to 0, 1 to 1
                    filepath = validation_dir / f"{filename}.wav"
                    if filepath.exists():
                        labels_data.append({
                            'filename': filename,
                            'filepath': str(filepath),
                            'split': 'validation',
                            'label': label
                        })
            else:
                # Fallback to filename-based labeling
                for file_path in validation_dir.glob("*.wav"):
                    filename = file_path.stem
                    labels_data.append({
                        'filename': filename,
                        'filepath': str(file_path),
                        'split': 'validation',
                        'label': self._determine_label_from_filename(filename)
                    })
        
        df = pd.DataFrame(labels_data)
        df.to_csv(self.labels_file, index=False)
        print(f"Created labels file with {len(df)} samples")
        return df
    
    def _determine_label_from_filename(self, filename: str) -> int:
        """
        Determine label from filename patterns.
        Normal = 0, Abnormal = 1.
        """
        # Common patterns for abnormal heart sounds
        abnormal_patterns = ['murmur', 'abnormal', 'pathological', 'disease']
        normal_patterns = ['normal', 'healthy', 'control']
        
        filename_lower = filename.lower()
        
        for pattern in abnormal_patterns:
            if pattern in filename_lower:
                return 1
        
        for pattern in normal_patterns:
            if pattern in filename_lower:
                return 0
        
        # Default to normal if unclear
        return 0
    
    def create_synthetic_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Create synthetic heart sound data for testing when real data is unavailable.
        """
        print(f"Creating {num_samples} synthetic heart sound samples...")
        
        synthetic_data = []
        sample_rate = 2000
        
        for i in range(num_samples):
            # Generate synthetic heart sound
            duration = np.random.uniform(5, 15)  # 5-15 seconds
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Generate S1 and S2 sounds
            s1_freq = np.random.uniform(20, 50)  # S1 frequency
            s2_freq = np.random.uniform(15, 40)   # S2 frequency
            
            # Create heart rhythm
            heart_rate = np.random.uniform(60, 100)  # BPM
            beat_interval = 60 / heart_rate
            
            signal = np.zeros_like(t)
            
            # Add S1 and S2 sounds
            for beat_time in np.arange(0, duration, beat_interval):
                if beat_time < duration:
                    # S1 sound
                    s1_start = int(beat_time * sample_rate)
                    s1_end = min(s1_start + int(0.1 * sample_rate), len(signal))
                    if s1_start < len(signal) and s1_end <= len(signal):
                        s1_signal = np.sin(2 * np.pi * s1_freq * t[s1_start:s1_end]) * np.exp(-t[s1_start:s1_end] * 10)
                        signal[s1_start:s1_end] += s1_signal
                    
                    # S2 sound
                    s2_start = int((beat_time + 0.3) * sample_rate)
                    s2_end = min(s2_start + int(0.1 * sample_rate), len(signal))
                    if s2_start < len(signal) and s2_end <= len(signal):
                        s2_signal = np.sin(2 * np.pi * s2_freq * t[s2_start:s2_end]) * np.exp(-t[s2_start:s2_end] * 10)
                        signal[s2_start:s2_end] += s2_signal
            
            # Add noise
            noise_level = np.random.uniform(0.01, 0.05)
            signal += np.random.normal(0, noise_level, len(signal))
            
            # Add murmurs for abnormal cases
            label = np.random.choice([0, 1], p=[0.7, 0.3])  # 70% normal, 30% abnormal
            
            if label == 1:  # Abnormal
                # Add murmur
                murmur_freq = np.random.uniform(100, 300)
                murmur_signal = np.sin(2 * np.pi * murmur_freq * t) * 0.1
                signal += murmur_signal
            
            # Normalize
            signal = signal / np.max(np.abs(signal))
            
            # Save synthetic audio file
            filename = f"synthetic_{i:04d}.wav"
            filepath = self.processed_dir / filename
            sf.write(filepath, signal, sample_rate)
            
            synthetic_data.append({
                'filename': filename,
                'filepath': str(filepath),
                'split': 'synthetic',
                'label': label
            })
        
        df = pd.DataFrame(synthetic_data)
        df.to_csv(self.labels_file, index=False)
        print(f"Created synthetic dataset with {len(df)} samples")
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get summary statistics of the dataset.
        """
        summary = {
            'total_samples': len(df),
            'normal_samples': len(df[df['label'] == 0]),
            'abnormal_samples': len(df[df['label'] == 1]),
            'train_samples': len(df[df['split'] == 'train']),
            'validation_samples': len(df[df['split'] == 'validation']),
            'synthetic_samples': len(df[df['split'] == 'synthetic']),
            'class_balance': len(df[df['label'] == 1]) / len(df)
        }
        
        print("Dataset Summary:")
        print(f"Total samples: {summary['total_samples']}")
        print(f"Normal samples: {summary['normal_samples']}")
        print(f"Abnormal samples: {summary['abnormal_samples']}")
        print(f"Class balance: {summary['class_balance']:.2%}")
        
        return summary
    
    def load_labels(self) -> pd.DataFrame:
        """
        Load existing labels dataframe.
        """
        if self.labels_file.exists():
            return pd.read_csv(self.labels_file)
        else:
            print("Labels file not found. Please run create_labels_dataframe() first.")
            return pd.DataFrame()
    
    def prepare_dataset(self, use_synthetic: bool = False) -> pd.DataFrame:
        """
        Prepare dataset using manually downloaded data only.
        """
        print("Using manually downloaded PhysioNet dataset...")
        
        # Check if manual data exists
        if self._check_manual_data_exists():
            print("✅ Manual data found, creating labels...")
            return self.create_labels_dataframe()
        else:
            print("❌ Manual data not found. Please follow the manual download guide.")
            self.print_manual_download_instructions()
            return pd.DataFrame()
    
    def _check_manual_data_exists(self) -> bool:
        """
        Check if manually downloaded data exists in the expected structure.
        """
        required_dirs = [
            'training-a', 'training-b', 'training-c', 
            'training-d', 'training-e', 'training-f', 'validation'
        ]
        
        for dir_name in required_dirs:
            dir_path = self.raw_dir / dir_name
            if not dir_path.exists():
                print(f"❌ Missing directory: {dir_name}")
                return False
            
            # Check for .wav files
            wav_files = list(dir_path.glob('*.wav'))
            if len(wav_files) == 0:
                print(f"❌ No .wav files found in {dir_name}")
                return False
                
        print("✅ All required directories and files found")
        return True
    
    def print_manual_download_instructions(self):
        """
        Print detailed manual download instructions.
        """
        print("\n" + "="*80)
        print("MANUAL DOWNLOAD INSTRUCTIONS FOR PHYSIONET CINC CHALLENGE 2016")
        print("="*80)
        print("\n1. VISIT THE OFFICIAL WEBSITE:")
        print("   https://archive.physionet.org/challenge/2016/")
        print("\n2. DOWNLOAD THE FOLLOWING FILES:")
        print("   - training-a.zip (Training set A)")
        print("   - training-b.zip (Training set B)")  
        print("   - training-c.zip (Training set C)")
        print("   - training-d.zip (Training set D)")
        print("   - training-e.zip (Training set E)")
        print("   - training-f.zip (Training set F)")
        print("   - validation.zip (Validation set)")
        print("   - test.zip (Test set - optional)")
        print("\n3. PLACE DOWNLOADED FILES IN YOUR PROJECT:")
        print(f"   {self.data_dir.absolute()}/")
        print("\n4. EXTRACT THE FILES:")
        print("   - Extract training-a.zip to: data/raw/training-a/")
        print("   - Extract training-b.zip to: data/raw/training-b/")
        print("   - Extract training-c.zip to: data/raw/training-c/")
        print("   - Extract training-d.zip to: data/raw/training-d/")
        print("   - Extract training-e.zip to: data/raw/training-e/")
        print("   - Extract training-f.zip to: data/raw/training-f/")
        print("   - Extract validation.zip to: data/raw/validation/")
        print("\n5. RUN THE DATA PREPARATION SCRIPT:")
        print("   python -c \"from src.data_acquisition import PhysioNetDataManager; dm = PhysioNetDataManager(); dm.create_labels_dataframe()\"")
        print("\n6. VERIFY THE DATA:")
        print("   Check that data/raw/ contains subdirectories with .wav files")
        print("   Check that data/labels.csv was created successfully")
        print("\n" + "="*80)
    
    def verify_data_structure(self) -> bool:
        """
        Verify that the data structure is correct.
        """
        print("Verifying data structure...")
        
        required_dirs = [
            self.raw_dir / "training-a",
            self.raw_dir / "validation"
        ]
        
        all_good = True
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"Missing directory: {dir_path}")
                all_good = False
            else:
                wav_files = list(dir_path.glob("*.wav"))
                print(f"Found {dir_path.name}: {len(wav_files)} .wav files")
        
        if self.labels_file.exists():
            df = pd.read_csv(self.labels_file)
            print(f"Labels file: {len(df)} samples")
        else:
            print("Labels file not found")
            all_good = False
        
        return all_good
    
    def get_download_status(self) -> Dict[str, bool]:
        """
        Check which datasets are already downloaded.
        """
        status = {}
        for dataset_name, url in self.dataset_urls.items():
            zip_filename = f"{dataset_name.replace('_', '-')}.zip"
            zip_path = self.data_dir / zip_filename
            status[dataset_name] = zip_path.exists()
        
        return status
