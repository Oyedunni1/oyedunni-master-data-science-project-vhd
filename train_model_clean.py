"""
Model Training Script for VHD Detection
Automated training pipeline with optimization for 99% accuracy
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline import VHDPredictionPipeline

def main():
    """Main training function"""
    print("Starting VHD Detection Model Training")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = VHDPredictionPipeline()
    
    try:
        # Step 1: Prepare data
        print("Preparing dataset...")
        df = pipeline.prepare_data(use_synthetic=False)  # Use real data
        
        if df is None or len(df) == 0:
            print("Error: No data available for training")
            return
        
        print(f"Dataset prepared: {len(df)} samples")
        
        # Step 2: Train model
        print("Training statistical models...")
        results = pipeline.train_model()
        
        if results and 'status' in results:
            print("Training completed successfully!")
            print(f"Status: {results['status']}")
            
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                print(f"Model Accuracy: {metrics.get('accuracy', 0):.1%}")
                print(f"Model Precision: {metrics.get('precision', 0):.1%}")
                print(f"Model Recall: {metrics.get('recall', 0):.1%}")
        else:
            print("Training failed or returned no results")
            
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
