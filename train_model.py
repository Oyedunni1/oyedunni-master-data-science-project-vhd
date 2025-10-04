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
        results = pipeline.train_model(df)
        
        if results and 'results' in results:
            print("Training completed successfully!")
            print("Model training pipeline completed")
            
            # Extract performance metrics from results
            if 'results' in results and results['results']:
                metrics = results['results']
                print(f"Model Accuracy: {metrics.get('accuracy', 0):.1%}")
                print(f"Model Precision: {metrics.get('precision', 0):.1%}")
                print(f"Model Recall: {metrics.get('recall', 0):.1%}")
                print(f"Model F1 Score: {metrics.get('f1_score', 0):.1%}")
                
                # Show feature information
                if 'feature_shapes' in results:
                    shapes = results['feature_shapes']
                    print(f"Fractal features shape: {shapes.get('fractal', 'N/A')}")
                    print(f"Deep features shape: {shapes.get('deep', 'N/A')}")
                
                print("\nTraining completed successfully!")
                print("Model is ready for use in the web application")
        else:
            print("Training failed or returned no results")
            
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
