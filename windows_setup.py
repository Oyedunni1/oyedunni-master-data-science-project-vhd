#!/usr/bin/env python3
"""
Windows-specific setup and configuration for VHD Prediction Project
This script addresses common Windows compatibility issues
"""

import os
import sys
import platform
import warnings

def setup_windows_environment():
    """Configure environment for Windows compatibility"""
    
    print("🖥️  Setting up Windows environment for VHD Prediction...")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version}")
    
    # Set environment variables for better Windows compatibility
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    os.environ['OMP_NUM_THREADS'] = '4'       # Limit OpenMP threads
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Allow duplicate libraries
    
    # Suppress warnings that are common on Windows
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    print("✓ Environment variables configured")

def check_tensorflow_compatibility():
    """Check TensorFlow/Keras compatibility on Windows"""
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU available: {len(gpus)} device(s)")
        else:
            print("ℹ️  No GPU detected, using CPU")
            
    except ImportError as e:
        print(f"❌ TensorFlow import error: {e}")
        return False
    
    return True

def check_dependencies():
    """Check if all required dependencies are available"""
    
    required_packages = [
        'numpy', 'pandas', 'scipy', 'librosa', 'scikit-learn', 
        'tensorflow', 'matplotlib', 'seaborn', 'streamlit', 
        'plotly', 'soundfile', 'opencv-python'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("✓ All dependencies available")
    return True

def create_windows_config():
    """Create Windows-specific configuration"""
    
    config = {
        'use_pretrained_weights': False,  # Disable by default on Windows
        'max_workers': 4,                 # Limit parallel processing
        'memory_limit_gb': 8,            # Set memory limit
        'enable_gpu': False,             # Disable GPU by default
        'fallback_to_cpu': True,         # Always fallback to CPU
    }
    
    # Save configuration
    import json
    with open('windows_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Windows configuration created")
    return config

def main():
    """Main setup function"""
    
    print("=" * 60)
    print("🖥️  VHD Prediction - Windows Setup")
    print("=" * 60)
    
    # Setup environment
    setup_windows_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return False
    
    # Check TensorFlow
    if not check_tensorflow_compatibility():
        print("\n❌ TensorFlow compatibility issues detected")
        return False
    
    # Create Windows config
    config = create_windows_config()
    
    print("\n" + "=" * 60)
    print("✅ Windows setup complete!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("1. Run: python train_model.py")
    print("2. Or run: streamlit run app.py")
    print("\n💡 Note: The system will automatically fallback to")
    print("   random initialization if ImageNet weights fail to load.")
    
    return True

if __name__ == "__main__":
    main()
