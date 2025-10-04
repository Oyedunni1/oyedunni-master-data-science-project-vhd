"""
Platform Detection and Resource Optimization System
Automatically detects platform and configures optimal settings for Windows/Mac
"""

import os
import sys
import platform
import psutil
import multiprocessing
import warnings
from typing import Dict, Any, Tuple
import json
from pathlib import Path

class PlatformDetector:
    """
    Advanced platform detection and resource optimization system
    Automatically configures the VHD system for optimal performance
    """
    
    def __init__(self):
        self.platform_info = self._detect_platform()
        self.resource_info = self._detect_resources()
        self.optimal_config = self._generate_optimal_config()
        self._apply_platform_optimizations()
    
    def _detect_platform(self) -> Dict[str, Any]:
        """Detect current platform with detailed information"""
        
        platform_info = {
            'system': platform.system(),
            'platform': sys.platform,
            'architecture': platform.architecture(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'python_executable': sys.executable
        }
        
        # Enhanced platform detection
        if sys.platform.startswith('win'):
            platform_info['os_type'] = 'Windows'
            platform_info['is_windows'] = True
            platform_info['is_mac'] = False
            platform_info['is_linux'] = False
        elif sys.platform.startswith('darwin'):
            platform_info['os_type'] = 'macOS'
            platform_info['is_windows'] = False
            platform_info['is_mac'] = True
            platform_info['is_linux'] = False
        elif sys.platform.startswith('linux'):
            platform_info['os_type'] = 'Linux'
            platform_info['is_windows'] = False
            platform_info['is_mac'] = False
            platform_info['is_linux'] = True
        else:
            platform_info['os_type'] = 'Unknown'
            platform_info['is_windows'] = False
            platform_info['is_mac'] = False
            platform_info['is_linux'] = False
        
        return platform_info
    
    def _detect_resources(self) -> Dict[str, Any]:
        """Detect available system resources"""
        
        try:
            # CPU Information
            cpu_count = multiprocessing.cpu_count()
            cpu_freq = psutil.cpu_freq()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory Information
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            available_memory_gb = memory.available / (1024**3)
            
            # Disk Information
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024**3)
            
            # GPU Detection (if available)
            gpu_available = self._detect_gpu()
            
            resource_info = {
                'cpu_count': cpu_count,
                'cpu_frequency': cpu_freq.max if cpu_freq else None,
                'cpu_usage': cpu_percent,
                'memory_total_gb': round(memory_gb, 2),
                'memory_available_gb': round(available_memory_gb, 2),
                'memory_usage_percent': memory.percent,
                'disk_free_gb': round(disk_free_gb, 2),
                'gpu_available': gpu_available,
                'gpu_count': len(gpu_available) if gpu_available else 0
            }
            
            return resource_info
            
        except Exception as e:
            print(f"⚠️  Resource detection failed: {e}")
            # Fallback to basic detection
            return {
                'cpu_count': multiprocessing.cpu_count(),
                'memory_total_gb': 8.0,  # Conservative estimate
                'memory_available_gb': 4.0,
                'gpu_available': False,
                'gpu_count': 0
            }
    
    def _detect_gpu(self) -> bool:
        """Detect GPU availability"""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            return len(gpus) > 0
        except:
            try:
                import torch
                return torch.cuda.is_available()
            except:
                return False
    
    def _generate_optimal_config(self) -> Dict[str, Any]:
        """Generate optimal configuration based on platform and resources"""
        
        config = {
            'platform': self.platform_info['os_type'],
            'optimization_level': 'maximum',
            'memory_management': 'aggressive',
            'parallel_processing': True,
            'gpu_acceleration': False,
            'batch_processing': True,
            'feature_extraction': 'ultra_fast',
            'model_loading': 'optimized'
        }
        
        # Platform-specific optimizations
        if self.platform_info['is_windows']:
            config.update(self._get_windows_config())
        elif self.platform_info['is_mac']:
            config.update(self._get_mac_config())
        else:
            config.update(self._get_linux_config())
        
        # Resource-based optimizations
        config.update(self._get_resource_based_config())
        
        return config
    
    def _get_windows_config(self) -> Dict[str, Any]:
        """Windows-specific optimizations"""
        
        return {
            'max_workers': min(self.resource_info['cpu_count'], 8),  # Windows limit
            'memory_limit_gb': min(self.resource_info['memory_available_gb'] * 0.8, 16),
            'batch_size': 50,  # Conservative for Windows
            'use_pretrained_weights': False,  # Disable by default on Windows
            'tensorflow_logging': 'minimal',
            'openmp_threads': 4,
            'enable_gpu': False,  # Conservative approach
            'fallback_to_cpu': True,
            'environment_vars': {
                'TF_CPP_MIN_LOG_LEVEL': '2',
                'OMP_NUM_THREADS': '4',
                'KMP_DUPLICATE_LIB_OK': 'TRUE'
            },
            'warnings_suppression': True,
            'memory_optimization': 'aggressive',
            'processing_mode': 'stable'
        }
    
    def _get_mac_config(self) -> Dict[str, Any]:
        """macOS-specific optimizations"""
        
        # Mac can handle more aggressive settings
        max_workers = min(self.resource_info['cpu_count'], 16)  # Mac can handle more
        
        return {
            'max_workers': max_workers,
            'memory_limit_gb': min(self.resource_info['memory_available_gb'] * 0.9, 32),
            'batch_size': 100,  # More aggressive on Mac
            'use_pretrained_weights': True,  # Enable on Mac
            'tensorflow_logging': 'normal',
            'openmp_threads': max_workers,
            'enable_gpu': self.resource_info['gpu_available'],
            'fallback_to_cpu': not self.resource_info['gpu_available'],
            'environment_vars': {
                'TF_CPP_MIN_LOG_LEVEL': '1',
                'OMP_NUM_THREADS': str(max_workers),
                'MKL_NUM_THREADS': str(max_workers)
            },
            'warnings_suppression': False,
            'memory_optimization': 'balanced',
            'processing_mode': 'aggressive',
            'unified_memory': True,  # Mac advantage
            'metal_performance': True  # Mac GPU acceleration
        }
    
    def _get_linux_config(self) -> Dict[str, Any]:
        """Linux-specific optimizations"""
        
        return {
            'max_workers': min(self.resource_info['cpu_count'], 12),
            'memory_limit_gb': min(self.resource_info['memory_available_gb'] * 0.85, 24),
            'batch_size': 75,
            'use_pretrained_weights': True,
            'enable_gpu': self.resource_info['gpu_available'],
            'processing_mode': 'balanced'
        }
    
    def _get_resource_based_config(self) -> Dict[str, Any]:
        """Optimize based on available resources"""
        
        config = {}
        
        # Memory-based optimizations
        memory_gb = self.resource_info['memory_total_gb']
        if memory_gb >= 32:
            config['memory_optimization'] = 'maximum'
            config['batch_size'] = 200
            config['max_workers'] = min(self.resource_info['cpu_count'], 20)
        elif memory_gb >= 16:
            config['memory_optimization'] = 'high'
            config['batch_size'] = 100
            config['max_workers'] = min(self.resource_info['cpu_count'], 12)
        elif memory_gb >= 8:
            config['memory_optimization'] = 'balanced'
            config['batch_size'] = 50
            config['max_workers'] = min(self.resource_info['cpu_count'], 8)
        else:
            config['memory_optimization'] = 'conservative'
            config['batch_size'] = 25
            config['max_workers'] = min(self.resource_info['cpu_count'], 4)
        
        # CPU-based optimizations
        cpu_count = self.resource_info['cpu_count']
        if cpu_count >= 16:
            config['parallel_processing'] = True
            config['feature_extraction'] = 'ultra_fast'
        elif cpu_count >= 8:
            config['parallel_processing'] = True
            config['feature_extraction'] = 'fast'
        else:
            config['parallel_processing'] = False
            config['feature_extraction'] = 'standard'
        
        # GPU optimizations
        if self.resource_info['gpu_available']:
            config['gpu_acceleration'] = True
            config['batch_size'] = min(config.get('batch_size', 50) * 2, 200)
            config['model_loading'] = 'gpu_optimized'
        
        return config
    
    def _apply_platform_optimizations(self):
        """Apply platform-specific optimizations immediately"""
        
        # Set environment variables
        env_vars = self.optimal_config.get('environment_vars', {})
        for key, value in env_vars.items():
            os.environ[key] = str(value)
        
        # Suppress warnings if configured
        if self.optimal_config.get('warnings_suppression', False):
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=DeprecationWarning)
        
        # Configure TensorFlow if available
        if self.optimal_config.get('tensorflow_logging'):
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' if self.optimal_config['tensorflow_logging'] == 'minimal' else '1'
    
    def get_platform_info(self) -> Dict[str, Any]:
        """Get detailed platform information"""
        return {
            'platform': self.platform_info,
            'resources': self.resource_info,
            'configuration': self.optimal_config
        }
    
    def get_optimal_settings(self) -> Dict[str, Any]:
        """Get optimal settings for the current platform"""
        return self.optimal_config
    
    def save_configuration(self, filepath: str = None):
        """Save configuration to file"""
        if filepath is None:
            filepath = 'platform_config.json'
        
        config_data = {
            'platform_info': self.platform_info,
            'resource_info': self.resource_info,
            'optimal_config': self.optimal_config,
            'timestamp': str(Path().cwd())
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Platform configuration saved to {filepath}")
    
    def print_platform_summary(self):
        """Print comprehensive platform summary"""
        print("=" * 80)
        print("🖥️  VHD PREDICTION SYSTEM - PLATFORM DETECTION")
        print("=" * 80)
        
        print(f"\n📊 PLATFORM INFORMATION:")
        print(f"   Operating System: {self.platform_info['os_type']}")
        print(f"   Architecture: {self.platform_info['architecture'][0]}")
        print(f"   Python Version: {sys.version.split()[0]}")
        print(f"   Python Executable: {self.platform_info['python_executable']}")
        
        print(f"\n💻 SYSTEM RESOURCES:")
        print(f"   CPU Cores: {self.resource_info['cpu_count']}")
        print(f"   Total Memory: {self.resource_info['memory_total_gb']:.1f} GB")
        print(f"   Available Memory: {self.resource_info['memory_available_gb']:.1f} GB")
        print(f"   Memory Usage: {self.resource_info['memory_usage_percent']:.1f}%")
        print(f"   GPU Available: {'Yes' if self.resource_info['gpu_available'] else 'No'}")
        
        print(f"\n⚙️  OPTIMAL CONFIGURATION:")
        print(f"   Max Workers: {self.optimal_config['max_workers']}")
        print(f"   Batch Size: {self.optimal_config['batch_size']}")
        print(f"   Memory Limit: {self.optimal_config['memory_limit_gb']:.1f} GB")
        print(f"   GPU Acceleration: {'Enabled' if self.optimal_config.get('gpu_acceleration', False) else 'Disabled'}")
        print(f"   Parallel Processing: {'Enabled' if self.optimal_config['parallel_processing'] else 'Disabled'}")
        print(f"   Processing Mode: {self.optimal_config.get('processing_mode', 'standard').title()}")
        
        print(f"\n🚀 PERFORMANCE OPTIMIZATIONS:")
        optimizations = []
        if self.optimal_config.get('unified_memory'):
            optimizations.append("Unified Memory Architecture")
        if self.optimal_config.get('metal_performance'):
            optimizations.append("Metal Performance Shaders")
        if self.optimal_config.get('gpu_acceleration'):
            optimizations.append("GPU Acceleration")
        if self.optimal_config.get('parallel_processing'):
            optimizations.append("Multi-threaded Processing")
        if self.optimal_config.get('batch_processing'):
            optimizations.append("Batch Processing")
        
        for opt in optimizations:
            print(f"   ✓ {opt}")
        
        print("\n" + "=" * 80)
        print("✅ PLATFORM DETECTION COMPLETE - SYSTEM OPTIMIZED")
        print("=" * 80)

def get_platform_detector() -> PlatformDetector:
    """Get singleton platform detector instance"""
    if not hasattr(get_platform_detector, '_instance'):
        get_platform_detector._instance = PlatformDetector()
    return get_platform_detector._instance

# Global platform detector instance
platform_detector = get_platform_detector()
