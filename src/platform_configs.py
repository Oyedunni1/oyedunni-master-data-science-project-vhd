"""
Platform-Specific Configuration System
Pre-configured settings for optimal performance on different platforms
"""

import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path

class PlatformConfigs:
    """
    Platform-specific configuration management
    Provides optimized settings for Windows, macOS, and Linux
    """
    
    def __init__(self):
        self.configs = {
            'windows': self._get_windows_config(),
            'macos': self._get_macos_config(),
            'linux': self._get_linux_config()
        }
    
    def _get_windows_config(self) -> Dict[str, Any]:
        """Windows-optimized configuration"""
        
        return {
            'platform': 'Windows',
            'optimization_level': 'stable',
            'memory_management': {
                'strategy': 'conservative',
                'limit_gb': 8,
                'cleanup_frequency': 5,
                'gc_threshold': 0.7,
                'fragmentation_handling': True,
                'virtual_memory': True
            },
            'cpu_optimization': {
                'max_workers': 8,
                'thread_affinity': True,
                'priority_class': 'NORMAL_PRIORITY_CLASS',
                'processor_affinity': True,
                'thread_pool_size': 4
            },
            'gpu_optimization': {
                'enabled': False,  # Disabled by default on Windows
                'memory_fraction': 0.6,
                'allow_growth': False,
                'device_placement': 'CPU'
            },
            'tensorflow_config': {
                'log_level': 'ERROR',
                'memory_growth': False,
                'inter_op_parallelism': 4,
                'intra_op_parallelism': 4,
                'cpu_optimization': True
            },
            'processing_config': {
                'batch_size': 50,
                'parallel_processing': True,
                'chunk_size': 1000,
                'memory_efficient': True,
                'error_handling': 'robust'
            },
            'environment_vars': {
                'TF_CPP_MIN_LOG_LEVEL': '2',
                'OMP_NUM_THREADS': '4',
                'KMP_DUPLICATE_LIB_OK': 'TRUE',
                'PYTHONHASHSEED': '0',
                'NUMBA_NUM_THREADS': '4'
            },
            'warnings': {
                'suppress_future': True,
                'suppress_user': True,
                'suppress_deprecation': True,
                'suppress_tensorflow': True
            },
            'file_handling': {
                'path_separator': '\\',
                'temp_dir': os.environ.get('TEMP', 'C:\\temp'),
                'cache_dir': os.environ.get('LOCALAPPDATA', 'C:\\Users\\temp'),
                'long_path_support': True
            },
            'performance': {
                'mode': 'stable',
                'aggressive_optimization': False,
                'memory_pressure_handling': True,
                'thermal_management': True
            }
        }
    
    def _get_macos_config(self) -> Dict[str, Any]:
        """macOS-optimized configuration"""
        
        return {
            'platform': 'macOS',
            'optimization_level': 'maximum',
            'memory_management': {
                'strategy': 'aggressive',
                'limit_gb': 32,
                'cleanup_frequency': 10,
                'gc_threshold': 0.8,
                'unified_memory': True,
                'memory_pressure_handling': True,
                'automatic_cleanup': True
            },
            'cpu_optimization': {
                'max_workers': 16,
                'thread_affinity': True,
                'performance_cores': True,
                'efficiency_cores': True,
                'thermal_management': True,
                'unified_cores': True
            },
            'gpu_optimization': {
                'enabled': True,
                'metal_performance': True,
                'unified_memory_gpu': True,
                'gpu_shared_memory': True,
                'memory_fraction': 0.9,
                'allow_growth': True
            },
            'tensorflow_config': {
                'log_level': 'INFO',
                'memory_growth': True,
                'inter_op_parallelism': 8,
                'intra_op_parallelism': 8,
                'cpu_optimization': True,
                'metal_optimization': True
            },
            'processing_config': {
                'batch_size': 200,
                'parallel_processing': True,
                'chunk_size': 2000,
                'memory_efficient': False,
                'aggressive_processing': True,
                'unified_memory_processing': True
            },
            'environment_vars': {
                'TF_CPP_MIN_LOG_LEVEL': '1',
                'OMP_NUM_THREADS': '16',
                'MKL_NUM_THREADS': '16',
                'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
                'PYTORCH_ENABLE_MPS_FALLBACK': '1'
            },
            'warnings': {
                'suppress_future': False,
                'suppress_user': False,
                'suppress_deprecation': False,
                'suppress_tensorflow': False
            },
            'file_handling': {
                'path_separator': '/',
                'temp_dir': '/tmp',
                'cache_dir': os.path.expanduser('~/Library/Caches'),
                'unified_memory_files': True
            },
            'performance': {
                'mode': 'aggressive',
                'aggressive_optimization': True,
                'unified_memory_optimization': True,
                'metal_performance_shaders': True,
                'thermal_management': True
            }
        }
    
    def _get_linux_config(self) -> Dict[str, Any]:
        """Linux-optimized configuration"""
        
        return {
            'platform': 'Linux',
            'optimization_level': 'balanced',
            'memory_management': {
                'strategy': 'balanced',
                'limit_gb': 24,
                'cleanup_frequency': 8,
                'gc_threshold': 0.75,
                'swap_optimization': True,
                'huge_pages': True
            },
            'cpu_optimization': {
                'max_workers': 12,
                'thread_affinity': True,
                'cpu_governor': 'performance',
                'irq_affinity': True,
                'numa_optimization': True
            },
            'gpu_optimization': {
                'enabled': True,
                'cuda_optimization': True,
                'opencl_optimization': True,
                'memory_fraction': 0.8,
                'allow_growth': True
            },
            'tensorflow_config': {
                'log_level': 'WARNING',
                'memory_growth': True,
                'inter_op_parallelism': 6,
                'intra_op_parallelism': 6,
                'cpu_optimization': True,
                'cuda_optimization': True
            },
            'processing_config': {
                'batch_size': 100,
                'parallel_processing': True,
                'chunk_size': 1500,
                'memory_efficient': True,
                'distributed_processing': True
            },
            'environment_vars': {
                'TF_CPP_MIN_LOG_LEVEL': '1',
                'OMP_NUM_THREADS': '12',
                'MKL_NUM_THREADS': '12',
                'CUDA_VISIBLE_DEVICES': '0',
                'PYTHONHASHSEED': '0'
            },
            'warnings': {
                'suppress_future': False,
                'suppress_user': False,
                'suppress_deprecation': False,
                'suppress_tensorflow': False
            },
            'file_handling': {
                'path_separator': '/',
                'temp_dir': '/tmp',
                'cache_dir': os.path.expanduser('~/.cache'),
                'tmpfs_optimization': True
            },
            'performance': {
                'mode': 'balanced',
                'aggressive_optimization': True,
                'distributed_optimization': True,
                'thermal_management': True
            }
        }
    
    def get_config(self, platform: str) -> Dict[str, Any]:
        """Get configuration for specific platform"""
        
        platform = platform.lower()
        if platform in self.configs:
            return self.configs[platform]
        else:
            # Return balanced config for unknown platforms
            return self._get_balanced_config()
    
    def _get_balanced_config(self) -> Dict[str, Any]:
        """Balanced configuration for unknown platforms"""
        
        return {
            'platform': 'Unknown',
            'optimization_level': 'balanced',
            'memory_management': {
                'strategy': 'balanced',
                'limit_gb': 8,
                'cleanup_frequency': 8,
                'gc_threshold': 0.75
            },
            'cpu_optimization': {
                'max_workers': 4,
                'thread_affinity': False
            },
            'gpu_optimization': {
                'enabled': False,
                'memory_fraction': 0.5
            },
            'processing_config': {
                'batch_size': 50,
                'parallel_processing': False,
                'chunk_size': 1000
            },
            'environment_vars': {
                'TF_CPP_MIN_LOG_LEVEL': '2',
                'OMP_NUM_THREADS': '4'
            },
            'warnings': {
                'suppress_future': True,
                'suppress_user': True,
                'suppress_deprecation': True
            }
        }
    
    def apply_config(self, config: Dict[str, Any]) -> None:
        """Apply platform configuration to system"""
        
        # Set environment variables
        env_vars = config.get('environment_vars', {})
        for key, value in env_vars.items():
            os.environ[key] = str(value)
        
        # Configure warnings
        warnings_config = config.get('warnings', {})
        if warnings_config.get('suppress_future'):
            import warnings
            warnings.filterwarnings('ignore', category=FutureWarning)
        if warnings_config.get('suppress_user'):
            warnings.filterwarnings('ignore', category=UserWarning)
        if warnings_config.get('suppress_deprecation'):
            warnings.filterwarnings('ignore', category=DeprecationWarning)
        if warnings_config.get('suppress_tensorflow'):
            warnings.filterwarnings('ignore', module='tensorflow')
    
    def get_optimal_settings(self, platform: str, resources: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal settings based on platform and available resources"""
        
        base_config = self.get_config(platform)
        
        # Adjust based on available resources
        memory_gb = resources.get('memory_total_gb', 8)
        cpu_count = resources.get('cpu_count', 4)
        gpu_available = resources.get('gpu_available', False)
        
        # Memory-based adjustments
        if memory_gb >= 32:
            base_config['memory_management']['limit_gb'] = min(32, memory_gb * 0.9)
            base_config['processing_config']['batch_size'] = 200
        elif memory_gb >= 16:
            base_config['memory_management']['limit_gb'] = min(16, memory_gb * 0.8)
            base_config['processing_config']['batch_size'] = 100
        elif memory_gb >= 8:
            base_config['memory_management']['limit_gb'] = min(8, memory_gb * 0.7)
            base_config['processing_config']['batch_size'] = 50
        else:
            base_config['memory_management']['limit_gb'] = min(4, memory_gb * 0.6)
            base_config['processing_config']['batch_size'] = 25
        
        # CPU-based adjustments
        if cpu_count >= 16:
            base_config['cpu_optimization']['max_workers'] = min(16, cpu_count)
        elif cpu_count >= 8:
            base_config['cpu_optimization']['max_workers'] = min(8, cpu_count)
        else:
            base_config['cpu_optimization']['max_workers'] = min(4, cpu_count)
        
        # GPU-based adjustments
        if gpu_available:
            base_config['gpu_optimization']['enabled'] = True
            base_config['processing_config']['batch_size'] *= 2
        
        return base_config
    
    def save_config(self, config: Dict[str, Any], filepath: str = None) -> None:
        """Save configuration to file"""
        
        if filepath is None:
            platform = config.get('platform', 'unknown').lower()
            filepath = f'config_{platform}.json'
        
        import json
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to {filepath}")
    
    def load_config(self, filepath: str) -> Dict[str, Any]:
        """Load configuration from file"""
        
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Configuration loaded from {filepath}")
        return config

# Global platform configs instance
platform_configs = PlatformConfigs()
