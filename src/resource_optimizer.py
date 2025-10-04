"""
Resource Optimization System
Automatically optimizes system resources based on platform detection
"""

import os
import sys
import gc
import psutil
import threading
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import warnings

class ResourceOptimizer:
    """
    Advanced resource optimization system
    Automatically manages memory, CPU, and GPU resources for optimal performance
    """
    
    def __init__(self, platform_config: Dict[str, Any]):
        self.config = platform_config
        self.memory_monitor = MemoryMonitor()
        self.cpu_monitor = CPUMonitor()
        self.gpu_monitor = GPUMonitor()
        self._setup_optimization()
    
    def _setup_optimization(self):
        """Setup resource optimization based on platform configuration"""
        
        # Configure memory management
        self._configure_memory_management()
        
        # Configure CPU optimization
        self._configure_cpu_optimization()
        
        # Configure GPU optimization
        self._configure_gpu_optimization()
        
        # Setup monitoring
        self._setup_monitoring()
    
    def _configure_memory_management(self):
        """Configure memory management based on platform"""
        
        memory_config = {
            'limit_gb': self.config.get('memory_limit_gb', 8),
            'optimization': self.config.get('memory_optimization', 'balanced'),
            'cleanup_frequency': 10,  # Cleanup every 10 operations
            'gc_threshold': 0.8,  # Trigger GC at 80% memory usage
        }
        
        if self.config.get('platform') == 'macOS':
            # Mac can handle more aggressive memory management
            memory_config.update({
                'unified_memory': True,
                'memory_pressure_handling': True,
                'automatic_cleanup': True
            })
        elif self.config.get('platform') == 'Windows':
            # Windows needs more conservative memory management
            memory_config.update({
                'conservative_allocation': True,
                'frequent_cleanup': True,
                'memory_fragmentation_handling': True
            })
        
        self.memory_config = memory_config
    
    def _configure_cpu_optimization(self):
        """Configure CPU optimization based on platform"""
        
        cpu_config = {
            'max_workers': self.config.get('max_workers', cpu_count()),
            'parallel_processing': self.config.get('parallel_processing', True),
            'thread_affinity': True,
            'cpu_usage_limit': 0.9,  # Don't exceed 90% CPU usage
        }
        
        if self.config.get('platform') == 'macOS':
            # Mac can handle more aggressive CPU usage
            cpu_config.update({
                'unified_cores': True,
                'performance_cores': True,
                'efficiency_cores': True,
                'thermal_management': True
            })
        elif self.config.get('platform') == 'Windows':
            # Windows needs more conservative CPU management
            cpu_config.update({
                'conservative_threading': True,
                'thread_pool_management': True,
                'cpu_priority_management': True
            })
        
        self.cpu_config = cpu_config
    
    def _configure_gpu_optimization(self):
        """Configure GPU optimization if available"""
        
        gpu_config = {
            'enabled': self.config.get('gpu_acceleration', False),
            'memory_fraction': 0.8,  # Use 80% of GPU memory
            'allow_growth': True,
            'mixed_precision': True,
        }
        
        if self.config.get('platform') == 'macOS' and self.config.get('metal_performance'):
            gpu_config.update({
                'metal_performance': True,
                'unified_memory_gpu': True,
                'gpu_shared_memory': True
            })
        
        self.gpu_config = gpu_config
    
    def _setup_monitoring(self):
        """Setup resource monitoring"""
        
        self.monitoring_enabled = True
        self.monitoring_interval = 5  # Check every 5 seconds
        
        # Start monitoring thread
        if self.monitoring_enabled:
            self._start_monitoring()
    
    def _start_monitoring(self):
        """Start resource monitoring in background"""
        
        def monitor_resources():
            while self.monitoring_enabled:
                try:
                    # Monitor memory usage
                    memory_usage = psutil.virtual_memory().percent
                    if memory_usage > 90:
                        self._trigger_memory_cleanup()
                    
                    # Monitor CPU usage
                    cpu_usage = psutil.cpu_percent(interval=1)
                    if cpu_usage > 95:
                        self._throttle_processing()
                    
                    # Monitor disk space
                    disk_usage = psutil.disk_usage('/').percent
                    if disk_usage > 95:
                        self._trigger_disk_cleanup()
                    
                except Exception as e:
                    print(f"⚠️  Monitoring error: {e}")
                
                threading.Event().wait(self.monitoring_interval)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
    
    def get_optimal_workers(self) -> int:
        """Get optimal number of workers based on current resources"""
        
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Base calculation
        if memory_gb >= 32:
            optimal_workers = min(cpu_count, 20)
        elif memory_gb >= 16:
            optimal_workers = min(cpu_count, 12)
        elif memory_gb >= 8:
            optimal_workers = min(cpu_count, 8)
        else:
            optimal_workers = min(cpu_count, 4)
        
        # Platform-specific adjustments
        if self.config.get('platform') == 'Windows':
            optimal_workers = min(optimal_workers, 8)  # Windows limit
        elif self.config.get('platform') == 'macOS':
            optimal_workers = min(optimal_workers, 16)  # Mac can handle more
        
        return max(1, optimal_workers)
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on available resources"""
        
        memory_gb = psutil.virtual_memory().available / (1024**3)
        cpu_count = psutil.cpu_count()
        
        # Base calculation
        if memory_gb >= 16:
            batch_size = 200
        elif memory_gb >= 8:
            batch_size = 100
        elif memory_gb >= 4:
            batch_size = 50
        else:
            batch_size = 25
        
        # Platform-specific adjustments
        if self.config.get('platform') == 'Windows':
            batch_size = min(batch_size, 100)  # Conservative for Windows
        elif self.config.get('platform') == 'macOS':
            batch_size = min(batch_size, 200)  # Mac can handle more
        
        return batch_size
    
    def optimize_memory_usage(self):
        """Optimize memory usage"""
        
        # Force garbage collection
        gc.collect()
        
        # Clear unused variables
        if hasattr(self, '_temp_objects'):
            del self._temp_objects
            self._temp_objects = []
        
        # Platform-specific memory optimization
        if self.config.get('platform') == 'macOS':
            self._optimize_mac_memory()
        elif self.config.get('platform') == 'Windows':
            self._optimize_windows_memory()
    
    def _optimize_mac_memory(self):
        """macOS-specific memory optimization"""
        
        # Use unified memory efficiently
        if self.config.get('unified_memory'):
            # Mac can share memory between CPU and GPU
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    def _optimize_windows_memory(self):
        """Windows-specific memory optimization"""
        
        # Windows memory management
        if self.config.get('memory_fragmentation_handling'):
            # Handle memory fragmentation
            gc.set_threshold(700, 10, 10)
    
    def _trigger_memory_cleanup(self):
        """Trigger memory cleanup when usage is high"""
        
        print("🧹 Triggering memory cleanup...")
        gc.collect()
        
        # Clear caches if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    def _throttle_processing(self):
        """Throttle processing when CPU usage is high"""
        
        print("⚡ Throttling processing due to high CPU usage...")
        # Reduce worker count temporarily
        self.cpu_config['max_workers'] = max(1, self.cpu_config['max_workers'] // 2)
    
    def _trigger_disk_cleanup(self):
        """Trigger disk cleanup when space is low"""
        
        print("💾 Triggering disk cleanup...")
        # Clean temporary files
        temp_dirs = ['/tmp', './temp', './cache']
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    for file in os.listdir(temp_dir):
                        if file.endswith('.tmp') or file.endswith('.cache'):
                            os.remove(os.path.join(temp_dir, file))
                except:
                    pass
    
    def get_executor(self, max_workers: int = None) -> ThreadPoolExecutor:
        """Get optimized thread pool executor"""
        
        if max_workers is None:
            max_workers = self.get_optimal_workers()
        
        return ThreadPoolExecutor(max_workers=max_workers)
    
    def get_process_executor(self, max_workers: int = None) -> ProcessPoolExecutor:
        """Get optimized process pool executor"""
        
        if max_workers is None:
            max_workers = min(self.get_optimal_workers(), 4)  # Limit processes
        
        return ProcessPoolExecutor(max_workers=max_workers)
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_enabled = False
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        return {
            'memory': {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'usage_percent': memory.percent
            },
            'cpu': {
                'usage_percent': cpu_percent,
                'count': psutil.cpu_count()
            },
            'disk': {
                'total_gb': disk.total / (1024**3),
                'free_gb': disk.free / (1024**3),
                'usage_percent': (disk.used / disk.total) * 100
            }
        }

class MemoryMonitor:
    """Memory usage monitoring and optimization"""
    
    def __init__(self):
        self.peak_usage = 0
        self.cleanup_threshold = 0.8
    
    def get_memory_usage(self) -> float:
        """Get current memory usage as percentage"""
        return psutil.virtual_memory().percent / 100
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        return self.get_memory_usage() > self.cleanup_threshold
    
    def cleanup(self):
        """Perform memory cleanup"""
        import gc
        gc.collect()

class CPUMonitor:
    """CPU usage monitoring and optimization"""
    
    def __init__(self):
        self.peak_usage = 0
        self.throttle_threshold = 0.9
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage as percentage"""
        return psutil.cpu_percent(interval=1) / 100
    
    def should_throttle(self) -> bool:
        """Check if CPU throttling is needed"""
        return self.get_cpu_usage() > self.throttle_threshold

class GPUMonitor:
    """GPU usage monitoring and optimization"""
    
    def __init__(self):
        self.available = False
        self.memory_usage = 0
    
    def is_available(self) -> bool:
        """Check if GPU is available"""
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except:
            return False
    
    def get_memory_usage(self) -> float:
        """Get GPU memory usage"""
        try:
            import tensorflow as tf
            # This would need GPU-specific implementation
            return 0.0
        except:
            return 0.0
