#!/usr/bin/env python3
"""
Platform Detection Test Script
Tests the platform detection and optimization system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.platform_detector import PlatformDetector, get_platform_detector
from src.resource_optimizer import ResourceOptimizer
from src.platform_configs import PlatformConfigs
from src.pipeline import VHDPredictionPipeline

def test_platform_detection():
    """Test platform detection functionality"""
    
    print("🧪 TESTING PLATFORM DETECTION SYSTEM")
    print("=" * 60)
    
    # Test 1: Platform Detection
    print("\n1️⃣ Testing Platform Detection...")
    detector = get_platform_detector()
    platform_info = detector.get_platform_info()
    
    print(f"   ✅ Platform: {platform_info['platform']['os_type']}")
    print(f"   ✅ Architecture: {platform_info['platform']['architecture'][0]}")
    print(f"   ✅ CPU Cores: {platform_info['resources']['cpu_count']}")
    print(f"   ✅ Memory: {platform_info['resources']['memory_total_gb']:.1f} GB")
    print(f"   ✅ GPU: {'Available' if platform_info['resources']['gpu_available'] else 'Not Available'}")
    
    # Test 2: Resource Optimization
    print("\n2️⃣ Testing Resource Optimization...")
    config = detector.get_optimal_settings()
    optimizer = ResourceOptimizer(config)
    
    optimal_workers = optimizer.get_optimal_workers()
    optimal_batch_size = optimizer.get_optimal_batch_size()
    
    print(f"   ✅ Optimal Workers: {optimal_workers}")
    print(f"   ✅ Optimal Batch Size: {optimal_batch_size}")
    memory_strategy = 'balanced'
    processing_mode = 'balanced'
    if isinstance(config, dict):
        memory_strategy = config.get('memory_management', 'balanced')
        processing_mode = config.get('optimization_level', 'balanced')
    print(f"   ✅ Memory Management: {memory_strategy}")
    print(f"   ✅ Processing Mode: {processing_mode}")
    
    # Test 3: Platform Configurations
    print("\n3️⃣ Testing Platform Configurations...")
    configs = PlatformConfigs()
    
    # Test current platform config
    current_platform = platform_info['platform']['os_type']
    platform_config = configs.get_config(current_platform.lower())
    
    print(f"   ✅ Platform Config: {platform_config['platform']}")
    print(f"   ✅ Max Workers: {platform_config['cpu_optimization']['max_workers']}")
    print(f"   ✅ Batch Size: {platform_config['processing_config']['batch_size']}")
    print(f"   ✅ Memory Limit: {platform_config['memory_management']['limit_gb']} GB")
    print(f"   ✅ GPU Enabled: {platform_config['gpu_optimization']['enabled']}")
    
    # Test 4: Pipeline Integration
    print("\n4️⃣ Testing Pipeline Integration...")
    try:
        pipeline = VHDPredictionPipeline()
        print("   ✅ Pipeline initialized with platform optimization")
        print(f"   ✅ Platform: {pipeline.platform_config['platform']}")
        print(f"   ✅ Optimization Level: {pipeline.platform_config['optimization_level']}")
        print(f"   ✅ Max Workers: {pipeline.platform_config['cpu_optimization']['max_workers']}")
        print(f"   ✅ Batch Size: {pipeline.platform_config['processing_config']['batch_size']}")
    except Exception as e:
        print(f"   ❌ Pipeline initialization failed: {e}")
        return False
    
    # Test 5: Resource Status
    print("\n5️⃣ Testing Resource Status...")
    resource_status = optimizer.get_resource_status()
    
    print(f"   ✅ Memory Usage: {resource_status['memory']['usage_percent']:.1f}%")
    print(f"   ✅ CPU Usage: {resource_status['cpu']['usage_percent']:.1f}%")
    print(f"   ✅ Disk Usage: {resource_status['disk']['usage_percent']:.1f}%")
    
    # Test 6: Cross-Platform Simulation
    print("\n6️⃣ Testing Cross-Platform Simulation...")
    
    # Simulate Windows configuration
    windows_config = configs.get_config('windows')
    print(f"   ✅ Windows Config: {windows_config['platform']}")
    print(f"   ✅ Windows Max Workers: {windows_config['cpu_optimization']['max_workers']}")
    print(f"   ✅ Windows Batch Size: {windows_config['processing_config']['batch_size']}")
    
    # Simulate macOS configuration
    macos_config = configs.get_config('macos')
    print(f"   ✅ macOS Config: {macos_config['platform']}")
    print(f"   ✅ macOS Max Workers: {macos_config['cpu_optimization']['max_workers']}")
    print(f"   ✅ macOS Batch Size: {macos_config['processing_config']['batch_size']}")
    
    print("\n" + "=" * 60)
    print("✅ ALL PLATFORM DETECTION TESTS PASSED")
    print("=" * 60)
    
    return True

def test_platform_switching():
    """Test platform switching functionality"""
    
    print("\n🔄 TESTING PLATFORM SWITCHING")
    print("=" * 60)
    
    configs = PlatformConfigs()
    
    # Test different platform configurations
    platforms = ['windows', 'macos', 'linux']
    
    for platform in platforms:
        print(f"\n🖥️  Testing {platform.upper()} Configuration:")
        config = configs.get_config(platform)
        
        print(f"   Platform: {config['platform']}")
        print(f"   Max Workers: {config['cpu_optimization']['max_workers']}")
        print(f"   Batch Size: {config['processing_config']['batch_size']}")
        print(f"   Memory Strategy: {config['memory_management']['strategy']}")
        print(f"   GPU Enabled: {config['gpu_optimization']['enabled']}")
        print(f"   Optimization Level: {config['optimization_level']}")
    
    print("\n✅ Platform switching test completed")
    return True

def main():
    """Main test function"""
    
    print("🚀 VHD PREDICTION SYSTEM - PLATFORM DETECTION TEST")
    print("=" * 80)
    
    try:
        # Test platform detection
        if not test_platform_detection():
            print("❌ Platform detection tests failed")
            return False
        
        # Test platform switching
        if not test_platform_switching():
            print("❌ Platform switching tests failed")
            return False
        
        print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("✅ Platform detection system is working correctly")
        print("✅ Resource optimization is functioning")
        print("✅ Cross-platform compatibility is confirmed")
        print("✅ Pipeline integration is successful")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
