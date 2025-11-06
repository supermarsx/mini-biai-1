#!/usr/bin/env python3
"""
Edge Deployment Optimization Demo

Comprehensive demonstration of the edge deployment optimization system
for mobile and embedded devices.

This demo showcases:
- Device detection and capability analysis
- Model quantization and optimization
- Platform-specific acceleration
- Progressive loading
- Power and battery optimization
- WebAssembly deployment
"""

import os
import sys
import time
import logging
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.edge import (
        EdgeDevice, EdgePlatform, OptimizationConfig, 
        QuantizationType, AccelerationBackend, 
        create_edge_deployer, benchmark_edge_deployment,
        get_supported_devices
    )
    from src.edge.quantization import Quantizer
    from src.edge.optimization import MobileOptimizer
    from src.edge.webassembly import WebAssemblyDeployer, WASMConfig
    from src.edge.battery import BatteryOptimizer, BatteryOptimizationConfig
    from src.edge.accelerator import EdgeAccelerator
    from src.edge.tensorrt import TensorRTOptimizer
    from src.edge.progressive import ProgressiveLoader, ProgressiveConfig
    EDGE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some edge modules not available: {e}")
    EDGE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_device_detection():
    """Demonstrate device detection and capabilities."""
    print("\n" + "="*60)
    print("DEVICE DETECTION AND CAPABILITY ANALYSIS")
    print("="*60)
    
    if not EDGE_AVAILABLE:
        print("Edge modules not available - skipping device detection demo")
        return
    
    # Get supported devices
    devices = get_supported_devices()
    print(f"\nSupported devices detected: {len(devices)}")
    
    for device in devices:
        print(f"\n📱 {device.name} ({device.platform.value})")
        print(f"   CPU: {device.cpu}")
        print(f"   GPU: {device.gpu}")
        print(f"   Memory: {device.memory_gb}GB")
        print(f"   Power: {device.max_power_watts}W")
        print(f"   Backends: {[b.value for b in device.supported_backends]}")
        
        # Test accelerator
        try:
            accelerator = EdgeAccelerator(device.supported_backends)
            capabilities = accelerator.get_device_capabilities(device)
            print(f"   Acceleration: {capabilities.backend.value}")
            print(f"   Max Threads: {capabilities.max_threads}")
            print(f"   Neural Engine: {capabilities.has_neural_engine}")
            print(f"   SIMD Support: {capabilities.simd_support}")
        except Exception as e:
            print(f"   Acceleration test failed: {e}")


def demo_quantization():
    """Demonstrate model quantization."""
    print("\n" + "="*60)
    print("MODEL QUANTIZATION DEMO")
    print("="*60)
    
    if not EDGE_AVAILABLE:
        print("Edge modules not available - skipping quantization demo")
        return
    
    # Create a sample model file for demonstration
    sample_model_path = create_sample_model()
    if not sample_model_path:
        print("Could not create sample model - skipping quantization demo")
        return
    
    quantizer = Quantizer()
    
    # Test different quantization types
    quantization_types = [QuantizationType.FP16, QuantizationType.DYNAMIC]
    
    for qtype in quantization_types:
        print(f"\n🔧 Testing {qtype.value.upper()} Quantization")
        try:
            config = OptimizationConfig(quantization_type=qtype)
            result = quantizer.quantize_model(sample_model_path, qtype, config)
            
            if result.get("status") == "success":
                print(f"   ✅ Success!")
                print(f"   Size reduction: {result.get('size_reduction', 0):.1f}%")
                print(f"   Quantized path: {result.get('quantized_path', 'N/A')}")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Cleanup
    if os.path.exists(sample_model_path):
        os.remove(sample_model_path)


def demo_platform_optimization():
    """Demonstrate platform-specific optimization."""
    print("\n" + "="*60)
    print("PLATFORM-SPECIFIC OPTIMIZATION DEMO")
    print("="*60)
    
    if not EDGE_AVAILABLE:
        print("Edge modules not available - skipping platform optimization demo")
        return
    
    # Test with different devices
    test_devices = [
        EdgeDevice(
            platform=EdgePlatform.IOS,
            name="iPhone 14 Pro",
            cpu="Apple A16 Bionic",
            gpu="Apple GPU",
            memory_gb=6.0,
            supported_backends=[AccelerationBackend.COREML, AccelerationBackend.METAL]
        ),
        EdgeDevice(
            platform=EdgePlatform.ANDROID,
            name="Samsung Galaxy S23",
            cpu="Snapdragon 8 Gen 2", 
            gpu="Adreno 740",
            memory_gb=8.0,
            supported_backends=[AccelerationBackend.NNAPI, AccelerationBackend.OPENCL]
        ),
        EdgeDevice(
            platform=EdgePlatform.LINUX_EMBEDDED,
            name="NVIDIA Jetson Orin",
            cpu="ARM Cortex-A78AE",
            gpu="NVIDIA Ampere",
            memory_gb=8.0,
            supported_backends=[AccelerationBackend.TENSORRT, AccelerationBackend.CUDA]
        )
    ]
    
    for device in test_devices:
        print(f"\n🚀 Optimizing for {device.name}")
        try:
            optimizer = MobileOptimizer(backend=AccelerationBackend.CPU)  # Use CPU for demo
            config = OptimizationConfig(
                max_memory_mb=device.memory_gb * 1024 * 0.8,
                target_latency_ms=50.0,
                battery_optimization=True
            )
            
            # For demo, use a mock model path
            mock_model_path = create_sample_model()
            if mock_model_path:
                result = optimizer.optimize_for_device(mock_model_path, device, config)
                
                if result.get("status") != "failed":
                    print(f"   ✅ Optimization successful")
                    print(f"   Platform: {result.get('platform', 'N/A')}")
                    print(f"   Current backend: {result.get('results', {}).get('platform', {}).get('backend', 'N/A')}")
                    optimizations = result.get('results', {}).get('platform', {}).get('applied_optimizations', [])
                    if optimizations:
                        print(f"   Applied {len(optimizations)} optimizations")
                else:
                    print(f"   ❌ Optimization failed: {result.get('error', 'Unknown error')}")
                
                # Cleanup
                if os.path.exists(mock_model_path):
                    os.remove(mock_model_path)
            else:
                print("   ⚠️  Could not create sample model")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")


def demo_progressive_loading():
    """Demonstrate progressive model loading."""
    print("\n" + "="*60)
    print("PROGRESSIVE MODEL LOADING DEMO")
    print("="*60)
    
    if not EDGE_AVAILABLE:
        print("Edge modules not available - skipping progressive loading demo")
        return
    
    # Create sample model and prepare for progressive loading
    sample_model_path = create_sample_model()
    if not sample_model_path:
        print("Could not create sample model - skipping progressive loading demo")
        return
    
    try:
        loader = ProgressiveLoader()
        temp_dir = "demo_progressive_chunks"
        
        print(f"📦 Preparing progressive loading...")
        prep_result = loader.prepare_progressive_loading(sample_model_path, temp_dir)
        
        if prep_result.get("status") == "success":
            print(f"   ✅ Preparation successful")
            print(f"   Total chunks: {prep_result.get('total_chunks', 0)}")
            print(f"   Total size: {prep_result.get('total_size_mb', 0):.1f}MB")
            print(f"   Estimated load time: {prep_result.get('estimated_load_time', 0):.1f}s")
            
            print(f"\n📥 Testing progressive loading...")
            load_result = loader.load_model_progressive(temp_dir)
            
            if load_result.get("status") == "success":
                stats = load_result.get("statistics", {})
                print(f"   ✅ Loading successful")
                print(f"   Loaded chunks: {stats.get('loaded_chunks', 0)}/{stats.get('total_chunks', 0)}")
                print(f"   Success rate: {load_result.get('load_success_rate', 0)*100:.1f}%")
                print(f"   Total time: {load_result.get('total_time', 0):.1f}s")
            else:
                print(f"   ❌ Loading failed: {load_result.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ Preparation failed: {prep_result.get('error', 'Unknown error')}")
        
        # Cleanup
        if os.path.exists(sample_model_path):
            os.remove(sample_model_path)
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"   ❌ Error: {e}")


def demo_battery_optimization():
    """Demonstrate battery and power optimization."""
    print("\n" + "="*60)
    print("BATTERY AND POWER OPTIMIZATION DEMO")
    print("="*60)
    
    if not EDGE_AVAILABLE:
        print("Edge modules not available - skipping battery optimization demo")
        return
    
    # Create test device and model info
    device = EdgeDevice(
        platform=EdgePlatform.ANDROID,
        name="Google Pixel 7",
        cpu="Google Tensor G2",
        memory_gb=8.0,
        max_power_watts=5.0
    )
    
    model_info = {
        "size_mb": 150,
        "complexity": "medium",
        "ops_per_second": 1e6,
        "parameters": 50_000_000
    }
    
    try:
        battery_optimizer = BatteryOptimizer()
        
        print(f"🔋 Analyzing power consumption...")
        power_analysis = battery_optimizer.analyze_power_consumption(device, model_info)
        
        if "error" not in power_analysis:
            print(f"   ✅ Power analysis successful")
            print(f"   Estimated battery life: {power_analysis.get('battery_life_estimate', 0):.1f}h")
            
            # Thermal analysis
            thermal = power_analysis.get('thermal_analysis', {})
            print(f"   Estimated temperature: {thermal.get('estimated_temperature_celsius', 0):.1f}°C")
            print(f"   Thermal state: {thermal.get('thermal_state', 'unknown')}")
            
            # Optimization opportunities
            optimizations = power_analysis.get('optimization_opportunities', [])
            if optimizations:
                print(f"   📋 Optimization opportunities:")
                for opt in optimizations[:3]:  # Show first 3
                    print(f"      • {opt}")
        else:
            print(f"   ❌ Power analysis failed: {power_analysis.get('error')}")
        
        print(f"\n⚡ Testing battery optimization...")
        sample_model_path = create_sample_model()
        if sample_model_path:
            battery_config = BatteryOptimizationConfig(
                target_battery_life_hours=10.0,
                max_power_consumption_watts=3.0
            )
            
            optimization_result = battery_optimizer.optimize_for_battery(sample_model_path, battery_config)
            
            if optimization_result.get("status") == "success":
                print(f"   ✅ Battery optimization successful")
                improvements = optimization_result.get('expected_improvements', {})
                print(f"   Power reduction: {improvements.get('power_reduction_percent', 0):.1f}%")
                print(f"   Battery life improvement: {improvements.get('battery_life_improvement_percent', 0):.1f}%")
            else:
                print(f"   ❌ Battery optimization failed: {optimization_result.get('error')}")
            
            # Cleanup
            if os.path.exists(sample_model_path):
                os.remove(sample_model_path)
            
    except Exception as e:
        print(f"   ❌ Error: {e}")


def demo_webassembly_deployment():
    """Demonstrate WebAssembly deployment."""
    print("\n" + "="*60)
    print("WEBASSEMBLY DEPLOYMENT DEMO")
    print("="*60)
    
    if not EDGE_AVAILABLE:
        print("Edge modules not available - skipping WebAssembly demo")
        return
    
    # Create sample model for WebAssembly conversion
    sample_model_path = create_sample_model()
    if not sample_model_path:
        print("Could not create sample model - skipping WebAssembly demo")
        return
    
    try:
        deployer = WebAssemblyDeployer()
        config = WASMConfig(
            target_backend="webgl",
            enable_simd=True,
            enable_progressive_loading=True,
            memory_limit_mb=256
        )
        
        temp_dir = "demo_web_deployment"
        print(f"🌐 Converting model to WebAssembly...")
        
        result = deployer.convert_model_to_wasm(sample_model_path, temp_dir, config)
        
        print(f"   ✅ WebAssembly conversion successful")
        print(f"   WASM file: {result.wasm_file}")
        print(f"   JavaScript API: {result.js_api_file}")
        print(f"   HTML demo: {result.html_demo}")
        print(f"   Total size: {result.total_size / 1024 / 1024:.1f}MB")
        print(f"   Estimated load time: {result.estimated_load_time:.1f}s")
        
        # Show progressive chunks
        if result.progressive_chunks:
            print(f"   Progressive chunks: {len(result.progressive_chunks)} files")
        
        # Show enabled features
        features = result.features_enabled
        print(f"   Features enabled:")
        for feature, enabled in features.items():
            if enabled:
                print(f"      ✅ {feature}")
        
        # Cleanup
        if os.path.exists(sample_model_path):
            os.remove(sample_model_path)
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"   ❌ Error: {e}")


def demo_comprehensive_optimization():
    """Demonstrate comprehensive end-to-end optimization."""
    print("\n" + "="*60)
    print("COMPREHENSIVE END-TO-END OPTIMIZATION DEMO")
    print("="*60)
    
    if not EDGE_AVAILABLE:
        print("Edge modules not available - skipping comprehensive demo")
        return
    
    # Create test device and configuration
    device = EdgeDevice(
        platform=EdgePlatform.ANDROID,
        name="Samsung Galaxy S23",
        cpu="Snapdragon 8 Gen 2",
        gpu="Adreno 740",
        memory_gb=8.0,
        max_power_watts=5.0,
        supported_backends=[AccelerationBackend.NNAPI, AccelerationBackend.OPENCL]
    )
    
    config = OptimizationConfig(
        quantization_type=QuantizationType.FP16,
        target_latency_ms=30.0,
        max_memory_mb=512,
        battery_optimization=True,
        enable_progressive_loading=True
    )
    
    print(f"🎯 Target: {device.name}")
    print(f"   Platform: {device.platform.value}")
    print(f"   Memory: {device.memory_gb}GB")
    print(f"   Power limit: {device.max_power_watts}W")
    
    print(f"\n🔧 Optimization configuration:")
    print(f"   Quantization: {config.quantization_type.value if config.quantization_type else 'None'}")
    print(f"   Target latency: {config.target_latency_ms}ms")
    print(f"   Memory limit: {config.max_memory_mb}MB")
    print(f"   Battery optimization: {config.battery_optimization}")
    
    try:
        # Create comprehensive deployer
        deployers = create_edge_deployer(device, config)
        
        print(f"\n📦 Deployer components:")
        for name in deployers.keys():
            print(f"   ✅ {name}")
        
        # Create sample model
        sample_model_path = create_sample_model()
        if sample_model_path:
            print(f"\n🧪 Running optimization pipeline...")
            
            # 1. Quantization
            if 'quantizer' in deployers:
                print(f"   1️⃣ Quantizing model...")
                quant_result = deployers['quantizer'].quantize_model(
                    sample_model_path, config.quantization_type
                )
                if quant_result.get('status') == 'success':
                    print(f"      ✅ Size reduction: {quant_result.get('size_reduction', 0):.1f}%")
            
            # 2. Battery optimization
            if 'battery' in deployers:
                print(f"   2️⃣ Analyzing power consumption...")
                battery_result = deployers['battery'].analyze_power_consumption(
                    device, {"size_mb": 100, "complexity": "medium"}
                )
                if "error" not in battery_result:
                    print(f"      ✅ Battery life: {battery_result.get('battery_life_estimate', 0):.1f}h")
            
            # 3. Benchmark overall performance
            print(f"   3️⃣ Benchmarking performance...")
            benchmark_result = benchmark_edge_deployment(sample_model_path, device, config)
            
            if benchmark_result.get("results"):
                print(f"      ✅ Benchmark completed")
                for component, result in benchmark_result["results"].items():
                    if "error" not in result:
                        print(f"         {component}: ✓")
                    else:
                        print(f"         {component}: ✗ {result.get('error', 'Unknown error')}")
            else:
                print(f"      ❌ Benchmark failed")
            
            # Cleanup
            if os.path.exists(sample_model_path):
                os.remove(sample_model_path)
        else:
            print("   ❌ Could not create sample model")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")


def create_sample_model() -> str:
    """Create a sample model file for testing."""
    try:
        # Create a minimal ONNX model for testing
        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(suffix='.onnx')
        os.close(temp_fd)
        
        # Write minimal ONNX model (simplified)
        minimal_onnx = '''{
            "opset_import": [{"version": 11}],
            "producer_name": "edge_demo",
            "graph": {
                "name": "test_model",
                "input": [{"name": "input", "type": "tensor(float)", "shape": [1, 3, 224, 224]}],
                "output": [{"name": "output", "type": "tensor(float)", "shape": [1, 1000]}],
                "node": [
                    {"op": "Conv", "input": ["input", "conv1.weight"], "output": ["conv1_out"]},
                    {"op": "Relu", "input": ["conv1_out"], "output": ["relu1_out"]},
                    {"op": "GlobalAveragePool", "input": ["relu1_out"], "output": ["gap_out"]},
                    {"op": "Reshape", "input": ["gap_out"], "output": ["output"]}
                ],
                "initializer": [
                    {"name": "conv1.weight", "dims": [64, 3, 7, 7], "data_type": 1}
                ],
                "value_info": []
            }
        }'''
        
        # For demo purposes, just create a simple file
        with open(temp_path, 'w') as f:
            f.write("Demo ONNX model file")
        
        return temp_path
        
    except Exception as e:
        print(f"Warning: Could not create sample model: {e}")
        return None


def main():
    """Main demo function."""
    print("🚀 Edge Deployment Optimization System Demo")
    print("=" * 60)
    print("This demo showcases comprehensive edge optimization capabilities")
    print("for mobile and embedded device deployment.\n")
    
    if not EDGE_AVAILABLE:
        print("❌ Some edge modules are not available.")
        print("Please ensure all dependencies are installed:")
        print("   pip install torch tensorflow onnx onnxruntime")
        print("   pip install psutil numpy")
        return
    
    # Run all demos
    try:
        demo_device_detection()
        demo_quantization()
        demo_platform_optimization()
        demo_progressive_loading()
        demo_battery_optimization()
        demo_webassembly_deployment()
        demo_comprehensive_optimization()
        
        print("\n" + "="*60)
        print("🎉 Demo completed successfully!")
        print("="*60)
        print("\nFor production use:")
        print("1. Install required ML framework dependencies")
        print("2. Use actual model files instead of samples")
        print("3. Test on target devices for performance validation")
        print("4. Configure platform-specific optimizations")
        print("5. Monitor power consumption in real-world scenarios")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
