"""
Battery and Power Optimization Module

Provides comprehensive battery and power optimization for edge devices
including energy-aware inference, power management, and thermal optimization.

Features:
- Power consumption analysis
- Battery life optimization
- Dynamic frequency scaling
- Thermal management
- Energy-efficient scheduling
- Power-aware model selection
- Charging optimization
- Performance vs battery trade-offs
"""

import os
import logging
import json
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import platform
import subprocess
import psutil

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Removed problematic imports for standalone usage
# from . import EdgeDevice, OptimizationConfig

logger = logging.getLogger(__name__)


class PowerMode(Enum):
    """Power management modes."""
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    BATTERY_SAVE = "battery_save"
    ULTRA_SAVE = "ultra_save"


class ThermalState(Enum):
    """Thermal management states."""
    NORMAL = "normal"
    WARM = "warm"
    HOT = "hot"
    THROTTLING = "throttling"
    SHUTDOWN = "shutdown"


@dataclass
class PowerProfile:
    """Power consumption profile."""
    mode: PowerMode
    max_power_watts: float
    typical_power_watts: float
    idle_power_watts: float
    thermal_limit_celsius: float
    battery_optimization: bool
    performance_scaling: float


@dataclass
class PowerMetrics:
    """Power consumption metrics."""
    current_power_watts: float
    average_power_watts: float
    energy_consumed_wh: float
    battery_remaining_percent: float
    estimated_battery_life_hours: float
    thermal_state: ThermalState
    cpu_frequency_mhz: float
    gpu_frequency_mhz: Optional[float] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class BatteryOptimizationConfig:
    """Configuration for battery optimization."""
    target_battery_life_hours: float = 8.0
    max_power_consumption_watts: float = 5.0
    thermal_limit_celsius: float = 80.0
    enable_dynamic_scaling: bool = True
    enable_charging_optimization: bool = True
    performance_penalty_tolerance: float = 0.1  # 10% performance penalty acceptable
    power_mode: PowerMode = PowerMode.BALANCED


class BatteryOptimizer:
    """Battery and power optimization engine."""
    
    def __init__(self):
        """Initialize battery optimizer."""
        self.power_monitoring_active = False
        self.power_history = []
        self.battery_profile = None
        self.thermal_monitor = ThermalMonitor()
        self.power_monitor_thread = None
        
        # Platform detection
        self.platform_info = self._detect_platform()
        
        # Initialize battery profile
        self._initialize_battery_profile()
        
        logger.info(f"Battery optimizer initialized for {self.platform_info['platform']}")
    
    def _detect_platform(self) -> Dict[str, Any]:
        """Detect platform for power management."""
        system_info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "has_battery": False,
            "battery_level": None,
            "power_sources": [],
            "thermal_zones": []
        }
        
        try:
            if platform.system() == "Linux":
                self._detect_linux_power_info(system_info)
            elif platform.system() == "Darwin":  # macOS
                self._detect_macos_power_info(system_info)
            elif platform.system() == "Windows":
                self._detect_windows_power_info(system_info)
        except Exception as e:
            logger.debug(f"Power detection failed: {e}")
        
        return system_info
    
    def _detect_linux_power_info(self, system_info: Dict[str, Any]):
        """Detect power information on Linux."""
        try:
            # Check for battery
            battery_paths = [
                "/sys/class/power_supply/BAT0/capacity",
                "/sys/class/power_supply/BAT1/capacity"
            ]
            
            for path in battery_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        battery_level = int(f.read().strip())
                        system_info["has_battery"] = True
                        system_info["battery_level"] = battery_level
                        break
            
            # Check thermal zones
            thermal_zone_paths = ["/sys/class/thermal/thermal_zone{}".format(i) 
                                for i in range(10)]
            
            for path in thermal_zone_paths:
                if os.path.exists(path):
                    try:
                        temp_path = os.path.join(path, "temp")
                        if os.path.exists(temp_path):
                            with open(temp_path, 'r') as f:
                                temp_millidegrees = int(f.read().strip())
                                temp_celsius = temp_millidegrees / 1000.0
                                system_info["thermal_zones"].append(temp_celsius)
                    except:
                        continue
                        
        except Exception as e:
            logger.debug(f"Linux power detection failed: {e}")
    
    def _detect_macos_power_info(self, system_info: Dict[str, Any]):
        """Detect power information on macOS."""
        try:
            # Use system_profiler to get battery info
            result = subprocess.run(
                ["system_profiler", "SPPowerDataType", "-json"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                import plistlib
                data = plistlib.loads(result.stdout.encode())
                power_data = data.get("SPPowerDataType", [])
                
                if power_data:
                    battery_info = power_data[0]
                    system_info["has_battery"] = battery_info.get("BatteryInstalled", False)
                    if system_info["has_battery"]:
                        system_info["battery_level"] = battery_info.get("CurrentCapacity", 0)
        except Exception as e:
            logger.debug(f"macOS power detection failed: {e}")
    
    def _detect_windows_power_info(self, system_info: Dict[str, Any]):
        """Detect power information on Windows."""
        try:
            # Use WMIC to get battery info
            result = subprocess.run([
                "wmic", "path", "Win32_PerfRawData_PowerMeter_PowerMeter",
                "get", "CurrentPowerConsumptionWatt"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                if lines and lines[0].strip():
                    system_info["current_power_watts"] = float(lines[0].strip())
        except Exception as e:
            logger.debug(f"Windows power detection failed: {e}")
    
    def _initialize_battery_profile(self):
        """Initialize battery profile based on platform."""
        power_profiles = {
            PowerMode.PERFORMANCE: PowerProfile(
                mode=PowerMode.PERFORMANCE,
                max_power_watts=10.0,
                typical_power_watts=8.0,
                idle_power_watts=2.0,
                thermal_limit_celsius=85.0,
                battery_optimization=False,
                performance_scaling=1.0
            ),
            PowerMode.BALANCED: PowerProfile(
                mode=PowerMode.BALANCED,
                max_power_watts=6.0,
                typical_power_watts=4.0,
                idle_power_watts=1.5,
                thermal_limit_celsius=75.0,
                battery_optimization=True,
                performance_scaling=0.8
            ),
            PowerMode.BATTERY_SAVE: PowerProfile(
                mode=PowerMode.BATTERY_SAVE,
                max_power_watts=3.0,
                typical_power_watts=2.0,
                idle_power_watts=1.0,
                thermal_limit_celsius=70.0,
                battery_optimization=True,
                performance_scaling=0.6
            ),
            PowerMode.ULTRA_SAVE: PowerProfile(
                mode=PowerMode.ULTRA_SAVE,
                max_power_watts=1.5,
                typical_power_watts=1.0,
                idle_power_watts=0.5,
                thermal_limit_celsius=65.0,
                battery_optimization=True,
                performance_scaling=0.4
            )
        }
        
        self.battery_profile = power_profiles
    
    def analyze_power_consumption(self, device: Dict[str, Any], 
                                model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze power consumption for device and model.
        
        Args:
            device: Edge device
            model_info: Model information including size, complexity
            
        Returns:
            Power analysis results
        """
        logger.info(f"Analyzing power consumption for {device.get('name', 'Unknown Device')}")
        
        try:
            # Get device power characteristics
            device_power = self._analyze_device_power_characteristics(device)
            
            # Analyze model power requirements
            model_power = self._analyze_model_power_requirements(model_info)
            
            # Calculate overall power profile
            power_profile = self._calculate_power_profile(device_power, model_power)
            
            # Estimate battery life
            battery_life = self._estimate_battery_life(power_profile, device)
            
            # Identify optimization opportunities
            optimizations = self._identify_power_optimizations(power_profile, device, model_info)
            
            analysis = {
                "device": device.get("name", "Unknown Device"),
                "model_size_mb": model_info.get("size_mb", 0),
                "device_power": device_power,
                "model_power": model_power,
                "combined_power": power_profile,
                "battery_life_estimate": battery_life,
                "optimization_opportunities": optimizations,
                "thermal_analysis": self._analyze_thermal_requirements(device, power_profile)
            }
            
            logger.info(f"Power analysis completed - estimated battery life: {battery_life:.1f}h")
            
        except Exception as e:
            logger.error(f"Power analysis failed: {e}")
            analysis = {
                "device": device.get("name", "Unknown Device"),
                "error": str(e),
                "status": "failed"
            }
        
        return analysis
    
    def _analyze_device_power_characteristics(self, device: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze device power characteristics."""
        base_power = device.get("max_power_watts", 5.0)
        
        # Adjust based on device specifications
        power_factors = {
            "memory_factor": min(device.get("memory_gb", 4.0) / 4.0, 2.0),  # More memory = more power
            "storage_factor": min(device.get("storage_gb", 64.0) / 64.0, 1.5),  # More storage = more power
            "platform_factor": 1.0
        }
        
        # Platform-specific adjustments
        platform = device.get("platform", "").lower()
        if platform == "ios":
            power_factors["platform_factor"] = 0.8  # iOS devices are power efficient
        elif platform == "android":
            power_factors["platform_factor"] = 1.0
        elif platform == "linux_embedded":
            power_factors["platform_factor"] = 1.2  # Embedded systems often less efficient
        
        return {
            "base_power_watts": base_power,
            "power_factors": power_factors,
            "max_power_watts": base_power * power_factors["memory_factor"] * 
                              power_factors["storage_factor"] * power_factors["platform_factor"],
            "idle_power_watts": base_power * 0.1,  # Assume 10% of max power at idle
            "efficiency_factor": 0.85  # Assume 85% power efficiency
        }
    
    def _analyze_model_power_requirements(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze power requirements for model."""
        model_size_mb = model_info.get("size_mb", 0)
        complexity = model_info.get("complexity", "medium")
        operations_per_second = model_info.get("ops_per_second", 1e6)
        
        # Power per MB of model size
        base_power_per_mb = 0.05  # Watts per MB
        
        # Power per complexity factor
        complexity_factors = {
            "low": 0.5,
            "medium": 1.0,
            "high": 2.0
        }
        
        # Power per operations (simplified model)
        power_per_op = 1e-9  # Watts per operation per second
        
        model_power = {
            "size_power_watts": model_size_mb * base_power_per_mb,
            "complexity_power_watts": operations_per_second * power_per_op * 
                                    complexity_factors.get(complexity, 1.0),
            "total_power_watts": 0.0,
            "power_factors": {
                "size_mb": model_size_mb,
                "complexity": complexity,
                "operations_per_second": operations_per_second
            }
        }
        
        model_power["total_power_watts"] = model_power["size_power_watts"] + model_power["complexity_power_watts"]
        
        return model_power
    
    def _calculate_power_profile(self, device_power: Dict[str, Any], 
                               model_power: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate combined power profile."""
        total_power = device_power["base_power_watts"] + model_power["total_power_watts"]
        max_power = device_power["max_power_watts"] + model_power["total_power_watts"]
        idle_power = device_power["idle_power_watts"]
        
        return {
            "total_power_watts": total_power,
            "max_power_watts": max_power,
            "idle_power_watts": idle_power,
            "model_power_watts": model_power["total_power_watts"],
            "device_power_watts": device_power["base_power_watts"],
            "efficiency": device_power["efficiency_factor"],
            "power_ratio": total_power / max_power if max_power > 0 else 0
        }
    
    def _estimate_battery_life(self, power_profile: Dict[str, Any], device: Dict[str, Any]) -> float:
        """Estimate battery life based on power profile."""
        # Estimate battery capacity (simplified)
        # Assume 100Wh per GB of storage for rough estimation
        battery_capacity_wh = device.get("storage_gb", 64) * 100
        
        # Use typical power consumption for estimation
        power_consumption_watts = power_profile["total_power_watts"]
        
        if power_consumption_watts <= 0:
            return float('inf')
        
        battery_life_hours = battery_capacity_wh / power_consumption_watts
        
        return battery_life_hours
    
    def _identify_power_optimizations(self, power_profile: Dict[str, Any],
                                    device: Dict[str, Any], model_info: Dict[str, Any]) -> List[str]:
        """Identify power optimization opportunities."""
        optimizations = []
        
        # Power consumption analysis
        power_ratio = power_profile.get("power_ratio", 0)
        if power_ratio > 0.8:
            optimizations.append("High power consumption detected - consider model optimization")
        
        # Model size optimization
        model_size_mb = model_info.get("size_mb", 0)
        if model_size_mb > 100:
            optimizations.append("Large model detected - consider model pruning or quantization")
        
        # Efficiency improvements
        efficiency = power_profile.get("efficiency", 1.0)
        if efficiency < 0.8:
            optimizations.append("Low power efficiency - consider hardware acceleration")
        
        # Battery optimization suggestions
        if power_profile["total_power_watts"] > device.get("max_power_watts", 5.0) * 0.7:
            optimizations.append("Approaching power limit - enable power management features")
        
        # Platform-specific optimizations
        platform = device.get("platform", "").lower()
        if platform == "android":
            optimizations.append("Enable Android Doze mode for background processing")
        elif platform == "ios":
            optimizations.append("Use iOS background app refresh settings for battery optimization")
        elif platform == "linux_embedded":
            optimizations.append("Configure CPU frequency scaling for embedded systems")
        
        return optimizations
    
    def _analyze_thermal_requirements(self, device: Dict[str, Any], 
                                    power_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze thermal requirements and heat generation."""
        power_watts = power_profile["total_power_watts"]
        
        # Estimate heat generation (1W ≈ 3.41 BTU/h)
        heat_generation_btu_h = power_watts * 3.41
        
        # Estimate temperature rise
        # Simplified thermal model: ΔT = Power / Thermal Resistance
        thermal_resistance = 2.0  # °C/W (assumed)
        temp_rise_celsius = power_watts * thermal_resistance
        
        # Assume ambient temperature
        ambient_temp = 25.0  # °C
        estimated_temp = ambient_temp + temp_rise_celsius
        
        # Determine thermal state
        if estimated_temp < 50:
            thermal_state = ThermalState.NORMAL
        elif estimated_temp < 70:
            thermal_state = ThermalState.WARM
        elif estimated_temp < 80:
            thermal_state = ThermalState.HOT
        else:
            thermal_state = ThermalState.THROTTLING
        
        return {
            "heat_generation_btu_h": heat_generation_btu_h,
            "estimated_temperature_celsius": estimated_temp,
            "thermal_state": thermal_state.value,
            "thermal_limit_celsius": device.get("constraints", {}).get("thermal_limit", 80.0),
            "cooling_recommendations": self._get_cooling_recommendations(thermal_state, estimated_temp)
        }
    
    def _get_cooling_recommendations(self, thermal_state: ThermalState, 
                                   temp_celsius: float) -> List[str]:
        """Get cooling recommendations based on thermal state."""
        recommendations = []
        
        if thermal_state == ThermalState.WARM:
            recommendations.append("Monitor temperature - consider reducing workload")
        elif thermal_state == ThermalState.HOT:
            recommendations.extend([
                "Temperature approaching limit - reduce inference frequency",
                "Consider external cooling or improved ventilation"
            ])
        elif thermal_state in [ThermalState.THROTTLING, ThermalState.SHUTDOWN]:
            recommendations.extend([
                "Critical temperature - implement emergency throttling",
                "Consider reducing model precision or batch size",
                "Add thermal management to prevent damage"
            ])
        
        if temp_celsius > 70:
            recommendations.append("Enable dynamic frequency scaling")
        
        return recommendations
    
    def optimize_for_battery(self, model_path: str, 
                           config: BatteryOptimizationConfig = None) -> Dict[str, Any]:
        """
        Optimize model deployment for battery life.
        
        Args:
            model_path: Path to model
            config: Battery optimization configuration
            
        Returns:
            Optimization results
        """
        logger.info("Optimizing for battery life")
        
        if config is None:
            config = BatteryOptimizationConfig()
        
        try:
            # Analyze current setup
            current_analysis = self._analyze_current_power_state()
            
            # Create battery-optimized configuration
            optimized_config = self._create_battery_optimized_config(config)
            
            # Generate optimization recommendations
            recommendations = self._generate_battery_recommendations(optimized_config)
            
            # Create monitoring plan
            monitoring_plan = self._create_power_monitoring_plan(config)
            
            results = {
                "status": "success",
                "current_analysis": current_analysis,
                "optimized_config": asdict(optimized_config),
                "recommendations": recommendations,
                "monitoring_plan": monitoring_plan,
                "expected_improvements": self._calculate_expected_improvements(current_analysis, optimized_config)
            }
            
            logger.info("Battery optimization completed")
            
        except Exception as e:
            logger.error(f"Battery optimization failed: {e}")
            results = {
                "status": "failed",
                "error": str(e)
            }
        
        return results
    
    def _analyze_current_power_state(self) -> Dict[str, Any]:
        """Analyze current power consumption state."""
        return {
            "battery_level": self.platform_info.get("battery_level"),
            "platform": self.platform_info["platform"],
            "has_battery": self.platform_info["has_battery"],
            "current_power_estimate": self._estimate_current_power(),
            "thermal_state": self.thermal_monitor.get_current_state(),
            "power_sources": self.platform_info.get("power_sources", [])
        }
    
    def _estimate_current_power(self) -> float:
        """Estimate current power consumption."""
        try:
            # Get CPU usage and frequency
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            
            # Get memory usage
            memory = psutil.virtual_memory()
            
            # Simple power estimation model
            base_power = 5.0  # Watts
            cpu_power = (cpu_percent / 100.0) * 3.0  # CPU power component
            memory_power = (memory.percent / 100.0) * 1.0  # Memory power component
            
            if cpu_freq:
                freq_factor = cpu_freq.current / cpu_freq.max if cpu_freq.max > 0 else 1.0
                cpu_power *= freq_factor
            
            return base_power + cpu_power + memory_power
            
        except Exception as e:
            logger.debug(f"Power estimation failed: {e}")
            return 5.0  # Default estimate
    
    def _create_battery_optimized_config(self, config: BatteryOptimizationConfig) -> BatteryOptimizationConfig:
        """Create battery-optimized configuration."""
        optimized = BatteryOptimizationConfig(
            target_battery_life_hours=config.target_battery_life_hours,
            max_power_consumption_watts=config.max_power_consumption_watts * 0.7,  # Reduce by 30%
            thermal_limit_celsius=config.thermal_limit_celsius,
            enable_dynamic_scaling=True,
            enable_charging_optimization=True,
            performance_penalty_tolerance=config.performance_penalty_tolerance,
            power_mode=PowerMode.BATTERY_SAVE
        )
        
        return optimized
    
    def _generate_battery_recommendations(self, config: BatteryOptimizationConfig) -> List[str]:
        """Generate battery optimization recommendations."""
        recommendations = []
        
        # Power mode recommendations
        if config.power_mode == PowerMode.BATTERY_SAVE:
            recommendations.extend([
                "Enable battery save mode",
                "Reduce screen brightness if applicable",
                "Disable unnecessary background processes"
            ])
        
        # Model optimization recommendations
        recommendations.extend([
            "Consider model quantization to reduce memory usage",
            "Implement dynamic batching for variable workloads",
            "Use progressive loading to spread power consumption",
            "Enable hardware acceleration when available"
        ])
        
        # System-level recommendations
        recommendations.extend([
            "Configure CPU frequency scaling",
            "Enable power-aware scheduling",
            "Set thermal management thresholds",
            "Configure charging optimization"
        ])
        
        # Monitoring recommendations
        recommendations.extend([
            "Monitor power consumption continuously",
            "Track battery health over time",
            "Log thermal events",
            "Set alerts for high power consumption"
        ])
        
        return recommendations
    
    def _create_power_monitoring_plan(self, config: BatteryOptimizationConfig) -> Dict[str, Any]:
        """Create power monitoring plan."""
        return {
            "monitoring_interval_seconds": 5,
            "metrics_to_track": [
                "power_consumption_watts",
                "battery_level_percent",
                "cpu_frequency_mhz",
                "temperature_celsius",
                "memory_usage_percent"
            ],
            "alert_thresholds": {
                "max_power_watts": config.max_power_consumption_watts,
                "max_temperature_celsius": config.thermal_limit_celsius,
                "min_battery_percent": 20
            },
            "data_retention_hours": 24,
            "reporting_interval_minutes": 60
        }
    
    def _calculate_expected_improvements(self, current_analysis: Dict[str, Any],
                                       optimized_config: BatteryOptimizationConfig) -> Dict[str, Any]:
        """Calculate expected improvements from optimization."""
        current_power = current_analysis.get("current_power_estimate", 5.0)
        optimized_power = current_power * 0.7  # Assume 30% reduction
        
        # Estimate battery life improvement
        current_battery_life = 10.0  # Assume 10 hours current
        optimized_battery_life = current_battery_life * (current_power / optimized_power)
        
        return {
            "power_reduction_percent": (1 - optimized_power / current_power) * 100,
            "battery_life_improvement_percent": (optimized_battery_life / current_battery_life - 1) * 100,
            "estimated_power_savings_watts": current_power - optimized_power,
            "daily_energy_savings_wh": (current_power - optimized_power) * 24
        }
    
    def start_power_monitoring(self, interval_seconds: int = 5) -> Dict[str, Any]:
        """Start continuous power monitoring."""
        logger.info(f"Starting power monitoring (interval: {interval_seconds}s)")
        
        if self.power_monitoring_active:
            return {"status": "already_running"}
        
        self.power_monitoring_active = True
        self.power_history = []
        
        def monitor_loop():
            while self.power_monitoring_active:
                try:
                    metrics = self._collect_power_metrics()
                    self.power_history.append(metrics)
                    
                    # Keep only recent data (last hour)
                    current_time = time.time()
                    self.power_history = [
                        m for m in self.power_history 
                        if current_time - m.timestamp < 3600
                    ]
                    
                    # Check for alerts
                    self._check_power_alerts(metrics)
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Power monitoring error: {e}")
                    time.sleep(interval_seconds)
        
        self.power_monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.power_monitor_thread.start()
        
        return {
            "status": "started",
            "interval_seconds": interval_seconds,
            "thread_id": self.power_monitor_thread.ident
        }
    
    def stop_power_monitoring(self) -> Dict[str, Any]:
        """Stop power monitoring."""
        logger.info("Stopping power monitoring")
        
        self.power_monitoring_active = False
        
        if self.power_monitor_thread:
            self.power_monitor_thread.join(timeout=5)
        
        return {
            "status": "stopped",
            "samples_collected": len(self.power_history)
        }
    
    def _collect_power_metrics(self) -> PowerMetrics:
        """Collect current power metrics."""
        try:
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            cpu_freq = psutil.cpu_freq()
            
            # Estimate power consumption
            current_power = self._estimate_current_power()
            
            # Get thermal state
            temp_celsius = self._get_system_temperature()
            thermal_state = self._determine_thermal_state(temp_celsius)
            
            # Get battery information
            battery_level = self.platform_info.get("battery_level")
            
            # Estimate battery life
            if battery_level is not None and current_power > 0:
                # Simple linear estimation (in practice, would use more sophisticated model)
                battery_life_hours = (battery_level / 100.0) * 8.0 / max(current_power / 5.0, 0.1)
            else:
                battery_life_hours = 0
            
            metrics = PowerMetrics(
                current_power_watts=current_power,
                average_power_watts=current_power,  # Simplified
                energy_consumed_wh=0.0,  # Would need cumulative tracking
                battery_remaining_percent=battery_level or 0.0,
                estimated_battery_life_hours=battery_life_hours,
                thermal_state=thermal_state,
                cpu_frequency_mhz=cpu_freq.current if cpu_freq else 0.0
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Power metrics collection failed: {e}")
            return PowerMetrics(
                current_power_watts=0.0,
                average_power_watts=0.0,
                energy_consumed_wh=0.0,
                battery_remaining_percent=0.0,
                estimated_battery_life_hours=0.0,
                thermal_state=ThermalState.NORMAL,
                cpu_frequency_mhz=0.0
            )
    
    def _get_system_temperature(self) -> float:
        """Get current system temperature."""
        try:
            # Use psutil if available
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get average temperature across all sensors
                    all_temps = []
                    for name, entries in temps.items():
                        for entry in entries:
                            if hasattr(entry, 'current'):
                                all_temps.append(entry.current)
                    return sum(all_temps) / len(all_temps) if all_temps else 25.0
        except:
            pass
        
        # Fallback to platform-specific methods
        if self.platform_info["platform"] == "Linux":
            try:
                with open("/sys/class/thermal/thermal_zone0/temp", 'r') as f:
                    temp_millidegrees = int(f.read().strip())
                    return temp_millidegrees / 1000.0
            except:
                pass
        
        return 25.0  # Default ambient temperature
    
    def _determine_thermal_state(self, temp_celsius: float) -> ThermalState:
        """Determine thermal state based on temperature."""
        if temp_celsius < 50:
            return ThermalState.NORMAL
        elif temp_celsius < 65:
            return ThermalState.WARM
        elif temp_celsius < 75:
            return ThermalState.HOT
        elif temp_celsius < 85:
            return ThermalState.THROTTLING
        else:
            return ThermalState.SHUTDOWN
    
    def _check_power_alerts(self, metrics: PowerMetrics):
        """Check for power-related alerts."""
        alerts = []
        
        # Power consumption alerts
        if metrics.current_power_watts > 10.0:
            alerts.append(f"High power consumption: {metrics.current_power_watts:.1f}W")
        
        # Thermal alerts
        if metrics.thermal_state in [ThermalState.THROTTLING, ThermalState.SHUTDOWN]:
            alerts.append(f"Thermal throttling active: {metrics.thermal_state.value}")
        
        # Battery alerts
        if metrics.battery_remaining_percent < 20:
            alerts.append(f"Low battery: {metrics.battery_remaining_percent:.1f}%")
        
        if alerts:
            logger.warning(f"Power alerts: {', '.join(alerts)}")
    
    def get_power_report(self) -> Dict[str, Any]:
        """Generate comprehensive power report."""
        if not self.power_history:
            return {"error": "No power data available"}
        
        # Calculate statistics
        power_values = [m.current_power_watts for m in self.power_history]
        battery_values = [m.battery_remaining_percent for m in self.power_history if m.battery_remaining_percent > 0]
        
        report = {
            "duration_minutes": (self.power_history[-1].timestamp - self.power_history[0].timestamp) / 60,
            "samples_count": len(self.power_history),
            "power_statistics": {
                "mean_watts": float(np.mean(power_values)),
                "max_watts": float(np.max(power_values)),
                "min_watts": float(np.min(power_values)),
                "std_watts": float(np.std(power_values))
            },
            "battery_statistics": None,
            "thermal_analysis": self._analyze_thermal_history(),
            "power_recommendations": self._generate_power_recommendations(),
            "data_points": [asdict(m) for m in self.power_history[-100:]]  # Last 100 samples
        }
        
        if battery_values:
            report["battery_statistics"] = {
                "mean_level_percent": float(np.mean(battery_values)),
                "max_level_percent": float(np.max(battery_values)),
                "min_level_percent": float(np.min(battery_values))
            }
        
        return report
    
    def _analyze_thermal_history(self) -> Dict[str, Any]:
        """Analyze thermal history."""
        temps = [self._get_system_temperature()]  # Current temp only for now
        
        return {
            "current_temperature_celsius": temps[0],
            "thermal_events": [],
            "throttling_incidents": 0,
            "cooling_effectiveness": 1.0
        }
    
    def _generate_power_recommendations(self) -> List[str]:
        """Generate power optimization recommendations."""
        if not self.power_history:
            return ["No power data available for analysis"]
        
        recent_metrics = self.power_history[-10:]  # Last 10 samples
        avg_power = np.mean([m.current_power_watts for m in recent_metrics])
        
        recommendations = []
        
        if avg_power > 8.0:
            recommendations.append("High average power consumption - consider reducing model complexity")
        
        if avg_power < 2.0:
            recommendations.append("Low power consumption - good for battery life")
        
        # Check for thermal issues
        thermal_issues = sum(1 for m in recent_metrics 
                           if m.thermal_state in [ThermalState.HOT, ThermalState.THROTTLING])
        
        if thermal_issues > 0:
            recommendations.append("Thermal issues detected - implement cooling solutions")
        
        return recommendations


class ThermalMonitor:
    """Monitor thermal state of the system."""
    
    def __init__(self):
        self.thermal_history = []
    
    def get_current_state(self) -> ThermalState:
        """Get current thermal state."""
        try:
            temp = self._get_temperature()
            return self._state_from_temperature(temp)
        except:
            return ThermalState.NORMAL
    
    def _get_temperature(self) -> float:
        """Get current temperature."""
        try:
            if platform.system() == "Linux":
                with open("/sys/class/thermal/thermal_zone0/temp", 'r') as f:
                    return int(f.read().strip()) / 1000.0
        except:
            pass
        return 25.0  # Default
    
    def _state_from_temperature(self, temp_celsius: float) -> ThermalState:
        """Convert temperature to thermal state."""
        if temp_celsius < 50:
            return ThermalState.NORMAL
        elif temp_celsius < 65:
            return ThermalState.WARM
        elif temp_celsius < 75:
            return ThermalState.HOT
        elif temp_celsius < 85:
            return ThermalState.THROTTLING
        else:
            return ThermalState.SHUTDOWN
