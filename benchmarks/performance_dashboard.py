#!/usr/bin/env python3
"""
Performance Monitoring Dashboard
================================

Real-time monitoring dashboard for performance metrics including tok/s, TTFT, memory usage,
energy consumption, and spike rates. Supports regression detection and alerting.

Usage:
    python performance_dashboard.py
    python performance_dashboard.py --port 8080 --baseline results/baseline.json
    python performance_dashboard.py --monitor-only --config monitoring.yaml
"""

import asyncio
import json
import time
import statistics
import numpy as np
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import argparse
import logging
import webbrowser
from collections import deque

try:
    import psutil
    import GPUtil
    HAS_PSUTIL = True
    HAS_GPUTIL = True
except ImportError:
    HAS_PSUTIL = False
    HAS_GPUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from aiohttp import web, web_runner
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring."""
    # Dashboard settings
    port: int = 8080
    update_interval_ms: int = 1000
    history_size: int = 1000
    
    # Alert thresholds
    alert_tok_s_threshold: float = 30.0
    alert_ttft_threshold_ms: float = 5.0
    alert_spike_rate_threshold: float = 15.0
    alert_memory_threshold_percent: float = 90.0
    alert_cpu_threshold_percent: float = 90.0
    
    # Monitoring targets
    target_tok_s: float = 30.0
    target_ttft_ms: float = 5.0
    target_spike_rate_range: tuple = (5.0, 15.0)
    
    # Baseline comparison
    baseline_file: Optional[str] = None
    
    # Storage
    data_dir: str = "monitoring_data"
    log_file: str = "monitoring.log"
    
    # Features
    enable_alerts: bool = True
    enable_regression_detection: bool = True
    enable_export: bool = True


class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.setup_logging()
        
        # Data storage
        self.metrics_history = deque(maxlen=config.history_size)
        self.system_metrics = deque(maxlen=config.history_size)
        self.alerts = []
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        
        # Baseline data
        self.baseline = self._load_baseline()
        
        # Alert tracking
        self.alert_cooldown = {}
        
        # Setup data directory
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler()
            ]
        )
        
    def _load_baseline(self) -> Dict[str, Any]:
        """Load baseline metrics for comparison."""
        if not self.config.baseline_file:
            return {}
            
        try:
            with open(self.config.baseline_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load baseline: {e}")
            return {}
            
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logging.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
        logging.info("Performance monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                timestamp = datetime.now()
                
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                
                # Collect performance metrics (simulated for demo)
                perf_metrics = self._collect_performance_metrics()
                
                # Combine metrics
                combined_metrics = {
                    'timestamp': timestamp.isoformat(),
                    'uptime_seconds': time.time() - self.start_time,
                    'system': system_metrics,
                    'performance': perf_metrics
                }
                
                self.metrics_history.append(combined_metrics)
                self.system_metrics.append(system_metrics)
                
                # Check for alerts
                if self.config.enable_alerts:
                    self._check_alerts(combined_metrics)
                    
                # Check for regression
                if self.config.enable_regression_detection:
                    self._check_regression(combined_metrics)
                    
                time.sleep(self.config.update_interval_ms / 1000.0)
                
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(1.0)
                
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system resource metrics."""
        metrics = {}
        
        if HAS_PSUTIL:
            # CPU metrics
            metrics['cpu_percent'] = psutil.cpu_percent()
            metrics['cpu_count'] = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_used_gb'] = memory.used / (1024**3)
            metrics['memory_available_gb'] = memory.available / (1024**3)
            metrics['memory_total_gb'] = memory.total / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk_percent'] = disk.percent
            metrics['disk_free_gb'] = disk.free / (1024**3)
            
            # Process metrics
            process = psutil.Process()
            metrics['process_cpu_percent'] = process.cpu_percent()
            metrics['process_memory_mb'] = process.memory_info().rss / 1024 / 1024
            
        # GPU metrics
        if HAS_GPUTIL and HAS_TORCH and torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics['gpu_util_percent'] = gpu.load * 100
                    metrics['gpu_memory_percent'] = (gpu.memoryUsed / gpu.memoryTotal) * 100
                    metrics['gpu_temperature'] = gpu.temperature
            except Exception:
                metrics['gpu_util_percent'] = 0
                metrics['gpu_memory_percent'] = 0
                metrics['gpu_temperature'] = 0
                
        return metrics
        
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics (simulated)."""
        # This would typically collect real metrics from your system
        # For demo purposes, we'll simulate realistic values
        
        current_time = time.time()
        uptime = current_time - self.start_time if self.start_time else 0
        
        # Simulate token throughput with some variation
        base_tok_s = 35.0  # Base throughput
        noise = np.random.normal(0, 2.0)  # Small variation
        tok_s = max(0, base_tok_s + noise)
        
        # Simulate TTFT with correlation to tok/s
        ttft_ms = max(1.0, 200.0 / tok_s + np.random.normal(0, 0.5))
        
        # Simulate spike rate
        spike_rate = np.random.uniform(3.0, 12.0)  # Within target range
        
        # Simulate retrieval latency
        retrieval_ms = np.random.lognormal(np.log(15), 0.3)
        
        # Energy metrics (simulated)
        energy_watts = np.random.normal(250, 20) if HAS_TORCH and torch.cuda.is_available() else np.random.normal(50, 5)
        
        return {
            'tokens_per_second': tok_s,
            'ttft_ms': ttft_ms,
            'spike_rate_percent': spike_rate,
            'retrieval_latency_ms': retrieval_ms,
            'energy_consumption_watts': energy_watts,
            'inference_count': int(uptime * tok_s / 50),  # Simulated count
            'memory_efficiency': np.random.uniform(0.7, 0.95)
        }
        
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for alert conditions."""
        alerts = []
        
        performance = metrics.get('performance', {})
        system = metrics.get('system', {})
        
        # Token throughput alert
        tok_s = performance.get('tokens_per_second', 0)
        if tok_s < self.config.alert_tok_s_threshold:
            alerts.append({
                'type': 'token_throughput',
                'message': f'Token throughput {tok_s:.1f} tok/s below threshold {self.config.alert_tok_s_threshold}',
                'severity': 'warning' if tok_s > self.config.alert_tok_s_threshold * 0.8 else 'critical'
            })
            
        # TTFT alert
        ttft = performance.get('ttft_ms', 0)
        if ttft > self.config.alert_ttft_threshold_ms:
            alerts.append({
                'type': 'ttft',
                'message': f'TTFT {ttft:.2f}ms exceeds threshold {self.config.alert_ttft_threshold_ms}ms',
                'severity': 'warning' if ttft < self.config.alert_ttft_threshold_ms * 1.5 else 'critical'
            })
            
        # Spike rate alert
        spike_rate = performance.get('spike_rate_percent', 0)
        if spike_rate > self.config.alert_spike_rate_threshold:
            alerts.append({
                'type': 'spike_rate',
                'message': f'Spike rate {spike_rate:.1f}% exceeds threshold {self.config.alert_spike_rate_threshold}%',
                'severity': 'warning'
            })
            
        # Memory alert
        memory_percent = system.get('memory_percent', 0)
        if memory_percent > self.config.alert_memory_threshold_percent:
            alerts.append({
                'type': 'memory',
                'message': f'Memory usage {memory_percent:.1f}% exceeds threshold {self.config.alert_memory_threshold_percent}%',
                'severity': 'critical'
            })
            
        # CPU alert
        cpu_percent = system.get('cpu_percent', 0)
        if cpu_percent > self.config.alert_cpu_threshold_percent:
            alerts.append({
                'type': 'cpu',
                'message': f'CPU usage {cpu_percent:.1f}% exceeds threshold {self.config.alert_cpu_threshold_percent}%',
                'severity': 'warning' if cpu_percent < 95 else 'critical'
            })
            
        # Add alerts with cooldown
        for alert in alerts:
            alert_type = alert['type']
            if alert_type not in self.alert_cooldown or \
               time.time() - self.alert_cooldown[alert_type] > 300:  # 5 minute cooldown
                
                self.alerts.append({
                    'timestamp': metrics['timestamp'],
                    'type': alert['type'],
                    'message': alert['message'],
                    'severity': alert['severity']
                })
                
                self.alert_cooldown[alert_type] = time.time()
                
                logging.warning(f"Alert: {alert['message']}")
                
    def _check_regression(self, current_metrics: Dict[str, Any]):
        """Check for performance regression against baseline."""
        if not self.baseline or 'benchmarks' not in self.baseline:
            return
            
        performance = current_metrics.get('performance', {})
        
        # Check token throughput regression
        if 'token_throughput' in self.baseline['benchmarks']:
            baseline_tok_s = self.baseline['benchmarks']['token_throughput'].get('tokens_per_second', 0)
            current_tok_s = performance.get('tokens_per_second', 0)
            
            if baseline_tok_s > 0:
                regression_percent = ((current_tok_s - baseline_tok_s) / baseline_tok_s) * 100
                
                if regression_percent < -10.0:  # 10% regression
                    self.alerts.append({
                        'timestamp': current_metrics['timestamp'],
                        'type': 'regression_token_throughput',
                        'message': f'Token throughput regression: {regression_percent:.1f}% below baseline',
                        'severity': 'warning'
                    })
                    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        if not self.metrics_history:
            return {}
        return self.metrics_history[-1]
        
    def get_historical_data(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get historical data for specified duration."""
        cutoff_time = time.time() - (minutes * 60)
        
        return [
            metrics for metrics in self.metrics_history
            if datetime.fromisoformat(metrics['timestamp']).timestamp() > cutoff_time
        ]
        
    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']).timestamp() > cutoff_time
        ]
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics_history:
            return {}
            
        recent_data = self.get_historical_data(60)  # Last hour
        
        if not recent_data:
            return {}
            
        # Collect metrics
        tok_s_values = [m['performance'].get('tokens_per_second', 0) for m in recent_data]
        ttft_values = [m['performance'].get('ttft_ms', 0) for m in recent_data]
        spike_rate_values = [m['performance'].get('spike_rate_percent', 0) for m in recent_data]
        
        memory_values = [m['system'].get('memory_percent', 0) for m in recent_data]
        cpu_values = [m['system'].get('cpu_percent', 0) for m in recent_data]
        
        return {
            'token_throughput': {
                'current': tok_s_values[-1] if tok_s_values else 0,
                'average': statistics.mean(tok_s_values) if tok_s_values else 0,
                'min': min(tok_s_values) if tok_s_values else 0,
                'max': max(tok_s_values) if tok_s_values else 0,
                'target': self.config.target_tok_s
            },
            'ttft': {
                'current': ttft_values[-1] if ttft_values else 0,
                'average': statistics.mean(ttft_values) if ttft_values else 0,
                'min': min(ttft_values) if ttft_values else 0,
                'max': max(ttft_values) if ttft_values else 0,
                'target': self.config.target_ttft_ms
            },
            'spike_rate': {
                'current': spike_rate_values[-1] if spike_rate_values else 0,
                'average': statistics.mean(spike_rate_values) if spike_rate_values else 0,
                'range': self.config.target_spike_rate_range
            },
            'system': {
                'memory_avg_percent': statistics.mean(memory_values) if memory_values else 0,
                'cpu_avg_percent': statistics.mean(cpu_values) if cpu_values else 0,
                'memory_current_percent': memory_values[-1] if memory_values else 0,
                'cpu_current_percent': cpu_values[-1] if cpu_values else 0
            },
            'uptime_hours': (time.time() - self.start_time) / 3600 if self.start_time else 0
        }
        
    def export_data(self, filename: str = None) -> str:
        """Export monitoring data to file."""
        if not filename:
            timestamp = int(time.time())
            filename = self.data_dir / f"monitoring_export_{timestamp}.json"
            
        data = {
            'config': asdict(self.config),
            'start_time': self.start_time,
            'metrics_history': list(self.metrics_history),
            'alerts': self.alerts,
            'summary': self.get_performance_summary()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        logging.info(f"Monitoring data exported to {filename}")
        return str(filename)


class PerformanceDashboard:
    """Web dashboard for performance monitoring."""
    
    def __init__(self, monitor: PerformanceMonitor, config: MonitoringConfig):
        self.monitor = monitor
        self.config = config
        
    async def start_server(self):
        """Start the web dashboard server."""
        if not HAS_AIOHTTP:
            print("Error: aiohttp not available. Install with: pip install aiohttp")
            return
            
        app = web.Application()
        
        # Add routes
        app.router.add_get('/', self.dashboard_page)
        app.router.add_get('/api/metrics', self.metrics_api)
        app.router.add_get('/api/summary', self.summary_api)
        app.router.add_get('/api/alerts', self.alerts_api)
        app.router.add_get('/api/export', self.export_api)
        app.router.add_static('/static', Path(__file__).parent / 'dashboard_static')
        
        # Start server
        runner = web_runner.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, 'localhost', self.config.port)
        await site.start()
        
        print(f"🚀 Performance Dashboard running at http://localhost:{self.config.port}")
        print("Press Ctrl+C to stop")
        
        return runner
        
    async def dashboard_page(self, request):
        """Main dashboard HTML page."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Performance Monitoring Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5; 
        }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
        }
        .metrics-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
            margin-bottom: 20px; 
        }
        .metric-card { 
            background: white; 
            padding: 20px; 
            border-radius: 10px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }
        .metric-value { 
            font-size: 2em; 
            font-weight: bold; 
            margin: 10px 0; 
        }
        .metric-label { 
            color: #666; 
            font-size: 0.9em; 
        }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
        .alerts { 
            background: white; 
            padding: 20px; 
            border-radius: 10px; 
            margin-top: 20px; 
        }
        .alert-item { 
            padding: 10px; 
            margin: 5px 0; 
            border-radius: 5px; 
        }
        .alert-warning { background-color: #fff3cd; border-left: 4px solid #ffc107; }
        .alert-critical { background-color: #f8d7da; border-left: 4px solid #dc3545; }
        .refresh-time { 
            position: fixed; 
            top: 10px; 
            right: 10px; 
            background: rgba(0,0,0,0.7); 
            color: white; 
            padding: 5px 10px; 
            border-radius: 5px; 
            font-size: 0.8em; 
        }
    </style>
</head>
<body>
    <div class="refresh-time">Last update: <span id="refresh-time">Loading...</span></div>
    
    <div class="header">
        <h1>🚀 Performance Monitoring Dashboard</h1>
        <p>Real-time monitoring of tok/s, TTFT, memory usage, and system health</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Token Throughput</div>
            <div class="metric-value" id="tok-throughput">--</div>
            <div class="metric-label">tokens/second (Target: ≥30)</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Time to First Token</div>
            <div class="metric-value" id="ttft">--</div>
            <div class="metric-label">milliseconds (Target: <5)</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Spike Rate</div>
            <div class="metric-value" id="spike-rate">--</div>
            <div class="metric-label">percentage (Target: 5-15%)</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Memory Usage</div>
            <div class="metric-value" id="memory-usage">--</div>
            <div class="metric-label">percentage</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">CPU Usage</div>
            <div class="metric-value" id="cpu-usage">--</div>
            <div class="metric-label">percentage</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Retrieval Latency</div>
            <div class="metric-value" id="retrieval-latency">--</div>
            <div class="metric-label">milliseconds (Target: <20)</div>
        </div>
    </div>
    
    <div class="alerts">
        <h3>📊 System Summary</h3>
        <div id="system-summary">Loading...</div>
        
        <h3>🚨 Recent Alerts</h3>
        <div id="alerts-list">No alerts</div>
    </div>
    
    <script>
        function updateMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    if (data.current) {
                        // Update metrics
                        document.getElementById('tok-throughput').textContent = 
                            data.current.performance.tokens_per_second.toFixed(1);
                        document.getElementById('ttft').textContent = 
                            data.current.performance.ttft_ms.toFixed(2);
                        document.getElementById('spike-rate').textContent = 
                            data.current.performance.spike_rate_percent.toFixed(1);
                        document.getElementById('memory-usage').textContent = 
                            data.current.system.memory_percent.toFixed(1);
                        document.getElementById('cpu-usage').textContent = 
                            data.current.system.cpu_percent.toFixed(1);
                        document.getElementById('retrieval-latency').textContent = 
                            data.current.performance.retrieval_latency_ms.toFixed(1);
                        
                        // Update status colors
                        updateStatusColors(data.current);
                    }
                    
                    document.getElementById('refresh-time').textContent = 
                        new Date().toLocaleTimeString();
                })
                .catch(error => console.error('Error fetching metrics:', error));
        }
        
        function updateSummary() {
            fetch('/api/summary')
                .then(response => response.json())
                .then(data => {
                    const summaryElement = document.getElementById('system-summary');
                    summaryElement.innerHTML = `
                        <p><strong>Uptime:</strong> ${data.uptime_hours.toFixed(1)} hours</p>
                        <p><strong>Avg Token Throughput:</strong> ${data.token_throughput.average.toFixed(1)} tok/s</p>
                        <p><strong>Avg TTFT:</strong> ${data.ttft.average.toFixed(2)} ms</p>
                        <p><strong>Avg Memory:</strong> ${data.system.memory_avg_percent.toFixed(1)}%</p>
                        <p><strong>Avg CPU:</strong> ${data.system.cpu_avg_percent.toFixed(1)}%</p>
                    `;
                })
                .catch(error => console.error('Error fetching summary:', error));
        }
        
        function updateAlerts() {
            fetch('/api/alerts')
                .then(response => response.json())
                .then(data => {
                    const alertsElement = document.getElementById('alerts-list');
                    if (data.alerts && data.alerts.length > 0) {
                        alertsElement.innerHTML = data.alerts.slice(-5).map(alert => 
                            `<div class="alert-item alert-${alert.severity}">
                                <strong>${alert.type}</strong>: ${alert.message} (${alert.timestamp})
                            </div>`
                        ).join('');
                    } else {
                        alertsElement.innerHTML = 'No recent alerts';
                    }
                })
                .catch(error => console.error('Error fetching alerts:', error));
        }
        
        function updateStatusColors(data) {
            // Tok/s status
            const tokS = data.performance.tokens_per_second;
            const tokSElement = document.getElementById('tok-throughput');
            if (tokS >= 30) {
                tokSElement.className = 'metric-value status-good';
            } else if (tokS >= 25) {
                tokSElement.className = 'metric-value status-warning';
            } else {
                tokSElement.className = 'metric-value status-critical';
            }
            
            // TTFT status
            const ttft = data.performance.ttft_ms;
            const ttftElement = document.getElementById('ttft');
            if (ttft < 5) {
                ttftElement.className = 'metric-value status-good';
            } else if (ttft < 10) {
                ttftElement.className = 'metric-value status-warning';
            } else {
                ttftElement.className = 'metric-value status-critical';
            }
            
            // Memory status
            const memory = data.system.memory_percent;
            const memoryElement = document.getElementById('memory-usage');
            if (memory < 80) {
                memoryElement.className = 'metric-value status-good';
            } else if (memory < 90) {
                memoryElement.className = 'metric-value status-warning';
            } else {
                memoryElement.className = 'metric-value status-critical';
            }
        }
        
        // Update every second
        updateMetrics();
        updateSummary();
        updateAlerts();
        
        setInterval(() => {
            updateMetrics();
            updateSummary();
            updateAlerts();
        }, 1000);
        
        // Auto-open browser
        setTimeout(() => {
            window.open('http://localhost:""" + str(self.config.port) + """', '_blank');
        }, 1000);
    </script>
</body>
</html>
        """
        
        return web.Response(text=html, content_type='text/html')
        
    async def metrics_api(self, request):
        """API endpoint for current metrics."""
        current = self.monitor.get_current_metrics()
        historical = self.monitor.get_historical_data(60)
        
        return web.json_response({
            'current': current,
            'historical': historical[-100:],  # Last 100 data points
            'count': len(historical)
        })
        
    async def summary_api(self, request):
        """API endpoint for performance summary."""
        summary = self.monitor.get_performance_summary()
        return web.json_response(summary)
        
    async def alerts_api(self, request):
        """API endpoint for alerts."""
        alerts = self.monitor.get_alerts(24)  # Last 24 hours
        return web.json_response({
            'alerts': alerts,
            'count': len(alerts)
        })
        
    async def export_api(self, request):
        """API endpoint for data export."""
        filename = self.monitor.export_data()
        return web.json_response({
            'export_file': filename,
            'message': 'Data exported successfully'
        })


async def main():
    """Main CLI interface for performance dashboard."""
    parser = argparse.ArgumentParser(
        description="Performance Monitoring Dashboard - Real-time monitoring with alerting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start dashboard on default port
  python performance_dashboard.py
  
  # Start on custom port with baseline
  python performance_dashboard.py --port 9000 --baseline results/baseline.json
  
  # Monitoring-only mode (no web interface)
  python performance_dashboard.py --monitor-only
  
  # Export current monitoring data
  python performance_dashboard.py --export
        """
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Dashboard port (default: 8080)"
    )
    
    parser.add_argument(
        "--update-interval",
        type=int,
        default=1000,
        help="Update interval in milliseconds (default: 1000)"
    )
    
    parser.add_argument(
        "--baseline", "-b",
        type=str,
        help="Baseline file for regression detection"
    )
    
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help="Run monitoring without web interface"
    )
    
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export current monitoring data and exit"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="monitoring_data",
        help="Directory for storing monitoring data"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = MonitoringConfig(
        port=args.port,
        update_interval_ms=args.update_interval,
        baseline_file=args.baseline,
        data_dir=args.data_dir
    )
    
    # Initialize monitor
    monitor = PerformanceMonitor(config)
    
    try:
        if args.export:
            # Export data and exit
            monitor.start_monitoring()
            await asyncio.sleep(2)  # Wait for some data
            filename = monitor.export_data()
            print(f"Monitoring data exported to: {filename}")
            return 0
            
        elif args.monitor_only:
            # Run monitoring only
            print("Starting performance monitoring...")
            monitor.start_monitoring()
            
            try:
                while True:
                    await asyncio.sleep(30)
                    summary = monitor.get_performance_summary()
                    print(f"Token throughput: {summary['token_throughput']['current']:.1f} tok/s")
                    print(f"TTFT: {summary['ttft']['current']:.2f}ms")
                    print(f"Memory: {summary['system']['memory_current_percent']:.1f}%")
                    print("---")
            except KeyboardInterrupt:
                print("\nStopping monitoring...")
                
        else:
            # Run dashboard
            print("🚀 Starting Performance Dashboard...")
            print(f"📊 Monitoring configuration:")
            print(f"   Port: {config.port}")
            print(f"   Update interval: {config.update_interval_ms}ms")
            print(f"   Alert thresholds:")
            print(f"     • Tok/s: <{config.alert_tok_s_threshold}")
            print(f"     • TTFT: >{config.alert_ttft_threshold_ms}ms")
            print(f"     • Memory: >{config.alert_memory_threshold_percent}%")
            
            if config.baseline_file:
                print(f"   Baseline: {config.baseline_file}")
            else:
                print("   Baseline: None")
                
            # Start monitoring
            monitor.start_monitoring()
            
            # Start dashboard server
            dashboard = PerformanceDashboard(monitor, config)
            runner = await dashboard.start_server()
            
            try:
                # Keep running
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down dashboard...")
                
        # Cleanup
        monitor.stop_monitoring()
        
        if 'runner' in locals():
            await runner.cleanup()
            
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
