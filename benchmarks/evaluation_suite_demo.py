#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite Demo

Demonstrates the usage of the ComprehensiveEvaluationSuite for evaluating
Mini BAI model capabilities across multiple dimensions.

Usage:
    python evaluation_suite_demo.py --preset quick
    python evaluation_suite_demo.py --preset full
    python evaluation_suite_demo.py --custom --evaluators mmlu_lite gsm8k_lite
    python evaluation_suite_demo.py --ab-test
"""

import asyncio
import argparse
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

# Add the current directory to the path so we can import the evaluation suite
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comprehensive_evaluation_suite import (
    ComprehensiveEvaluationSuite,
    MMLULiteEvaluator,
    GSM8KLiteEvaluator,
    VQALiteEvaluator,
    ToxicityEvaluator,
    EnergyMonitor,
    ABTestingFramework
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MiniBAIModel:
    """Mock Mini BAI Model for demonstration purposes"""
    
    def __init__(self, version: str = "1.0"):
        self.version = version
        self.name = f"Mini BAI v{version}"
        logger.info(f"Initialized {self.name}")
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response (mock implementation)"""
        # Simulate inference time
        inference_time = 0.02 + (len(prompt) * 0.0001)
        await asyncio.sleep(inference_time)
        
        # Mock response based on prompt type
        if any(word in prompt.lower() for word in ['question', 'answer', 'solve']):
            if 'math' in prompt.lower() or 'derivative' in prompt.lower():
                return "Based on the mathematical principle, the answer is 2x."
            elif 'history' in prompt.lower():
                return "Historically, this event occurred in 1945."
            else:
                return "The answer depends on various factors and context."
        
        elif any(word in prompt.lower() for word in ['color', 'image', 'picture']):
            return "The object appears to be red in color."
        
        elif any(word in prompt.lower() for word in ['toxic', 'harmful', 'offensive']):
            return "I cannot provide harmful or offensive content. I am designed to be helpful and safe."
        
        else:
            return f"This is a response from {self.name}. How can I help you today?"
    
    async def analyze_image(self, image_data: Any, question: str) -> str:
        """Analyze image and answer question (mock implementation)"""
        await asyncio.sleep(0.03)  # Simulate VQA inference time
        
        if 'color' in question.lower():
            return "red"
        elif 'where' in question.lower() and 'sitting' in question.lower():
            return "table"
        elif 'how many' in question.lower():
            return "3"
        elif 'who' in question.lower():
            return "person"
        else:
            return "I can see various objects in the image."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.name,
            'version': self.version,
            'parameters': '7B',
            'context_length': 2048,
            'vocab_size': 32000,
            'capabilities': ['text_generation', 'question_answering', 'image_analysis']
        }


class EvaluationDemo:
    """Demo class for evaluation suite"""
    
    def __init__(self, output_dir: str = "demo_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logger
        
    async def run_preset_evaluation(self, preset: str = "quick") -> Dict[str, Any]:
        """Run evaluation with predefined presets"""
        self.logger.info(f"Running {preset} preset evaluation")
        
        # Initialize model
        model = MiniBAIModel()
        
        # Configure evaluation suite based on preset
        if preset == "quick":
            config = {
                'mmlu_lite': {
                    'max_samples': 10,
                    'pass_at_k': 1,
                    'few_shot_examples': 0
                },
                'gsm8k_lite': {
                    'difficulty': 'easy',
                    'max_samples': 5,
                    'show_reasoning': False
                },
                'vqa_lite': {
                    'max_samples': 3,
                    'question_types': ['what', 'where']
                },
                'toxicity': {
                    'threshold': 0.5
                },
                'energy_monitoring': {
                    'enabled': True
                }
            }
            
        elif preset == "full":
            config = {
                'mmlu_lite': {
                    'max_samples': 50,
                    'pass_at_k': 1,
                    'few_shot_examples': 5
                },
                'gsm8k_lite': {
                    'difficulty': 'medium',
                    'max_samples': 25,
                    'show_reasoning': True
                },
                'vqa_lite': {
                    'max_samples': 20,
                    'question_types': ['what', 'where', 'when', 'who', 'why', 'how']
                },
                'toxicity': {
                    'threshold': 0.5
                },
                'energy_monitoring': {
                    'enabled': True
                }
            }
            
        else:
            raise ValueError(f"Unknown preset: {preset}")
        
        # Initialize evaluation suite
        suite = ComprehensiveEvaluationSuite(config=config)
        
        # Run evaluation
        start_time = time.time()
        results = await suite.run_full_evaluation(model)
        evaluation_time = time.time() - start_time
        
        # Save results
        self._save_results(results, f"{preset}_preset_results.json", {
            'preset': preset,
            'evaluation_time': evaluation_time,
            'model_info': model.get_model_info()
        })
        
        # Generate and save report
        self._generate_report(results, f"{preset}_preset_report.html")
        
        return results
    
    async def run_custom_evaluation(self, evaluators: List[str]) -> Dict[str, Any]:
        """Run evaluation with custom evaluator selection"""
        self.logger.info(f"Running custom evaluation with: {evaluators}")
        
        model = MiniBAIModel()
        suite = ComprehensiveEvaluationSuite()
        
        # Filter evaluators based on selection
        available_evaluators = ['mmlu_lite', 'gsm8k_lite', 'vqa_lite', 'toxicity', 'energy_monitoring']
        valid_evaluators = [e for e in evaluators if e in available_evaluators]
        
        if not valid_evaluators:
            raise ValueError(f"No valid evaluators selected. Available: {available_evaluators}")
        
        # Run selected evaluators
        start_time = time.time()
        results = await suite.evaluate_model(model, valid_evaluators)
        evaluation_time = time.time() - start_time
        
        # Save results
        self._save_results(results, "custom_evaluation_results.json", {
            'selected_evaluators': valid_evaluators,
            'evaluation_time': evaluation_time,
            'model_info': model.get_model_info()
        })
        
        # Generate and save report
        self._generate_report(results, "custom_evaluation_report.html")
        
        return results
    
    async def run_ab_testing(self) -> Dict[str, Any]:
        """Run A/B testing comparison"""
        self.logger.info("Running A/B testing comparison")
        
        # Initialize two model versions
        model_v1 = MiniBAIModel(version="1.0")
        model_v2 = MiniBAIModel(version="2.0")
        
        # Configure A/B testing
        suite = ComprehensiveEvaluationSuite()
        
        # Run comparison
        start_time = time.time()
        ab_results = await suite.compare_models([model_v1, model_v2])
        comparison_time = time.time() - start_time
        
        # Extract A/B test results
        ab_test_result = ab_results.get('ab_comparison')
        
        # Save A/B testing results
        ab_data = {
            'comparison_time': comparison_time,
            'models_compared': ['model_v1', 'model_v2'],
            'statistical_test': ab_test_result.statistical_test if ab_test_result else 'unknown',
            'alpha_level': ab_test_result.alpha_level if ab_test_result else 0.05,
            'p_values': ab_test_result.p_values if ab_test_result else {},
            'significant_differences': ab_test_result.significant_differences if ab_test_result else [],
            'effect_sizes': ab_test_result.effect_sizes if ab_test_result else {},
            'model1_metrics': ab_test_result.model1_metrics if ab_test_result else {},
            'model2_metrics': ab_test_result.model2_metrics if ab_test_result else {}
        }
        
        with open(self.output_dir / "ab_testing_results.json", 'w') as f:
            json.dump(ab_data, f, indent=2)
        
        # Generate A/B testing report
        self._generate_ab_testing_report(ab_data, "ab_testing_report.html")
        
        return ab_results
    
    async def run_individual_evaluator_demo(self):
        """Demonstrate individual evaluator usage"""
        self.logger.info("Running individual evaluator demonstrations")
        
        model = MiniBAIModel()
        
        # MMLU-lite Demo
        print("\n=== MMLU-lite Evaluator Demo ===")
        mmlu_evaluator = MMLULiteEvaluator(
            pass_at_k=1,
            subjects=['mathematics', 'science'],
            max_samples=5
        )
        mmlu_result = await mmlu_evaluator.evaluate(model)
        print(f"MMLU-lite Accuracy: {mmlu_result.accuracy:.3f}")
        print(f"Per-subject accuracy: {mmlu_result.per_subject_accuracy}")
        
        # GSM8K-lite Demo
        print("\n=== GSM8K-lite Evaluator Demo ===")
        gsm8k_evaluator = GSM8KLiteEvaluator(
            difficulty='easy',
            show_reasoning=True,
            max_samples=3
        )
        gsm8k_result = await gsm8k_evaluator.evaluate(model)
        print(f"GSM8K-lite Accuracy: {gsm8k_result.accuracy:.3f}")
        print(f"Difficulty: {gsm8k_result.difficulty}")
        print(f"Reasoning Quality: {gsm8k_result.reasoning_quality:.3f}")
        
        # VQA-lite Demo
        print("\n=== VQA-lite Evaluator Demo ===")
        vqa_evaluator = VQALiteEvaluator(
            question_types=['what', 'where'],
            max_samples=2
        )
        vqa_result = await vqa_evaluator.evaluate(model)
        print(f"VQA-lite Accuracy: {vqa_result.accuracy:.3f}")
        print(f"Question type performance: {vqa_result.question_types}")
        
        # Toxicity Demo
        print("\n=== Toxicity Evaluator Demo ===")
        toxicity_evaluator = ToxicityEvaluator(threshold=0.5)
        toxicity_result = await toxicity_evaluator.evaluate(model)
        print(f"Toxicity Rate: {toxicity_result.toxicity_rate:.3f}")
        print(f"False Positive Rate: {toxicity_result.false_positive_rate:.3f}")
        print(f"Categories Detected: {toxicity_result.categories_detected}")
        
        # Energy Monitoring Demo
        print("\n=== Energy Monitor Demo ===")
        energy_monitor = EnergyMonitor(monitoring_interval=1.0)
        energy_result = await energy_monitor.measure(model)
        print(f"Energy per Token: {energy_result.energy_per_token:.3f}J")
        print(f"Power Consumption: {energy_result.power_consumption:.3f}W")
        print(f"Efficiency Ratio: {energy_result.efficiency_ratio:.3f} tok/s/W")
        
    def _save_results(self, results: Dict[str, Any], filename: str, metadata: Dict[str, Any]):
        """Save evaluation results to file"""
        export_data = {
            'timestamp': time.time(),
            'metadata': metadata,
            'results': {}
        }
        
        for name, result in results.items():
            export_data['results'][name] = {
                'metric_name': result.metric_name,
                'value': result.value,
                'std_dev': result.std_dev,
                'metadata': result.metadata,
                'timestamp': result.timestamp.isoformat() if result.timestamp else None
            }
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def _generate_report(self, results: Dict[str, Any], filename: str):
        """Generate HTML evaluation report"""
        suite = ComprehensiveEvaluationSuite()
        report = suite.generate_report(results)
        
        filepath = self.output_dir / filename
        report.save(str(filepath))
        
        self.logger.info(f"Report saved to {filepath}")
    
    def _generate_ab_testing_report(self, ab_data: Dict[str, Any], filename: str):
        """Generate A/B testing HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>A/B Testing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .comparison {{ margin: 10px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .significant {{ background-color: #ffe6e6; }}
                .metric {{ font-weight: bold; color: #333; }}
                .value {{ font-size: 1.1em; color: #0066cc; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Mini BAI A/B Testing Report</h1>
                <p>Models Compared: {', '.join(ab_data['models_compared'])}</p>
                <p>Statistical Test: {ab_data['statistical_test']}</p>
                <p>Alpha Level: {ab_data['alpha_level']}</p>
                <p>Comparison Time: {ab_data['comparison_time']:.2f} seconds</p>
            </div>
            
            <h2>Statistical Results</h2>
            <h3>Significant Differences</h3>
            {self._format_list(ab_data['significant_differences'])}
            
            <h3>P-Values</h3>
            <table>
                <tr><th>Metric</th><th>P-Value</th><th>Significant</th></tr>
                {self._format_p_value_table(ab_data['p_values'], ab_data['alpha_level'])}
            </table>
            
            <h3>Effect Sizes</h3>
            <table>
                <tr><th>Metric</th><th>Effect Size</th><th>Interpretation</th></tr>
                {self._format_effect_size_table(ab_data['effect_sizes'])}
            </table>
            
            <h2>Model Comparison</h2>
            <table>
                <tr><th>Metric</th><th>Model 1</th><th>Model 2</th><th>Difference</th></tr>
                {self._format_model_comparison_table(ab_data['model1_metrics'], ab_data['model2_metrics'])}
            </table>
        </body>
        </html>
        """
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"A/B testing report saved to {filepath}")
    
    def _format_list(self, items: List[str]) -> str:
        """Format list as HTML"""
        if not items:
            return "<p>No significant differences found.</p>"
        
        html = "<ul>"
        for item in items:
            html += f"<li>{item}</li>"
        html += "</ul>"
        return html
    
    def _format_p_value_table(self, p_values: Dict[str, float], alpha: float) -> str:
        """Format p-value table"""
        html = ""
        for metric, p_value in p_values.items():
            significant = p_value < alpha
            color = "#ffe6e6" if significant else "#e6f3ff"
            html += f'<tr style="background-color: {color}">' \
                   f'<td>{metric}</td>' \
                   f'<td>{p_value:.4f}</td>' \
                   f'<td>{"Yes" if significant else "No"}</td>' \
                   f'</tr>'
        return html
    
    def _format_effect_size_table(self, effect_sizes: Dict[str, float]) -> str:
        """Format effect size table"""
        html = ""
        for metric, effect_size in effect_sizes.items():
            interpretation = self._interpret_effect_size(effect_size)
            html += f"<tr>" \
                   f"<td>{metric}</td>" \
                   f"<td>{effect_size:.3f}</td>" \
                   f"<td>{interpretation}</td>" \
                   f"</tr>"
        return html
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size"""
        if effect_size < 0.2:
            return "Negligible"
        elif effect_size < 0.5:
            return "Small"
        elif effect_size < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def _format_model_comparison_table(self, model1_metrics: Dict[str, float], 
                                     model2_metrics: Dict[str, float]) -> str:
        """Format model comparison table"""
        html = ""
        common_metrics = set(model1_metrics.keys()) & set(model2_metrics.keys())
        
        for metric in common_metrics:
            val1 = model1_metrics[metric]
            val2 = model2_metrics[metric]
            diff = val2 - val1
            html += f"<tr>" \
                   f"<td>{metric}</td>" \
                   f"<td>{val1:.3f}</td>" \
                   f"<td>{val2:.3f}</td>" \
                   f"<td>{diff:+.3f}</td>" \
                   f"</tr>"
        return html
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary to console"""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for name, result in results.items():
            print(f"{name.replace('_', ' ').title():<20}: {result.value:>8.3f}")
        
        print("\nTarget vs Actual:")
        targets = {
            'mmlu_lite': 0.70,
            'gsm8k_lite': 0.80,
            'vqa_lite': 0.65,
            'toxicity': 0.02,  # Note: lower is better for toxicity
            'energy_monitoring': 2.0  # Note: lower is better for energy
        }
        
        for name, result in results.items():
            if name in targets:
                target = targets[name]
                if name == 'toxicity' or name == 'energy_monitoring':
                    status = "✅ PASS" if result.value <= target else "❌ FAIL"
                    print(f"{name.replace('_', ' ').title():<20}: {result.value:>6.2f} (≤{target:.2f}) {status}")
                else:
                    status = "✅ PASS" if result.value >= target else "❌ FAIL"
                    print(f"{name.replace('_', ' ').title():<20}: {result.value:>6.1%} (≥{target:.0%}) {status}")


async def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation Suite Demo")
    
    # Main execution modes
    parser.add_argument('--preset', choices=['quick', 'full'], default='quick',
                       help='Run with predefined preset (default: quick)')
    parser.add_argument('--custom', action='store_true',
                       help='Run with custom evaluator selection')
    parser.add_argument('--evaluators', nargs='+', 
                       choices=['mmlu_lite', 'gsm8k_lite', 'vqa_lite', 'toxicity', 'energy_monitoring'],
                       help='Specify evaluators for custom mode')
    parser.add_argument('--ab-test', action='store_true',
                       help='Run A/B testing comparison')
    parser.add_argument('--individual-demo', action='store_true',
                       help='Demonstrate individual evaluators')
    
    # Output options
    parser.add_argument('--output-dir', default='demo_results',
                       help='Output directory for results (default: demo_results)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize demo
    demo = EvaluationDemo(output_dir=args.output_dir)
    
    try:
        if args.ab_test:
            print("🚀 Running A/B Testing Demo...")
            results = await demo.run_ab_testing()
            print("✅ A/B Testing completed successfully!")
            
        elif args.individual_demo:
            print("🔧 Running Individual Evaluator Demo...")
            await demo.run_individual_evaluator_demo()
            print("✅ Individual evaluator demo completed successfully!")
            
        elif args.custom:
            if not args.evaluators:
                print("❌ Please specify evaluators with --evaluators option")
                return
                
            print(f"🎯 Running Custom Evaluation with: {args.evaluators}")
            results = await demo.run_custom_evaluation(args.evaluators)
            demo.print_summary(results)
            print("✅ Custom evaluation completed successfully!")
            
        else:
            print(f"⚡ Running {args.preset} Preset Evaluation...")
            results = await demo.run_preset_evaluation(args.preset)
            demo.print_summary(results)
            print(f"✅ {args.preset} preset evaluation completed successfully!")
        
        print(f"\n📁 Results saved to: {args.output_dir}/")
        print(f"   - JSON data: *.json")
        print(f"   - HTML reports: *.html")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)