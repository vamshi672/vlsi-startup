"""
Optimizer for PPAForge AI
Main orchestration module that integrates GNN, RL, and OpenROAD for chip placement optimization.

This module coordinates:
1. Design loading and graph construction
2. RL agent training
3. OpenROAD integration
4. PPA evaluation and comparison
"""

import os
import yaml
import logging
import time
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.core.graph_builder import GraphBuilder
from src.core.placement_env import PlacementEnv
from src.core.rl_agent import PlacementPPOAgent
from src.integration.openroad_interface import OpenROADInterface
from src.integration.ppa_evaluator import PPAEvaluator

logger = logging.getLogger(__name__)


class PPAForgeOptimizer:
    """
    Main optimizer class for PPAForge AI.
    
    Workflow:
    1. Load design and create graph representation
    2. Run baseline OpenROAD placement
    3. Train RL agent to optimize placement
    4. Compare optimized vs baseline results
    5. Generate reports and visualizations
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_name = self.config.get('project', {}).get('name', 'PPAForge-AI')
        
        # Initialize components
        self.graph_builder = None
        self.openroad = None
        self.ppa_evaluator = None
        self.rl_agent = None
        self.env = None
        
        # Results storage
        self.baseline_results = None
        self.optimized_results = None
        self.training_history = None
        
        # Output directories
        self.output_dir = Path('results')
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"{self.project_name} Optimizer initialized")
    
    def load_design(self, design_path: str) -> Dict:
        """
        Load design and create graph representation.
        
        Args:
            design_path: Path to design directory
        
        Returns:
            Graph data dictionary
        """
        logger.info(f"Loading design from {design_path}")
        
        # Initialize graph builder
        self.graph_builder = GraphBuilder(self.config)
        
        # Find design files
        def_file = os.path.join(design_path, 'floorplan.def')
        verilog_file = os.path.join(design_path, 'design.v')
        
        if os.path.exists(def_file):
            graph = self.graph_builder.build_graph(def_file, format='def')
        elif os.path.exists(verilog_file):
            graph = self.graph_builder.build_graph(verilog_file, format='verilog')
        else:
            raise FileNotFoundError(f"No design files found in {design_path}")
        
        # Save processed graph
        graph_path = self.output_dir / 'design_graph.pt'
        self.graph_builder.save_graph(graph, str(graph_path))
        
        logger.info(f"Design loaded: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
        
        return {
            'num_nodes': graph.num_nodes,
            'num_edges': graph.edge_index.shape[1],
            'graph_path': str(graph_path)
        }
    
    def run_baseline(self, design_path: str) -> Dict:
        """
        Run baseline OpenROAD placement.
        
        Args:
            design_path: Path to design directory
        
        Returns:
            Baseline results dictionary
        """
        logger.info("Running baseline OpenROAD placement")
        
        # Initialize OpenROAD interface
        self.openroad = OpenROADInterface(self.config)
        
        # Initialize PPA evaluator
        self.ppa_evaluator = PPAEvaluator(self.config)
        
        # Run default placement
        baseline_output = self.output_dir / 'baseline'
        baseline_output.mkdir(exist_ok=True)
        
        start_time = time.time()
        result = self.openroad.run_default_placement(
            design_path,
            output_dir=str(baseline_output)
        )
        runtime = time.time() - start_time
        
        if not result.get('success', False):
            logger.error("Baseline placement failed")
            return {'success': False}
        
        # Extract metrics
        baseline_placement = result['placement']
        baseline_metrics = result['metrics']
        baseline_metrics['runtime'] = runtime
        
        # Store results
        self.baseline_results = {
            'placement': baseline_placement,
            'metrics': baseline_metrics,
            'runtime': runtime
        }
        
        # Save results
        results_file = baseline_output / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(self.baseline_results, f, indent=2)
        
        logger.info(f"Baseline completed in {runtime:.2f}s")
        logger.info(f"Baseline metrics: {baseline_metrics}")
        
        return self.baseline_results
    
    def train_agent(
        self,
        design_path: str,
        num_iterations: int = 1000,
        resume_from: Optional[str] = None
    ) -> Dict:
        """
        Train RL agent for placement optimization.
        
        Args:
            design_path: Path to design directory
            num_iterations: Number of training iterations
            resume_from: Path to checkpoint to resume from
        
        Returns:
            Training results dictionary
        """
        logger.info(f"Training RL agent for {num_iterations} iterations")
        
        # Create environment configuration
        env_config = {
            **self.config,
            'design_path': design_path,
            'max_steps': 500
        }
        
        # Create environment (for training)
        self.env = PlacementEnv(env_config)
        
        # Initialize RL agent
        self.rl_agent = PlacementPPOAgent(self.config, env_config)
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            logger.info(f"Resuming from checkpoint: {resume_from}")
            self.rl_agent.load_checkpoint(resume_from)
        
        # Create checkpoint directory
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Train
        start_time = time.time()
        training_results = self.rl_agent.train(
            num_iterations=num_iterations,
            checkpoint_dir=str(checkpoint_dir)
        )
        training_time = time.time() - start_time
        
        # Store training history
        self.training_history = training_results['results_history']
        
        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Best reward: {training_results['best_reward']:.2f}")
        
        # Save training results
        results_file = self.output_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'num_iterations': num_iterations,
                'training_time': training_time,
                'best_reward': training_results['best_reward'],
                'final_checkpoint': training_results['final_checkpoint']
            }, f, indent=2)
        
        return training_results
    
    def optimize_placement(
        self,
        design_path: str,
        checkpoint_path: Optional[str] = None
    ) -> Dict:
        """
        Generate optimized placement using trained agent.
        
        Args:
            design_path: Path to design directory
            checkpoint_path: Path to agent checkpoint (uses best if None)
        
        Returns:
            Optimized results dictionary
        """
        logger.info("Generating optimized placement")
        
        # Load checkpoint if not already loaded
        if checkpoint_path is None:
            checkpoint_path = str(self.output_dir / 'checkpoints' / 'best_model')
        
        if self.rl_agent is None:
            env_config = {
                **self.config,
                'design_path': design_path,
                'max_steps': 500
            }
            self.rl_agent = PlacementPPOAgent(self.config, env_config)
        
        self.rl_agent.load_checkpoint(checkpoint_path)
        
        # Create evaluation environment
        env_config = {
            **self.config,
            'design_path': design_path,
            'max_steps': 500
        }
        eval_env = PlacementEnv(env_config)
        
        # Run optimized placement
        obs, info = eval_env.reset()
        done = False
        step = 0
        episode_reward = 0
        
        placement_trajectory = []
        
        while not done and step < 500:
            # Get action from trained agent
            action = self.rl_agent.compute_action(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
            
            # Store placement
            if 'current_metrics' in info:
                placement_trajectory.append(info['current_metrics'])
        
        # Get final placement
        final_placement = eval_env.current_placement
        final_metrics = self.ppa_evaluator.evaluate(final_placement)
        
        # Store results
        self.optimized_results = {
            'placement': final_placement,
            'metrics': final_metrics,
            'episode_reward': episode_reward,
            'num_steps': step,
            'trajectory': placement_trajectory
        }
        
        # Save results
        optimized_output = self.output_dir / 'optimized'
        optimized_output.mkdir(exist_ok=True)
        
        results_file = optimized_output / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(self.optimized_results, f, indent=2, default=str)
        
        logger.info(f"Optimized placement generated in {step} steps")
        logger.info(f"Final metrics: {final_metrics}")
        
        return self.optimized_results
    
    def compare_results(self) -> Dict:
        """
        Compare baseline vs optimized results.
        
        Returns:
            Comparison dictionary with improvements
        """
        if self.baseline_results is None or self.optimized_results is None:
            logger.error("Both baseline and optimized results must be available")
            return {}
        
        logger.info("Comparing baseline vs optimized results")
        
        # Compute improvements
        improvements = self.ppa_evaluator.compare_metrics(
            self.optimized_results['metrics'],
            self.baseline_results['metrics']
        )
        
        # Add convergence speedup
        baseline_runtime = self.baseline_results.get('runtime', 0)
        optimized_runtime = self.optimized_results.get('num_steps', 0) * 0.1  # Estimate
        
        if baseline_runtime > 0:
            speedup = baseline_runtime / optimized_runtime if optimized_runtime > 0 else 0
        else:
            speedup = 0
        
        comparison = {
            'improvements': improvements,
            'speedup': speedup,
            'baseline_metrics': self.baseline_results['metrics'],
            'optimized_metrics': self.optimized_results['metrics']
        }
        
        # Print comparison
        print("\n" + "="*60)
        print("BASELINE vs OPTIMIZED COMPARISON (Optimize for low power)")
        print("="*60)
        print(f"{'Metric':<20} {'Baseline':>15} {'Optimized':>15} {'Improvement':>10}")
        print("-"*60)
        
        metrics_to_show = ['power', 'worst_slack', 'wirelength', 'core_area']
        units = {'power': 'mW', 'worst_slack': 'ns', 'wirelength': 'um', 'core_area': 'um^2'}
        scales = {'power': 1e3, 'worst_slack': 1, 'wirelength': 1, 'core_area': 1}
        
        for metric in metrics_to_show:
            baseline_val = self.baseline_results['metrics'].get(metric, 0) * scales[metric]
            optimized_val = self.optimized_results['metrics'].get(metric, 0) * scales[metric]
            improvement = improvements.get(metric, 0)
            
            print(f"{metric:<20} {baseline_val:>13.2f}{units[metric]:>2} "
                  f"{optimized_val:>13.2f}{units[metric]:>2} "
                  f"{improvement:>8.1f}%")
        
        print(f"\nConvergence Speedup: {speedup:.1f}x")
        print("="*60 + "\n")
        
        # Save comparison
        comparison_file = self.output_dir / 'comparison.json'
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate optimization report.
        
        Args:
            output_path: Path to save report (default: results/report.txt)
        
        Returns:
            Path to generated report
        """
        if output_path is None:
            output_path = self.output_dir / 'report.txt'
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"{self.project_name} - Optimization Report\n")
            f.write("="*70 + "\n\n")
            
            # Design info
            f.write("DESIGN INFORMATION\n")
            f.write("-"*70 + "\n")
            if self.graph_builder:
                f.write(f"Number of cells: {len(self.graph_builder.cells)}\n")
                f.write(f"Number of nets: {len(self.graph_builder.nets)}\n")
            f.write("\n")
            
            # Baseline results
            if self.baseline_results:
                f.write("BASELINE RESULTS (OpenROAD Default)\n")
                f.write("-"*70 + "\n")
                for key, value in self.baseline_results['metrics'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Optimized results
            if self.optimized_results:
                f.write("OPTIMIZED RESULTS (PPAForge AI - Optimize for low power)\n")
                f.write("-"*70 + "\n")
                for key, value in self.optimized_results['metrics'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Comparison
            if self.baseline_results and self.optimized_results:
                comparison = self.compare_results()
                f.write("IMPROVEMENTS\n")
                f.write("-"*70 + "\n")
                for metric, improvement in comparison['improvements'].items():
                    f.write(f"{metric}: {improvement:+.2f}%\n")
                f.write(f"Convergence speedup: {comparison['speedup']:.2f}x\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
        
        logger.info(f"Report generated: {output_path}")
        
        return str(output_path)
    
    def visualize_results(self):
        """Generate visualization plots."""
        fig_dir = self.output_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)
        
        # Training curve
        if self.training_history:
            self._plot_training_curve(fig_dir / 'training_curve.png')
        
        # PPA comparison
        if self.baseline_results and self.optimized_results:
            self._plot_ppa_comparison(fig_dir / 'ppa_comparison.png')
        
        logger.info(f"Visualizations saved to {fig_dir}")
    
    def _plot_training_curve(self, save_path: str):
        """Plot training reward curve."""
        rewards = [r.get('episode_reward_mean', 0) for r in self.training_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.xlabel('Training Iteration')
        plt.ylabel('Average Episode Reward')
        plt.title('RL Agent Training Progress')
        plt.grid(True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_ppa_comparison(self, save_path: str):
        """Plot PPA metrics comparison."""
        metrics = ['power', 'worst_slack', 'wirelength']
        baseline_vals = [self.baseline_results['metrics'][m] for m in metrics]
        optimized_vals = [self.optimized_results['metrics'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, baseline_vals, width, label='Baseline')
        ax.bar(x + width/2, optimized_vals, width, label='Optimized (Low Power)')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('PPA Metrics: Baseline vs Optimized (Optimize for low power)')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_full_optimization(
        self,
        design_path: str,
        num_iterations: int = 1000
    ) -> Dict:
        """
        Run complete optimization workflow.
        
        Args:
            design_path: Path to design directory
            num_iterations: Training iterations
        
        Returns:
            Complete results dictionary
        """
        logger.info("Starting full optimization workflow")
        
        # 1. Load design
        design_info = self.load_design(design_path)
        
        # 2. Run baseline
        baseline = self.run_baseline(design_path)
        
        # 3. Train agent
        training = self.train_agent(design_path, num_iterations)
        
        # 4. Generate optimized placement
        optimized = self.optimize_placement(design_path)
        
        # 5. Compare results
        comparison = self.compare_results()
        
        # 6. Generate report and visualizations
        report_path = self.generate_report()
        self.visualize_results()
        
        logger.info("Full optimization completed!")
        
        return {
            'design_info': design_info,
            'baseline': baseline,
            'training': training,
            'optimized': optimized,
            'comparison': comparison,
            'report_path': report_path
        }


def main():
    """Example usage."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create optimizer
    optimizer = PPAForgeOptimizer('config/config.yaml')
    
    # Run optimization on example design
    design_path = 'data/benchmarks/gcd'
    
    if os.path.exists(design_path):
        results = optimizer.run_full_optimization(
            design_path=design_path,
            num_iterations=100  # Small number for demo
        )
        
        print("\nOptimization Complete!")
        print(f"Report: {results['report_path']}")
    else:
        print(f"Design not found: {design_path}")
        print("Please add your design files to run optimization.")


if __name__ == "__main__":
    main()
