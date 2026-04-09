#!/usr/bin/env python3
"""
Training script for PPAForge AI
Command-line interface for training the RL agent.
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integration.optimizer import PPAForgeOptimizer


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description='Train PPAForge AI placement optimization agent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Design arguments
    parser.add_argument(
        '--design',
        type=str,
        required=True,
        help='Path to design directory (e.g., data/benchmarks/gcd)'
    )
    
    # Training arguments
    parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='Number of training iterations'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints'
    )
    
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    # Optimization objective
    parser.add_argument(
        '--objective',
        type=str,
        default='power',
        choices=['power', 'performance', 'area', 'balanced'],
        help='Primary optimization objective (Optimize for low power is default)'
    )
    
    # Logging
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("PPAForge AI - Training Script")
    logger.info("="*70)
    logger.info(f"Design: {args.design}")
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Objective: Optimize for low {args.objective}")
    logger.info(f"Config: {args.config}")
    logger.info("="*70)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update config with command line args
        config['optimization']['primary_objective'] = args.objective
        
        if args.wandb:
            config['logging']['wandb']['enabled'] = True
        
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        return 1
    
    # Verify design directory exists
    if not Path(args.design).exists():
        logger.error(f"Design directory not found: {args.design}")
        return 1
    
    try:
        # Create optimizer
        optimizer = PPAForgeOptimizer(args.config)
        
        # Load design
        logger.info("Loading design...")
        design_info = optimizer.load_design(args.design)
        logger.info(f"Design loaded: {design_info['num_nodes']} nodes, {design_info['num_edges']} edges")
        
        # Run baseline (for comparison)
        logger.info("Running baseline placement...")
        baseline_results = optimizer.run_baseline(args.design)
        
        if baseline_results.get('success'):
            logger.info(f"Baseline completed: {baseline_results['metrics']}")
        else:
            logger.warning("Baseline failed, continuing with training")
        
        # Train agent
        logger.info(f"Starting training for {args.iterations} iterations...")
        logger.info("Primary objective: Optimize for low power")
        
        training_results = optimizer.train_agent(
            design_path=args.design,
            num_iterations=args.iterations,
            resume_from=args.resume_from
        )
        
        logger.info("Training completed!")
        logger.info(f"Best reward: {training_results['best_reward']:.4f}")
        logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")
        
        # Generate optimized placement
        logger.info("Generating optimized placement...")
        optimized_results = optimizer.optimize_placement(args.design)
        logger.info(f"Optimization completed in {optimized_results['num_steps']} steps")
        
        # Compare results
        if baseline_results.get('success'):
            logger.info("Comparing results...")
            comparison = optimizer.compare_results()
            
            logger.info("="*70)
            logger.info("RESULTS SUMMARY (Optimize for low power)")
            logger.info("="*70)
            for metric, improvement in comparison['improvements'].items():
                logger.info(f"{metric}: {improvement:+.2f}%")
            logger.info("="*70)
        
        # Generate report
        report_path = optimizer.generate_report()
        logger.info(f"Report saved to: {report_path}")
        
        # Generate visualizations
        optimizer.visualize_results()
        logger.info("Visualizations saved to: results/figures/")
        
        logger.info("Training pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
