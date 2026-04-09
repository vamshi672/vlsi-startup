"""
PPA Evaluator for PPAForge AI
Evaluates Power, Performance, and Area metrics for chip placements.

This module computes detailed PPA metrics using OpenROAD analysis
and provides reward signals for the RL agent.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PPAMetrics:
    """Container for PPA metrics."""
    # Power metrics (Watts)
    total_power: float
    dynamic_power: float
    leakage_power: float
    
    # Performance metrics (nanoseconds)
    worst_slack: float
    total_negative_slack: float
    clock_period: float
    max_frequency: float
    
    # Area metrics (microns^2)
    core_area: float
    cell_area: float
    utilization: float
    
    # Wirelength (microns)
    total_wirelength: float
    steiner_wirelength: float
    
    # Congestion
    max_congestion: float
    avg_congestion: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'power': self.total_power,
            'dynamic_power': self.dynamic_power,
            'leakage_power': self.leakage_power,
            'worst_slack': self.worst_slack,
            'total_negative_slack': self.total_negative_slack,
            'clock_period': self.clock_period,
            'max_frequency': self.max_frequency,
            'core_area': self.core_area,
            'cell_area': self.cell_area,
            'utilization': self.utilization,
            'wirelength': self.total_wirelength,
            'steiner_wirelength': self.steiner_wirelength,
            'max_congestion': self.max_congestion,
            'avg_congestion': self.avg_congestion
        }
    
    def __repr__(self) -> str:
        return (
            f"PPAMetrics(\n"
            f"  Power: {self.total_power*1e3:.2f}mW "
            f"(Dynamic: {self.dynamic_power*1e3:.2f}mW, "
            f"Leakage: {self.leakage_power*1e6:.2f}uW)\n"
            f"  Timing: WNS={self.worst_slack:.3f}ns, TNS={self.total_negative_slack:.3f}ns\n"
            f"  Area: {self.core_area:.2f}um^2 (Util: {self.utilization*100:.1f}%)\n"
            f"  Wirelength: {self.total_wirelength:.2f}um\n"
            f")"
        )


class PPAEvaluator:
    """
    Evaluates PPA metrics for chip placements.
    
    This class computes detailed metrics and provides normalized
    scores for optimization.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.design_config = config.get('design', {})
        self.optimization_config = config.get('optimization', {})
        
        # Normalization factors (updated during runtime)
        self.norm_factors = {
            'power': 1.0,
            'timing': 1.0,
            'wirelength': 1.0,
            'area': 1.0
        }
        
        logger.info("PPA Evaluator initialized")
    
    def evaluate(self, placement: Dict) -> Dict:
        """
        Evaluate PPA metrics for a given placement.
        
        Args:
            placement: Dictionary mapping cell names to positions
        
        Returns:
            Dictionary of PPA metrics
        """
        if not placement:
            # Return default metrics for empty placement
            return self._get_default_metrics()
        
        # Compute each metric category
        power_metrics = self._compute_power_metrics(placement)
        timing_metrics = self._compute_timing_metrics(placement)
        area_metrics = self._compute_area_metrics(placement)
        wirelength_metrics = self._compute_wirelength_metrics(placement)
        congestion_metrics = self._compute_congestion_metrics(placement)
        
        # Combine into PPAMetrics object
        metrics = PPAMetrics(
            total_power=power_metrics['total'],
            dynamic_power=power_metrics['dynamic'],
            leakage_power=power_metrics['leakage'],
            worst_slack=timing_metrics['wns'],
            total_negative_slack=timing_metrics['tns'],
            clock_period=timing_metrics['clock_period'],
            max_frequency=timing_metrics['max_freq'],
            core_area=area_metrics['core_area'],
            cell_area=area_metrics['cell_area'],
            utilization=area_metrics['utilization'],
            total_wirelength=wirelength_metrics['total'],
            steiner_wirelength=wirelength_metrics['steiner'],
            max_congestion=congestion_metrics['max'],
            avg_congestion=congestion_metrics['avg']
        )
        
        return metrics.to_dict()
    
    def _compute_power_metrics(self, placement: Dict) -> Dict:
        """
        Compute power consumption metrics.
        
        Power model (simplified):
        - Dynamic power: P_dynamic = α * C * V^2 * f
        - Leakage power: P_leakage = V * I_leakage
        - Total power = Dynamic + Leakage
        
        In real implementation, this would use OpenROAD power analysis.
        
        Args:
            placement: Cell placements
        
        Returns:
            Power metrics dictionary
        """
        # Simplified power estimation
        # In production, this would call OpenROAD's report_power
        
        num_cells = len(placement)
        
        # Estimate based on cell count and wirelength
        # Longer wires -> higher capacitance -> higher dynamic power
        wirelength = self._compute_total_wirelength(placement)
        
        # Base power per cell (example values for Sky130)
        dynamic_power_per_cell = 1e-6  # 1 uW per cell
        leakage_power_per_cell = 1e-8  # 10 nW per cell
        
        # Wire capacitance effect
        wire_capacitance_factor = 1e-9  # Capacitance per micron
        wire_power = wirelength * wire_capacitance_factor * 1.8**2 * 1e9  # V^2 * f
        
        dynamic_power = num_cells * dynamic_power_per_cell + wire_power
        leakage_power = num_cells * leakage_power_per_cell
        total_power = dynamic_power + leakage_power
        
        # Optimize for low power: shorter wires reduce dynamic power
        logger.debug(f"Power: Total={total_power*1e3:.2f}mW (Dynamic={dynamic_power*1e3:.2f}mW)")
        
        return {
            'total': total_power,
            'dynamic': dynamic_power,
            'leakage': leakage_power
        }
    
    def _compute_timing_metrics(self, placement: Dict) -> Dict:
        """
        Compute timing metrics.
        
        Timing model (simplified):
        - Wire delay ∝ RC (wirelength * wire_resistance * wire_capacitance)
        - Cell delay from library characterization
        - Critical path delay = Σ(cell_delays + wire_delays)
        
        In real implementation, this would use OpenROAD static timing analysis.
        
        Args:
            placement: Cell placements
        
        Returns:
            Timing metrics dictionary
        """
        # Simplified timing estimation
        # In production, this would call OpenROAD's report_checks
        
        target_clock_period = self.design_config.get('clock_period', 10.0)  # ns
        
        # Estimate critical path delay based on wirelength
        wirelength = self._compute_total_wirelength(placement)
        
        # Wire delay model: delay = R * C * length
        # Typical Sky130 values: ~0.1 ps/um
        wire_delay_per_um = 0.0001  # ns/um
        total_wire_delay = wirelength * wire_delay_per_um
        
        # Assume fixed logic delay
        num_cells = len(placement)
        avg_logic_delay = 0.05  # ns per cell
        logic_delay = num_cells * avg_logic_delay * 0.1  # Only ~10% on critical path
        
        # Critical path delay
        critical_path_delay = logic_delay + total_wire_delay
        
        # Slack = Target - Actual
        worst_slack = target_clock_period - critical_path_delay
        
        # Total negative slack (simplified)
        tns = min(0, worst_slack * num_cells * 0.1)
        
        max_freq = 1.0 / (critical_path_delay * 1e-9) if critical_path_delay > 0 else 0.0
        
        logger.debug(f"Timing: WNS={worst_slack:.3f}ns, Path Delay={critical_path_delay:.3f}ns")
        
        return {
            'wns': worst_slack,
            'tns': tns,
            'clock_period': target_clock_period,
            'max_freq': max_freq
        }
    
    def _compute_area_metrics(self, placement: Dict) -> Dict:
        """
        Compute area metrics.
        
        Args:
            placement: Cell placements
        
        Returns:
            Area metrics dictionary
        """
        # Core area (from design config or computed from placement)
        core_width = 1000.0  # microns (example)
        core_height = 1000.0
        core_area = core_width * core_height
        
        # Cell area (sum of all cell areas)
        # In real implementation, would get from LEF library
        num_cells = len(placement)
        avg_cell_area = 5.0  # um^2 (example for Sky130 standard cells)
        cell_area = num_cells * avg_cell_area
        
        # Utilization
        utilization = cell_area / core_area if core_area > 0 else 0.0
        
        return {
            'core_area': core_area,
            'cell_area': cell_area,
            'utilization': utilization
        }
    
    def _compute_wirelength_metrics(self, placement: Dict) -> Dict:
        """
        Compute wirelength metrics.
        
        Args:
            placement: Cell placements
        
        Returns:
            Wirelength metrics dictionary
        """
        total_wl = self._compute_total_wirelength(placement)
        steiner_wl = total_wl * 0.9  # Steiner is typically ~90% of HPWL
        
        return {
            'total': total_wl,
            'steiner': steiner_wl
        }
    
    def _compute_total_wirelength(self, placement: Dict) -> float:
        """
        Compute total Half-Perimeter Wirelength (HPWL).
        
        This is a simplified version. Real implementation would:
        1. Parse netlist to get net connectivity
        2. Compute HPWL for each net
        3. Sum across all nets
        
        Args:
            placement: Cell placements
        
        Returns:
            Total HPWL in microns
        """
        if not placement:
            return 0.0
        
        # Simplified: estimate based on average distance between cells
        positions = np.array([[p['x'], p['y']] for p in placement.values()])
        
        if len(positions) < 2:
            return 0.0
        
        # Compute average pairwise distance as proxy for HPWL
        # In real implementation, would compute actual net HPWL
        mean_pos = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - mean_pos, axis=1)
        
        # Estimate total wirelength
        num_cells = len(placement)
        avg_fanout = 3.0  # Average fanout per cell
        total_wl = np.sum(distances) * avg_fanout
        
        return total_wl
    
    def _compute_congestion_metrics(self, placement: Dict) -> Dict:
        """
        Compute routing congestion metrics.
        
        Args:
            placement: Cell placements
        
        Returns:
            Congestion metrics dictionary
        """
        # Simplified congestion estimation
        # Real implementation would use OpenROAD's global routing
        
        # Estimate based on local cell density
        max_congestion = 0.0
        avg_congestion = 0.0
        
        if len(placement) > 0:
            # Simplified: higher density areas have higher congestion
            positions = np.array([[p['x'], p['y']] for p in placement.values()])
            
            # Compute local densities using grid
            grid_size = 100.0  # Grid cell size in microns
            
            # Simple congestion estimate: cells per grid cell
            # More sophisticated version would consider routing tracks
            avg_congestion = len(placement) / 100.0  # Normalized
            max_congestion = min(1.0, avg_congestion * 1.5)
        
        return {
            'max': max_congestion,
            'avg': avg_congestion
        }
    
    def _get_default_metrics(self) -> Dict:
        """Get default metrics for empty placement."""
        return PPAMetrics(
            total_power=0.0,
            dynamic_power=0.0,
            leakage_power=0.0,
            worst_slack=0.0,
            total_negative_slack=0.0,
            clock_period=self.design_config.get('clock_period', 10.0),
            max_frequency=0.0,
            core_area=0.0,
            cell_area=0.0,
            utilization=0.0,
            total_wirelength=0.0,
            steiner_wirelength=0.0,
            max_congestion=0.0,
            avg_congestion=0.0
        ).to_dict()
    
    def compute_normalized_score(
        self,
        metrics: Dict,
        baseline_metrics: Dict
    ) -> float:
        """
        Compute normalized score for placement quality.
        
        Lower score is better (minimization objective).
        
        Args:
            metrics: Current metrics
            baseline_metrics: Baseline metrics for normalization
        
        Returns:
            Normalized score
        """
        # Get optimization weights
        weights = self.config.get('reward', {}).get('weights', {})
        
        # Normalize each metric relative to baseline
        power_score = (metrics['power'] / baseline_metrics['power']) if baseline_metrics['power'] > 0 else 1.0
        
        # For timing, negative slack is bad
        timing_score = (abs(metrics['worst_slack']) / abs(baseline_metrics['worst_slack'])) if baseline_metrics['worst_slack'] != 0 else 1.0
        
        wirelength_score = (metrics['wirelength'] / baseline_metrics['wirelength']) if baseline_metrics['wirelength'] > 0 else 1.0
        
        area_score = (metrics['core_area'] / baseline_metrics['core_area']) if baseline_metrics['core_area'] > 0 else 1.0
        
        # Weighted sum (Optimize for low power as primary objective)
        score = (
            abs(weights.get('power', -0.25)) * power_score +
            abs(weights.get('timing', -0.35)) * timing_score +
            abs(weights.get('wirelength', -0.30)) * wirelength_score +
            abs(weights.get('area', -0.10)) * area_score
        )
        
        return score
    
    def compare_metrics(
        self,
        metrics_a: Dict,
        metrics_b: Dict
    ) -> Dict:
        """
        Compare two sets of metrics.
        
        Args:
            metrics_a: First metrics (e.g., optimized)
            metrics_b: Second metrics (e.g., baseline)
        
        Returns:
            Dictionary of improvements (positive = better)
        """
        improvements = {}
        
        # Power improvement (reduction is good)
        improvements['power'] = (metrics_b['power'] - metrics_a['power']) / metrics_b['power'] * 100
        
        # Timing improvement (more positive slack is good)
        improvements['timing'] = (metrics_a['worst_slack'] - metrics_b['worst_slack']) / abs(metrics_b['worst_slack']) * 100 if metrics_b['worst_slack'] != 0 else 0.0
        
        # Wirelength improvement (reduction is good)
        improvements['wirelength'] = (metrics_b['wirelength'] - metrics_a['wirelength']) / metrics_b['wirelength'] * 100
        
        # Area (no change expected usually)
        improvements['area'] = (metrics_b['core_area'] - metrics_a['core_area']) / metrics_b['core_area'] * 100
        
        return improvements


def test_ppa_evaluator():
    """Test PPA evaluator."""
    import yaml
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    evaluator = PPAEvaluator(config)
    
    # Create dummy placement
    placement = {
        f'cell_{i}': {'x': i * 10.0, 'y': i * 5.0, 'rotation': 0}
        for i in range(100)
    }
    
    # Evaluate
    metrics = evaluator.evaluate(placement)
    
    print("PPA Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ppa_evaluator()
