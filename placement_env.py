"""
Placement Environment for PPAForge AI
Gymnasium environment for chip placement optimization using OpenROAD.

The environment models chip placement as an RL task where the agent
iteratively places cells to optimize PPA (Power, Performance, Area).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Dict, Tuple, Optional, Any
import logging
import copy

from src.core.graph_builder import GraphBuilder
from src.integration.openroad_interface import OpenROADInterface
from src.integration.ppa_evaluator import PPAEvaluator

logger = logging.getLogger(__name__)


class PlacementEnv(gym.Env):
    """
    Gymnasium environment for chip placement.
    
    State Space:
        - Circuit graph (nodes: cells, edges: nets)
        - Current placement coordinates
        - PPA metrics
    
    Action Space:
        - Continuous: (dx, dy, scale) for macro placement
        - Or discrete: Grid-based cell positioning
    
    Reward:
        - Weighted combination of: -wirelength, -timing, -power, -area
        - Optimize for low power as primary objective
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary containing:
                - design_path: Path to design files
                - openroad_config: OpenROAD settings
                - reward_weights: Reward function weights
                - max_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.config = config
        self.design_path = config.get('design_path')
        self.max_steps = config.get('max_steps', 500)
        
        # Initialize components
        self.graph_builder = GraphBuilder(config)
        self.openroad = OpenROADInterface(config)
        self.ppa_evaluator = PPAEvaluator(config)
        
        # Load design and build graph
        self._load_design()
        
        # State tracking
        self.current_step = 0
        self.current_placement = None
        self.placement_history = []
        self.reward_history = []
        
        # Baseline metrics (from initial/default placement)
        self.baseline_metrics = None
        
        # Define observation and action spaces
        self._setup_spaces()
        
        logger.info(f"Placement Environment initialized for {self.design_path}")
    
    def _load_design(self):
        """Load design and create initial graph representation."""
        import os
        
        # Find design files
        def_file = os.path.join(self.design_path, 'floorplan.def')
        verilog_file = os.path.join(self.design_path, 'design.v')
        
        if os.path.exists(def_file):
            self.graph = self.graph_builder.build_graph(def_file, format='def')
        elif os.path.exists(verilog_file):
            self.graph = self.graph_builder.build_graph(verilog_file, format='verilog')
        else:
            raise FileNotFoundError(f"No design files found in {self.design_path}")
        
        self.num_cells = self.graph.num_nodes
        self.cell_names = self.graph.cell_names
        
        logger.info(f"Loaded design with {self.num_cells} cells")
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Observation space: Graph representation
        # We use a dict space to accommodate graph structure
        self.observation_space = spaces.Dict({
            'node_features': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_cells, self.graph.x.shape[1]),
                dtype=np.float32
            ),
            'edge_index': spaces.Box(
                low=0,
                high=self.num_cells - 1,
                shape=(2, self.graph.edge_index.shape[1]),
                dtype=np.int64
            ),
            'ppa_metrics': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),  # [power, timing, wirelength, area]
                dtype=np.float32
            )
        })
        
        # Action space: Placement adjustments
        # Continuous action for macro/cell repositioning
        # Action: [cell_idx, dx, dy, rotation]
        self.action_space = spaces.Box(
            low=np.array([0, -1, -1, 0]),
            high=np.array([self.num_cells - 1, 1, 1, 3]),
            shape=(4,),
            dtype=np.float32
        )
        
        logger.info(f"Observation space: {self.observation_space}")
        logger.info(f"Action space: {self.action_space}")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset environment to initial state.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = 0
        self.placement_history = []
        self.reward_history = []
        
        # Initialize placement (use default or random)
        if options and 'initial_placement' in options:
            self.current_placement = options['initial_placement']
        else:
            # Use OpenROAD default placement as starting point
            self.current_placement = self.openroad.get_default_placement()
        
        # Update graph with initial placement
        self._update_graph_with_placement(self.current_placement)
        
        # Compute baseline metrics if not already computed
        if self.baseline_metrics is None:
            self.baseline_metrics = self.ppa_evaluator.evaluate(self.current_placement)
            logger.info(f"Baseline metrics: {self.baseline_metrics}")
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take [cell_idx, dx, dy, rotation]
        
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        self.current_step += 1
        
        # Parse action
        cell_idx = int(action[0])
        dx = action[1]
        dy = action[2]
        rotation = int(action[3])
        
        # Apply action to placement
        new_placement = self._apply_action(
            self.current_placement,
            cell_idx,
            dx,
            dy,
            rotation
        )
        
        # Check placement validity
        is_valid, violation_info = self._check_placement_validity(new_placement)
        
        if is_valid:
            # Accept placement
            self.current_placement = new_placement
            self._update_graph_with_placement(new_placement)
            
            # Evaluate PPA metrics
            current_metrics = self.ppa_evaluator.evaluate(new_placement)
            
            # Compute reward (Optimize for low power)
            reward = self._compute_reward(current_metrics, violation_info)
            
            self.placement_history.append(new_placement)
            self.reward_history.append(reward)
        else:
            # Reject invalid placement, apply penalty
            reward = self._compute_penalty(violation_info)
            current_metrics = self.ppa_evaluator.evaluate(self.current_placement)
        
        # Get observation
        observation = self._get_observation()
        
        # Check termination conditions
        terminated = self._is_terminated(current_metrics)
        truncated = self.current_step >= self.max_steps
        
        # Info
        info = self._get_info()
        info['current_metrics'] = current_metrics
        info['is_valid_placement'] = is_valid
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(
        self,
        placement: Dict,
        cell_idx: int,
        dx: float,
        dy: float,
        rotation: int
    ) -> Dict:
        """
        Apply action to create new placement.
        
        Args:
            placement: Current placement
            cell_idx: Index of cell to move
            dx: X displacement (normalized -1 to 1)
            dy: Y displacement (normalized -1 to 1)
            rotation: Rotation (0: N, 1: E, 2: S, 3: W)
        
        Returns:
            New placement
        """
        new_placement = copy.deepcopy(placement)
        
        # Clip cell index
        cell_idx = np.clip(cell_idx, 0, self.num_cells - 1)
        cell_name = self.cell_names[cell_idx]
        
        if cell_name not in new_placement:
            return new_placement
        
        # Get current position
        current_x, current_y = new_placement[cell_name]['x'], new_placement[cell_name]['y']
        
        # Scale displacement (e.g., max 10% of die size per step)
        die_width, die_height = self.graph.die_area
        max_dx = die_width * 0.1
        max_dy = die_height * 0.1
        
        # Apply displacement
        new_x = current_x + dx * max_dx
        new_y = current_y + dy * max_dy
        
        # Clip to die area
        new_x = np.clip(new_x, 0, die_width)
        new_y = np.clip(new_y, 0, die_height)
        
        # Update placement
        new_placement[cell_name]['x'] = new_x
        new_placement[cell_name]['y'] = new_y
        new_placement[cell_name]['rotation'] = rotation
        
        return new_placement
    
    def _check_placement_validity(
        self,
        placement: Dict
    ) -> Tuple[bool, Dict]:
        """
        Check if placement is valid (no overlaps, within bounds).
        
        Returns:
            is_valid: Whether placement is valid
            violation_info: Information about violations
        """
        violations = {
            'overlap': False,
            'out_of_bounds': False,
            'density_violation': False
        }
        
        # Check overlaps (simplified - real implementation would use detailed geometry)
        cell_positions = []
        for cell_name, pos in placement.items():
            cell = self.graph_builder.cells[cell_name]
            x, y = pos['x'], pos['y']
            w, h = cell.width, cell.height
            cell_positions.append((x, y, w, h))
        
        # Simplified overlap check
        for i, (x1, y1, w1, h1) in enumerate(cell_positions):
            # Check bounds
            die_w, die_h = self.graph.die_area
            if x1 < 0 or y1 < 0 or x1 + w1 > die_w or y1 + h1 > die_h:
                violations['out_of_bounds'] = True
            
            # Check overlaps with other cells
            for j, (x2, y2, w2, h2) in enumerate(cell_positions):
                if i >= j:
                    continue
                
                # Rectangle overlap check
                if not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1):
                    violations['overlap'] = True
                    break
        
        is_valid = not any(violations.values())
        
        return is_valid, violations
    
    def _compute_reward(
        self,
        metrics: Dict,
        violation_info: Dict
    ) -> float:
        """
        Compute reward based on PPA metrics.
        
        Optimize for low power as primary objective.
        
        Args:
            metrics: Current PPA metrics
            violation_info: Placement violations
        
        Returns:
            Reward value
        """
        reward_config = self.config.get('reward', {})
        weights = reward_config.get('weights', {})
        normalization = reward_config.get('normalization', {})
        penalties = reward_config.get('penalties', {})
        
        # Normalize metrics relative to baseline
        norm_power = (metrics['power'] - self.baseline_metrics['power']) / self.baseline_metrics['power']
        norm_timing = (metrics['worst_slack'] - self.baseline_metrics['worst_slack']) / abs(self.baseline_metrics['worst_slack'] + 1e-9)
        norm_wirelength = (metrics['wirelength'] - self.baseline_metrics['wirelength']) / self.baseline_metrics['wirelength']
        norm_area = (metrics['area'] - self.baseline_metrics['area']) / self.baseline_metrics['area']
        
        # Compute weighted reward (negative because we want to minimize)
        # Optimize for low power - this is emphasized in the reward weights
        reward = (
            weights.get('power', -0.25) * norm_power +  # Optimize for low power
            weights.get('timing', -0.35) * norm_timing +
            weights.get('wirelength', -0.30) * norm_wirelength +
            weights.get('area', -0.10) * norm_area
        )
        
        # Add penalties for violations
        if violation_info['overlap']:
            reward += penalties.get('overlap_penalty', -10.0)
        if violation_info['out_of_bounds']:
            reward += penalties.get('out_of_bounds_penalty', -5.0)
        if violation_info['density_violation']:
            reward += penalties.get('density_violation_penalty', -3.0)
        
        return reward
    
    def _compute_penalty(self, violation_info: Dict) -> float:
        """Compute penalty for invalid placement."""
        penalties = self.config.get('reward', {}).get('penalties', {})
        
        penalty = 0.0
        if violation_info['overlap']:
            penalty += penalties.get('overlap_penalty', -10.0)
        if violation_info['out_of_bounds']:
            penalty += penalties.get('out_of_bounds_penalty', -5.0)
        if violation_info['density_violation']:
            penalty += penalties.get('density_violation_penalty', -3.0)
        
        return penalty
    
    def _update_graph_with_placement(self, placement: Dict):
        """Update graph node features with current placement."""
        # Update position features in graph
        for i, cell_name in enumerate(self.cell_names):
            if cell_name in placement:
                # Update x, y coordinates in node features
                # Assuming position features are at specific indices
                self.graph.x[i, -2] = placement[cell_name]['x']
                self.graph.x[i, -1] = placement[cell_name]['y']
    
    def _get_observation(self) -> Dict:
        """Get current observation."""
        # Evaluate current metrics
        current_metrics = self.ppa_evaluator.evaluate(self.current_placement)
        
        # Normalize PPA metrics
        ppa_array = np.array([
            current_metrics['power'] / self.baseline_metrics['power'],
            current_metrics['worst_slack'] / (abs(self.baseline_metrics['worst_slack']) + 1e-9),
            current_metrics['wirelength'] / self.baseline_metrics['wirelength'],
            current_metrics['area'] / self.baseline_metrics['area']
        ], dtype=np.float32)
        
        observation = {
            'node_features': self.graph.x.numpy(),
            'edge_index': self.graph.edge_index.numpy(),
            'ppa_metrics': ppa_array
        }
        
        return observation
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        return {
            'step': self.current_step,
            'num_cells': self.num_cells,
            'design': self.design_path
        }
    
    def _is_terminated(self, metrics: Dict) -> bool:
        """Check if episode should terminate (e.g., convergence)."""
        # Check if we've achieved optimization targets
        optimization = self.config.get('optimization', {})
        constraints = optimization.get('constraints', {})
        
        # Check power constraint (primary objective: Optimize for low power)
        power_improvement = (self.baseline_metrics['power'] - metrics['power']) / self.baseline_metrics['power']
        if power_improvement >= 0.20:  # 20% power reduction achieved
            logger.info(f"Power optimization target achieved: {power_improvement:.2%}")
            return True
        
        return False
    
    def render(self, mode='human'):
        """Render the environment (optional visualization)."""
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.max_steps}")
            if self.reward_history:
                print(f"Last Reward: {self.reward_history[-1]:.4f}")
        
        return None


def test_environment():
    """Test placement environment."""
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env_config = {
        **config,
        'design_path': 'data/benchmarks/gcd',
        'max_steps': 100
    }
    
    env = PlacementEnv(env_config)
    
    # Test reset
    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"Info: {info}")
    
    # Test random steps
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Reward={reward:.4f}, Terminated={terminated}")
        
        if terminated or truncated:
            break
    
    print("Environment test completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_environment()
