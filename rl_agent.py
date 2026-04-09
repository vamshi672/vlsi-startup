"""
RL Agent for PPAForge AI
Implements Proximal Policy Optimization (PPO) agent using RLlib for chip placement.

The agent learns to place cells optimally by maximizing a reward function that
balances wirelength, timing, power, and area objectives.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import ModelConfigDict
from gymnasium import spaces

from src.core.gnn_encoder import GNNEncoderFactory

logger = logging.getLogger(__name__)


class GNNPolicyNetwork(TorchModelV2, nn.Module):
    """
    Custom policy network that uses GNN encoder for chip placement.
    
    Architecture:
    1. GNN Encoder: Processes circuit graph to get node embeddings
    2. Graph Pooling: Aggregates node embeddings to graph-level representation
    3. Actor Head: Outputs action distribution (placement decisions)
    4. Critic Head: Outputs value estimate
    """
    
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_kwargs
    ):
        """
        Args:
            obs_space: Observation space
            action_space: Action space
            num_outputs: Number of action outputs
            model_config: Model configuration
            name: Model name
            **custom_model_kwargs: Custom configuration (includes our config)
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.config = custom_model_kwargs.get('config', {})
        self.gnn_config = self.config.get('gnn', {})
        
        # Extract dimensions
        self.hidden_dim = self.gnn_config.get('hidden_dims', [256, 512, 512, 256])[-1]
        self.action_dim = num_outputs
        
        # GNN Encoder
        self.gnn_encoder = GNNEncoderFactory.create_encoder(self.config)
        
        # Graph-level pooling
        self.pool_type = custom_model_kwargs.get('pool_type', 'mean')  # mean, max, attention
        
        if self.pool_type == 'attention':
            self.attention_weights = nn.Linear(self.hidden_dim, 1)
        
        # Actor head (policy network)
        self.actor_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.action_dim)
        )
        
        # Critic head (value network)
        self.critic_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # Value estimate cache
        self._value = None
        
        logger.info(f"GNN Policy Network initialized with {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        state: List[torch.Tensor],
        seq_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the policy network.
        
        Args:
            input_dict: Dictionary containing observations
            state: RNN state (not used for GNN)
            seq_lens: Sequence lengths (not used)
        
        Returns:
            Action logits and updated state
        """
        obs = input_dict["obs"]
        
        # Extract graph components from observation
        # Observation format: {node_features, edge_index, batch, ...]
        node_features = obs['node_features']
        edge_index = obs['edge_index'].long()
        batch = obs.get('batch', None)
        
        # GNN encoding
        node_embeddings = self.gnn_encoder(node_features, edge_index)
        
        # Graph-level pooling
        graph_embedding = self._pool_graph(node_embeddings, batch)
        
        # Actor: Compute action logits
        action_logits = self.actor_head(graph_embedding)
        
        # Critic: Compute value estimate
        self._value = self.critic_head(graph_embedding).squeeze(-1)
        
        return action_logits, state
    
    def value_function(self) -> torch.Tensor:
        """Return the value function estimate."""
        assert self._value is not None, "Must call forward() before value_function()"
        return self._value
    
    def _pool_graph(
        self,
        node_embeddings: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool node embeddings to graph-level representation.
        
        Args:
            node_embeddings: Node embeddings (num_nodes, hidden_dim)
            batch: Batch assignment for multiple graphs
        
        Returns:
            Graph embedding (batch_size, hidden_dim)
        """
        if batch is None:
            # Single graph
            if self.pool_type == 'mean':
                return node_embeddings.mean(dim=0, keepdim=True)
            elif self.pool_type == 'max':
                return node_embeddings.max(dim=0, keepdim=True)[0]
            elif self.pool_type == 'attention':
                attention_scores = torch.softmax(
                    self.attention_weights(node_embeddings),
                    dim=0
                )
                return (node_embeddings * attention_scores).sum(dim=0, keepdim=True)
        else:
            # Batched graphs
            batch_size = batch.max().item() + 1
            pooled = []
            
            for i in range(batch_size):
                mask = (batch == i)
                graph_nodes = node_embeddings[mask]
                
                if self.pool_type == 'mean':
                    pooled.append(graph_nodes.mean(dim=0))
                elif self.pool_type == 'max':
                    pooled.append(graph_nodes.max(dim=0)[0])
                elif self.pool_type == 'attention':
                    attention_scores = torch.softmax(
                        self.attention_weights(graph_nodes),
                        dim=0
                    )
                    pooled.append((graph_nodes * attention_scores).sum(dim=0))
            
            return torch.stack(pooled)


class PlacementPPOAgent:
    """
    PPO agent for chip placement optimization.
    
    This class wraps RLlib's PPO algorithm with custom configurations
    for chip placement tasks.
    """
    
    def __init__(self, config: Dict, env_config: Dict):
        """
        Args:
            config: Main configuration dictionary
            env_config: Environment configuration
        """
        self.config = config
        self.env_config = env_config
        self.rl_config = config.get('rl', {})
        self.ppo_config = self.rl_config.get('ppo', {})
        self.training_config = self.rl_config.get('training', {})
        
        # Register custom model
        ModelCatalog.register_custom_model("gnn_policy", GNNPolicyNetwork)
        
        # Build PPO configuration
        self.ppo_algo_config = self._build_ppo_config()
        
        # Initialize algorithm
        self.algo = None
        
        logger.info("Placement PPO Agent initialized")
    
    def _build_ppo_config(self) -> PPOConfig:
        """Build RLlib PPO configuration."""
        config = (
            PPOConfig()
            .environment(
                env="PlacementEnv",
                env_config=self.env_config
            )
            .framework("torch")
            .rollouts(
                num_rollout_workers=self.training_config.get('num_workers', 8),
                num_envs_per_worker=self.training_config.get('num_envs_per_worker', 1),
                rollout_fragment_length=200
            )
            .training(
                lr=self.ppo_config.get('lr', 3e-4),
                gamma=self.ppo_config.get('gamma', 0.99),
                lambda_=self.ppo_config.get('lambda_', 0.95),
                clip_param=self.ppo_config.get('clip_param', 0.2),
                vf_clip_param=self.ppo_config.get('vf_clip_param', 10.0),
                entropy_coeff=self.ppo_config.get('entropy_coeff', 0.01),
                kl_target=self.ppo_config.get('kl_target', 0.01),
                num_sgd_iter=self.ppo_config.get('num_sgd_iter', 10),
                sgd_minibatch_size=self.ppo_config.get('sgd_minibatch_size', 256),
                train_batch_size=self.ppo_config.get('train_batch_size', 4096),
                model={
                    "custom_model": "gnn_policy",
                    "custom_model_config": {
                        "config": self.config,
                        "pool_type": "attention"
                    }
                }
            )
            .resources(
                num_gpus=1 if torch.cuda.is_available() else 0,
                num_cpus_per_worker=1
            )
            .evaluation(
                evaluation_interval=self.training_config.get('evaluation_interval', 10),
                evaluation_duration=self.training_config.get('evaluation_num_episodes', 5),
                evaluation_config={
                    "explore": False
                }
            )
            .debugging(
                log_level="INFO"
            )
        )
        
        return config
    
    def train(self, num_iterations: int, checkpoint_dir: str = "./checkpoints") -> Dict:
        """
        Train the agent.
        
        Args:
            num_iterations: Number of training iterations
            checkpoint_dir: Directory to save checkpoints
        
        Returns:
            Training results dictionary
        """
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Build algorithm if not already built
        if self.algo is None:
            self.algo = self.ppo_algo_config.build()
        
        results_history = []
        best_reward = float('-inf')
        
        logger.info(f"Starting training for {num_iterations} iterations")
        
        for i in range(num_iterations):
            # Train one iteration
            result = self.algo.train()
            
            # Extract key metrics
            episode_reward_mean = result['episode_reward_mean']
            episode_len_mean = result['episode_len_mean']
            
            results_history.append(result)
            
            # Logging
            if i % 10 == 0:
                logger.info(
                    f"Iteration {i}/{num_iterations}: "
                    f"Reward={episode_reward_mean:.2f}, "
                    f"Length={episode_len_mean:.1f}"
                )
            
            # Checkpointing
            checkpoint_freq = self.training_config.get('checkpoint_freq', 50)
            if i % checkpoint_freq == 0 or i == num_iterations - 1:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{i}")
                self.algo.save(checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                best_path = os.path.join(checkpoint_dir, "best_model")
                self.algo.save(best_path)
                logger.info(f"Best model updated: {best_reward:.2f}")
        
        logger.info("Training completed!")
        
        return {
            'results_history': results_history,
            'best_reward': best_reward,
            'final_checkpoint': checkpoint_path
        }
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load agent from checkpoint."""
        if self.algo is None:
            self.algo = self.ppo_algo_config.build()
        
        self.algo.restore(checkpoint_path)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def compute_action(
        self,
        observation: Dict,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Compute action for given observation.
        
        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy
        
        Returns:
            Action array
        """
        if self.algo is None:
            raise RuntimeError("Agent not trained or loaded")
        
        action = self.algo.compute_single_action(
            observation,
            explore=not deterministic
        )
        
        return action
    
    def evaluate(
        self,
        eval_env,
        num_episodes: int = 10
    ) -> Dict:
        """
        Evaluate agent on environment.
        
        Args:
            eval_env: Evaluation environment
            num_episodes: Number of episodes to evaluate
        
        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs, info = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action = self.compute_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            logger.info(
                f"Eval Episode {episode + 1}/{num_episodes}: "
                f"Reward={episode_reward:.2f}, Length={episode_length}"
            )
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }


def test_ppo_agent():
    """Test PPO agent initialization."""
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    env_config = {
        'design_path': 'data/benchmarks/ibex',
        'max_steps': 500
    }
    
    # Create agent
    agent = PlacementPPOAgent(config, env_config)
    
    print("PPO Agent created successfully!")
    print(f"Training config: {agent.training_config}")
    print(f"PPO hyperparameters: {agent.ppo_config}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ppo_agent()
