"""
GNN Encoder for PPAForge AI
Implements Graph Neural Network architectures for learning chip placement representations.

Supports: GCN, GraphSAGE, GAT, and Hierarchical GraphSAGE with residual connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder with configurable depth and aggregation.
    
    GraphSAGE is particularly suitable for large graphs (chip designs)
    as it samples neighbors rather than using full neighborhood.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        dropout: float = 0.1,
        aggregation: str = 'mean',
        normalize: bool = True,
        residual: bool = True
    ):
        """
        Args:
            in_channels: Input feature dimension
            hidden_channels: List of hidden layer dimensions
            out_channels: Output embedding dimension
            dropout: Dropout probability
            aggregation: Aggregation method ('mean', 'max', 'lstm')
            normalize: Whether to normalize embeddings
            residual: Whether to use residual connections
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.normalize = normalize
        self.residual = residual
        
        # Build layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        dims = [in_channels] + hidden_channels + [out_channels]
        
        for i in range(len(dims) - 1):
            self.convs.append(
                SAGEConv(
                    dims[i],
                    dims[i + 1],
                    aggr=aggregation,
                    normalize=normalize
                )
            )
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))
        
        # Residual projections (if dimensions don't match)
        self.residual_projections = nn.ModuleList()
        if residual:
            for i in range(len(dims) - 1):
                if dims[i] != dims[i + 1]:
                    self.residual_projections.append(
                        nn.Linear(dims[i], dims[i + 1])
                    )
                else:
                    self.residual_projections.append(nn.Identity())
        
        logger.info(f"GraphSAGE Encoder initialized: {dims}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge connectivity (2, num_edges)
        
        Returns:
            Node embeddings (num_nodes, out_channels)
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_input = x
            
            # Graph convolution
            x = conv(x, edge_index)
            x = bn(x)
            
            # Residual connection
            if self.residual and i < len(self.residual_projections):
                x_res = self.residual_projections[i](x_input)
                x = x + x_res
            
            # Activation and dropout (except last layer)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class HierarchicalGraphSAGE(nn.Module):
    """
    Hierarchical GraphSAGE for multi-scale chip placement learning.
    
    This architecture captures both local (cell-level) and global (block-level)
    information, which is crucial for effective placement optimization.
    
    Architecture:
    1. Local encoder: Captures fine-grained cell interactions
    2. Pooling: Aggregates cells into coarser blocks
    3. Global encoder: Captures block-level interactions
    4. Unpooling: Refines embeddings back to cell level
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        aggregation: str = 'mean',
        pool_ratio: float = 0.5
    ):
        """
        Args:
            in_channels: Input feature dimension
            hidden_dims: List of hidden dimensions [local, global, refined]
            dropout: Dropout probability
            aggregation: Aggregation method
            pool_ratio: Ratio for graph pooling
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        
        # Local encoder (fine-grained)
        self.local_encoder = GraphSAGEEncoder(
            in_channels=in_channels,
            hidden_channels=[hidden_dims[0], hidden_dims[1]],
            out_channels=hidden_dims[1],
            dropout=dropout,
            aggregation=aggregation,
            residual=True
        )
        
        # Global encoder (coarse-grained)
        self.global_encoder = GraphSAGEEncoder(
            in_channels=hidden_dims[1],
            hidden_channels=[hidden_dims[2]],
            out_channels=hidden_dims[2],
            dropout=dropout,
            aggregation=aggregation,
            residual=True
        )
        
        # Refinement encoder
        self.refine_encoder = GraphSAGEEncoder(
            in_channels=hidden_dims[1] + hidden_dims[2],  # Concatenated
            hidden_channels=[hidden_dims[3]],
            out_channels=hidden_dims[3],
            dropout=dropout,
            aggregation=aggregation,
            residual=True
        )
        
        # Simple pooling using learnable cluster assignment
        self.pool_ratio = pool_ratio
        self.cluster_mlp = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] // 2, 1)
        )
        
        logger.info(f"Hierarchical GraphSAGE initialized with dims: {hidden_dims}")
    
    def pool_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pool graph to coarser representation.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment for multiple graphs
        
        Returns:
            pooled_x: Pooled node features
            pooled_edge_index: Pooled edge connectivity
            cluster_assignment: Cluster assignment for unpooling
        """
        num_nodes = x.size(0)
        
        # Compute cluster scores
        scores = self.cluster_mlp(x).squeeze(-1)
        
        # Top-k selection for pooling
        num_pooled_nodes = max(int(num_nodes * self.pool_ratio), 1)
        _, top_indices = torch.topk(scores, num_pooled_nodes)
        top_indices = top_indices.sort()[0]
        
        # Create pooled features
        pooled_x = x[top_indices]
        
        # Create mapping for edges
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        node_mask[top_indices] = True
        
        # Create new node indices
        new_node_indices = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        new_node_indices[top_indices] = torch.arange(num_pooled_nodes, device=x.device)
        
        # Filter edges (keep edges between pooled nodes)
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        pooled_edge_index = edge_index[:, edge_mask]
        pooled_edge_index = new_node_indices[pooled_edge_index]
        
        # Cluster assignment for unpooling
        cluster_assignment = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        cluster_assignment[top_indices] = torch.arange(num_pooled_nodes, device=x.device)
        
        return pooled_x, pooled_edge_index, cluster_assignment
    
    def unpool_graph(
        self,
        pooled_x: torch.Tensor,
        cluster_assignment: torch.Tensor,
        original_num_nodes: int
    ) -> torch.Tensor:
        """
        Unpool features back to original graph size.
        
        Args:
            pooled_x: Pooled node features
            cluster_assignment: Cluster assignment from pooling
            original_num_nodes: Original number of nodes
        
        Returns:
            Unpooled features
        """
        unpooled_x = torch.zeros(
            original_num_nodes,
            pooled_x.size(1),
            device=pooled_x.device
        )
        
        # Broadcast pooled features to original nodes
        unpooled_x = pooled_x[cluster_assignment]
        
        return unpooled_x
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Hierarchical forward pass.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment
        
        Returns:
            Refined node embeddings
        """
        original_num_nodes = x.size(0)
        
        # Local encoding (fine-grained)
        local_embed = self.local_encoder(x, edge_index)
        
        # Pool to coarser graph
        pooled_x, pooled_edge_index, cluster_assignment = self.pool_graph(
            local_embed, edge_index, batch
        )
        
        # Global encoding (coarse-grained)
        global_embed = self.global_encoder(pooled_x, pooled_edge_index)
        
        # Unpool back to original size
        global_embed_unpooled = self.unpool_graph(
            global_embed, cluster_assignment, original_num_nodes
        )
        
        # Concatenate local and global information
        combined = torch.cat([local_embed, global_embed_unpooled], dim=-1)
        
        # Refinement encoding
        refined_embed = self.refine_encoder(combined, edge_index)
        
        return refined_embed


class GCNEncoder(nn.Module):
    """Simple GCN encoder for baseline comparison."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_dims: List[int],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        dims = [in_channels] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
        
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoder(nn.Module):
    """Graph Attention Network encoder."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_dims: List[int],
        heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        dims = [in_channels] + hidden_dims
        
        for i in range(len(dims) - 1):
            out_channels = dims[i + 1]
            self.convs.append(
                GATConv(
                    dims[i] * (heads if i > 0 else 1),
                    out_channels,
                    heads=heads if i < len(dims) - 2 else 1,
                    dropout=dropout
                )
            )
        
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GNNEncoderFactory:
    """Factory for creating GNN encoders based on configuration."""
    
    @staticmethod
    def create_encoder(config: dict) -> nn.Module:
        """
        Create GNN encoder from configuration.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            GNN encoder module
        """
        gnn_config = config.get('gnn', {})
        architecture = gnn_config.get('architecture', 'graphsage')
        hidden_dims = gnn_config.get('hidden_dims', [256, 512, 512, 256])
        dropout = gnn_config.get('dropout', 0.1)
        aggregation = gnn_config.get('aggregation', 'mean')
        
        # Determine input dimension from node features
        node_features = gnn_config.get('node_features', [])
        in_channels = len(node_features)
        
        logger.info(f"Creating {architecture} encoder with {in_channels} input features")
        
        if architecture == 'graphsage':
            return GraphSAGEEncoder(
                in_channels=in_channels,
                hidden_channels=hidden_dims[:-1],
                out_channels=hidden_dims[-1],
                dropout=dropout,
                aggregation=aggregation,
                residual=True
            )
        
        elif architecture == 'hierarchical_graphsage':
            return HierarchicalGraphSAGE(
                in_channels=in_channels,
                hidden_dims=hidden_dims,
                dropout=dropout,
                aggregation=aggregation
            )
        
        elif architecture == 'gcn':
            return GCNEncoder(
                in_channels=in_channels,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        
        elif architecture == 'gat':
            return GATEncoder(
                in_channels=in_channels,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        
        else:
            raise ValueError(f"Unknown architecture: {architecture}")


def test_encoder():
    """Test GNN encoder with synthetic data."""
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create synthetic graph
    num_nodes = 100
    num_features = len(config['gnn']['node_features'])
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 500))
    
    # Create encoder
    encoder = GNNEncoderFactory.create_encoder(config)
    
    # Forward pass
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(x, edge_index)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in encoder.parameters())}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_encoder()
