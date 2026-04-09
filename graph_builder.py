"""
Graph Builder for PPAForge AI
Converts netlists (Verilog/DEF) to PyTorch Geometric graph representations.

This module extracts circuit topology and creates rich graph representations
with node features (cells, macros) and edge features (nets, timing arcs).
"""

import os
import re
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Cell:
    """Represents a cell (standard cell or macro) in the design."""
    name: str
    cell_type: str  # Type from library
    width: float
    height: float
    x: float = 0.0
    y: float = 0.0
    is_macro: bool = False
    is_fixed: bool = False
    pins: List[str] = None
    
    def __post_init__(self):
        if self.pins is None:
            self.pins = []
    
    @property
    def area(self):
        return self.width * self.height
    
    @property
    def center(self):
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class Net:
    """Represents a net connecting multiple pins."""
    name: str
    pins: List[Tuple[str, str]]  # (cell_name, pin_name)
    weight: float = 1.0
    is_critical: bool = False
    slack: float = float('inf')
    
    @property
    def degree(self):
        return len(self.pins)
    
    def get_hpwl(self, cells: Dict[str, Cell]) -> float:
        """Calculate Half-Perimeter Wirelength."""
        if len(self.pins) < 2:
            return 0.0
        
        x_coords = []
        y_coords = []
        
        for cell_name, _ in self.pins:
            if cell_name in cells:
                cx, cy = cells[cell_name].center
                x_coords.append(cx)
                y_coords.append(cy)
        
        if not x_coords:
            return 0.0
        
        hpwl = (max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords))
        return hpwl


class GraphBuilder:
    """
    Builds PyTorch Geometric graphs from circuit netlists.
    
    Supports multiple graph representations:
    - Homogeneous: All nodes are cells, edges are nets
    - Heterogeneous: Separate node types for cells, macros, IOs
    - Hypergraph: Nets as hyperedges (converted to clique or star)
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary with graph parameters
        """
        self.config = config
        self.gnn_config = config.get('gnn', {})
        self.graph_params = self.gnn_config.get('graph_params', {})
        
        self.cells: Dict[str, Cell] = {}
        self.nets: Dict[str, Net] = {}
        self.cell_to_idx: Dict[str, int] = {}
        self.net_to_idx: Dict[str, int] = {}
        
        # Design parameters
        self.die_area = None
        self.core_area = None
        
        logger.info("GraphBuilder initialized")
    
    def parse_def_file(self, def_path: str) -> None:
        """
        Parse DEF (Design Exchange Format) file.
        
        Args:
            def_path: Path to .def file
        """
        logger.info(f"Parsing DEF file: {def_path}")
        
        with open(def_path, 'r') as f:
            content = f.read()
        
        # Parse die area
        die_match = re.search(r'DIEAREA\s*\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)', content)
        if die_match:
            x1, y1, x2, y2 = map(int, die_match.groups())
            self.die_area = (x2 - x1, y2 - y1)
            logger.info(f"Die area: {self.die_area}")
        
        # Parse components (cells)
        components_section = re.search(r'COMPONENTS\s+(\d+)\s*;(.*?)END COMPONENTS', content, re.DOTALL)
        if components_section:
            num_components = int(components_section.group(1))
            components_text = components_section.group(2)
            
            # Pattern: - cell_name cell_type + PLACED ( x y ) orientation ;
            component_pattern = r'-\s+(\S+)\s+(\S+)\s+.*?PLACED\s*\(\s*(\d+)\s+(\d+)\s*\)\s*(\S+)'
            
            for match in re.finditer(component_pattern, components_text):
                cell_name = match.group(1)
                cell_type = match.group(2)
                x = int(match.group(3))
                y = int(match.group(4))
                
                # Determine if macro (simplified heuristic)
                is_macro = 'MACRO' in cell_type.upper() or cell_type.startswith('RAM') or cell_type.startswith('ROM')
                
                # Placeholder dimensions (will be updated from LEF)
                width = 100.0 if is_macro else 1.0
                height = 100.0 if is_macro else 1.0
                
                self.cells[cell_name] = Cell(
                    name=cell_name,
                    cell_type=cell_type,
                    width=width,
                    height=height,
                    x=x,
                    y=y,
                    is_macro=is_macro,
                    is_fixed=False
                )
            
            logger.info(f"Parsed {len(self.cells)} components")
        
        # Parse nets
        nets_section = re.search(r'NETS\s+(\d+)\s*;(.*?)END NETS', content, re.DOTALL)
        if nets_section:
            num_nets = int(nets_section.group(1))
            nets_text = nets_section.group(2)
            
            # Pattern: - net_name ( pin_connections ) ;
            net_pattern = r'-\s+(\S+)\s+(.*?)\s*;'
            
            for match in re.finditer(net_pattern, nets_text):
                net_name = match.group(1)
                connections_text = match.group(2)
                
                # Parse pin connections: ( cell_name pin_name )
                pin_pattern = r'\(\s*(\S+)\s+(\S+)\s*\)'
                pins = [(m.group(1), m.group(2)) for m in re.finditer(pin_pattern, connections_text)]
                
                self.nets[net_name] = Net(
                    name=net_name,
                    pins=pins,
                    weight=1.0
                )
            
            logger.info(f"Parsed {len(self.nets)} nets")
    
    def parse_verilog_netlist(self, verilog_path: str) -> None:
        """
        Parse Verilog netlist (gate-level).
        
        Args:
            verilog_path: Path to .v file
        """
        logger.info(f"Parsing Verilog netlist: {verilog_path}")
        
        with open(verilog_path, 'r') as f:
            content = f.read()
        
        # Remove comments
        content = re.sub(r'//.*?\n', '\n', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Parse module instances
        # Pattern: cell_type instance_name ( .pin(net), ... );
        instance_pattern = r'(\w+)\s+(\w+)\s*\((.*?)\)\s*;'
        
        for match in re.finditer(instance_pattern, content, re.DOTALL):
            cell_type = match.group(1)
            instance_name = match.group(2)
            connections = match.group(3)
            
            # Skip primitive gates in detailed parsing
            if cell_type.upper() in ['INPUT', 'OUTPUT', 'WIRE', 'REG']:
                continue
            
            if instance_name not in self.cells:
                self.cells[instance_name] = Cell(
                    name=instance_name,
                    cell_type=cell_type,
                    width=1.0,
                    height=1.0,
                    is_macro=False
                )
            
            # Parse connections: .pin_name(net_name)
            conn_pattern = r'\.(\w+)\s*\(\s*(\w+)\s*\)'
            for conn_match in re.finditer(conn_pattern, connections):
                pin_name = conn_match.group(1)
                net_name = conn_match.group(2)
                
                if net_name not in self.nets:
                    self.nets[net_name] = Net(name=net_name, pins=[])
                
                self.nets[net_name].pins.append((instance_name, pin_name))
                self.cells[instance_name].pins.append(pin_name)
        
        logger.info(f"Verilog parsing complete: {len(self.cells)} cells, {len(self.nets)} nets")
    
    def add_timing_info(self, timing_report_path: str) -> None:
        """
        Add timing information from OpenROAD timing report.
        
        Args:
            timing_report_path: Path to timing report file
        """
        if not os.path.exists(timing_report_path):
            logger.warning(f"Timing report not found: {timing_report_path}")
            return
        
        with open(timing_report_path, 'r') as f:
            content = f.read()
        
        # Parse critical paths and slacks
        # This is simplified - actual parsing depends on report format
        slack_pattern = r'slack\s+(\S+):\s*([-\d.]+)'
        
        for match in re.finditer(slack_pattern, content):
            net_name = match.group(1)
            slack = float(match.group(2))
            
            if net_name in self.nets:
                self.nets[net_name].slack = slack
                self.nets[net_name].is_critical = (slack < 0)
        
        logger.info("Timing information added")
    
    def compute_node_features(self) -> torch.Tensor:
        """
        Compute node feature matrix.
        
        Returns:
            Tensor of shape (num_nodes, num_features)
        """
        feature_list = []
        self.cell_to_idx = {name: idx for idx, name in enumerate(self.cells.keys())}
        
        # Feature configuration
        node_feature_names = self.gnn_config.get('node_features', [
            'cell_type', 'cell_area', 'cell_width', 'cell_height',
            'pin_count', 'fanin', 'fanout', 'x_coord', 'y_coord'
        ])
        
        # Compute fanin/fanout
        fanin = defaultdict(int)
        fanout = defaultdict(int)
        
        for net in self.nets.values():
            if len(net.pins) > 0:
                # First pin is typically output, rest are inputs (simplified)
                source_cell = net.pins[0][0] if net.pins[0][0] in self.cells else None
                if source_cell:
                    fanout[source_cell] += len(net.pins) - 1
                
                for cell_name, _ in net.pins[1:]:
                    if cell_name in self.cells:
                        fanin[cell_name] += 1
        
        # Normalize coordinates
        max_x = max((cell.x for cell in self.cells.values()), default=1.0)
        max_y = max((cell.y for cell in self.cells.values()), default=1.0)
        
        for cell_name, cell in self.cells.items():
            features = []
            
            # Cell type (one-hot encoding simplified to binary)
            if 'cell_type' in node_feature_names:
                features.append(1.0 if cell.is_macro else 0.0)
            
            # Geometric features
            if 'cell_area' in node_feature_names:
                features.append(np.log1p(cell.area))
            if 'cell_width' in node_feature_names:
                features.append(cell.width)
            if 'cell_height' in node_feature_names:
                features.append(cell.height)
            
            # Connectivity features
            if 'pin_count' in node_feature_names:
                features.append(len(cell.pins))
            if 'fanin' in node_feature_names:
                features.append(fanin[cell_name])
            if 'fanout' in node_feature_names:
                features.append(fanout[cell_name])
            
            # Position features (normalized)
            if 'x_coord' in node_feature_names:
                features.append(cell.x / max_x if max_x > 0 else 0.0)
            if 'y_coord' in node_feature_names:
                features.append(cell.y / max_y if max_y > 0 else 0.0)
            
            feature_list.append(features)
        
        node_features = torch.tensor(feature_list, dtype=torch.float)
        logger.info(f"Node features shape: {node_features.shape}")
        
        return node_features
    
    def compute_edge_index(self, use_hypergraph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute edge connectivity (COO format).
        
        Args:
            use_hypergraph: If True, handle hyperedges (nets with >2 pins)
        
        Returns:
            edge_index: Tensor of shape (2, num_edges)
            edge_attr: Tensor of shape (num_edges, num_edge_features)
        """
        edges = []
        edge_features = []
        
        for net in self.nets.values():
            if len(net.pins) < 2:
                continue
            
            # Get cell indices
            cell_indices = []
            for cell_name, _ in net.pins:
                if cell_name in self.cell_to_idx:
                    cell_indices.append(self.cell_to_idx[cell_name])
            
            if len(cell_indices) < 2:
                continue
            
            # Edge features
            edge_feat = [
                net.weight,
                net.slack if net.slack != float('inf') else 0.0,
                float(len(cell_indices)),  # net degree
                float(net.is_critical)
            ]
            
            if use_hypergraph and len(cell_indices) > 2:
                # Convert hyperedge to clique (all-to-all connections)
                for i in range(len(cell_indices)):
                    for j in range(i + 1, len(cell_indices)):
                        edges.append([cell_indices[i], cell_indices[j]])
                        edges.append([cell_indices[j], cell_indices[i]])  # Undirected
                        edge_features.append(edge_feat)
                        edge_features.append(edge_feat)
            else:
                # Two-pin net or simplified representation
                for i in range(len(cell_indices) - 1):
                    edges.append([cell_indices[i], cell_indices[i + 1]])
                    edges.append([cell_indices[i + 1], cell_indices[i]])  # Undirected
                    edge_features.append(edge_feat)
                    edge_features.append(edge_feat)
        
        if not edges:
            # Empty graph
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 4), dtype=torch.float)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        logger.info(f"Edge index shape: {edge_index.shape}, Edge features shape: {edge_attr.shape}")
        
        return edge_index, edge_attr
    
    def build_graph(self, design_path: str, format: str = 'def') -> Data:
        """
        Build PyTorch Geometric graph from design files.
        
        Args:
            design_path: Path to design file (.def or .v)
            format: File format ('def' or 'verilog')
        
        Returns:
            PyTorch Geometric Data object
        """
        logger.info(f"Building graph from {design_path} (format: {format})")
        
        # Parse design files
        if format == 'def':
            self.parse_def_file(design_path)
        elif format == 'verilog':
            self.parse_verilog_netlist(design_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Compute features
        node_features = self.compute_node_features()
        edge_index, edge_attr = self.compute_edge_index(
            use_hypergraph=self.graph_params.get('use_hypergraph', True)
        )
        
        # Create PyG Data object
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(self.cells)
        )
        
        # Add metadata
        graph.cell_names = list(self.cells.keys())
        graph.net_names = list(self.nets.keys())
        graph.die_area = self.die_area
        
        logger.info(f"Graph built successfully: {graph}")
        
        return graph
    
    def save_graph(self, graph: Data, output_path: str) -> None:
        """Save graph to disk."""
        torch.save(graph, output_path)
        logger.info(f"Graph saved to {output_path}")
    
    def load_graph(self, graph_path: str) -> Data:
        """Load graph from disk."""
        graph = torch.load(graph_path)
        logger.info(f"Graph loaded from {graph_path}")
        return graph


def main():
    """Example usage."""
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Build graph from DEF file
    builder = GraphBuilder(config)
    
    # Example: Parse a design
    design_path = "data/benchmarks/ibex/floorplan.def"
    if os.path.exists(design_path):
        graph = builder.build_graph(design_path, format='def')
        
        # Save processed graph
        builder.save_graph(graph, "data/processed/ibex_graph.pt")
        
        print(f"Graph statistics:")
        print(f"  Nodes: {graph.num_nodes}")
        print(f"  Edges: {graph.edge_index.shape[1]}")
        print(f"  Node features: {graph.x.shape[1]}")
        print(f"  Edge features: {graph.edge_attr.shape[1]}")
    else:
        print(f"Design file not found: {design_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
