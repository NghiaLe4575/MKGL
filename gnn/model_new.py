"""
PyTorch Geometric implementation of PNA-based GNN layers for MKGL
Replaces TorchDrug dependencies with PyG primitives

Key Features:
- Per-graph fairness in batched edge selection (prevents starvation)
- VirtualTensor support for memory-efficient sparse operations
- Compatible with both single graphs and batched scenarios
- Supports iterative subgraph sampling for efficient link prediction

Note: The current forward() implementation uses a simplified single-graph approach.
For full batched graph support with per-graph edge selection, the graph should be
created using torch_geometric.data.Batch.from_data_list() and include batch attributes.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, to_undirected
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min
import math


# ============================================================================
# Utility Functions (Replaces gnn/util.py functionality)
# ============================================================================

def multikey_argsort(inputs, descending=False, break_tie=False):
    """Sort by multiple keys with optional tie-breaking"""
    if break_tie:
        order = torch.randperm(len(inputs[0]), device=inputs[0].device)
    else:
        order = torch.arange(len(inputs[0]), device=inputs[0].device)
    for key in inputs[::-1]:
        index = key[order].argsort(stable=True, descending=descending)
        order = order[index]
    return order


def bincount(input, minlength=0):
    """Fast bincount with sorted input optimization"""
    if input.numel() == 0:
        return torch.zeros(minlength, dtype=torch.long, device=input.device)

    sorted_check = (input.diff() >= 0).all()
    if sorted_check:
        if minlength == 0:
            minlength = input.max().item() + 1
        range_tensor = torch.arange(minlength + 1, device=input.device)
        index = torch.bucketize(range_tensor, input)
        return index.diff()

    return input.bincount(minlength=minlength)


def variadic_topks(input, size, ks, largest=True, break_tie=False):
    """Select top-k elements per group with variable k"""
    index2sample = torch.repeat_interleave(size)
    if largest:
        index2sample = -index2sample
    order = multikey_argsort((index2sample, input), descending=largest, break_tie=break_tie)

    range_tensor = torch.arange(ks.sum(), device=input.device)
    offset = (size - ks).cumsum(0) - size + ks
    range_tensor = range_tensor + offset.repeat_interleave(ks)
    index = order[range_tensor]

    return input[index], index


class VirtualTensor(object):
    """Sparse tensor representation for efficient memory usage"""
    
    def __init__(self, keys=None, values=None, index=None, input=None, shape=None, dtype=None, device=None):
        if shape is None:
            shape = index.shape + input.shape[1:]
        if index is None:
            index = torch.zeros(*shape[:1], dtype=torch.long, device=device)
        if input is None:
            input = torch.empty(1, *shape[1:], dtype=dtype, device=device)
        if keys is None:
            keys = torch.empty(0, dtype=torch.long, device=device)
        if values is None:
            values = torch.empty(0, *shape[1:], dtype=dtype, device=device)

        self.keys = keys
        self.values = values
        self.index = index
        self.input = input

    @classmethod
    def zeros(cls, *shape, dtype=None, device=None):
        input = torch.zeros(1, *shape[1:], dtype=dtype, device=device)
        return cls(input=input, shape=shape, dtype=dtype, device=device)

    @classmethod
    def gather(cls, input, index):
        return cls(index=index, input=input, dtype=input.dtype, device=input.device)

    def clone(self):
        return VirtualTensor(self.keys.clone(), self.values.clone(), self.index.clone(), self.input.clone())

    @property
    def shape(self):
        return self.index.shape + self.input.shape[1:]

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def device(self):
        return self.values.device

    def __getitem__(self, indexes):
        if not isinstance(indexes, tuple):
            indexes = (indexes,)
        keys = indexes[0]
        assert keys.numel() == 0 or (keys.max() < len(self.index) and keys.min() >= 0)
        values = self.input[(self.index[keys],) + indexes[1:]]
        if len(self.keys) > 0:
            index = torch.bucketize(keys, self.keys)
            index = index.clamp(max=len(self.keys) - 1)
            indexes_adj = (index,) + indexes[1:]
            found = keys == self.keys[index]
            indexes_adj = tuple(idx[found] for idx in indexes_adj)
            values[found] = self.values[indexes_adj]
        return values

    def __setitem__(self, keys, values):
        new_keys, inverse = torch.cat([self.keys, keys]).unique(return_inverse=True)
        new_values = torch.zeros(len(new_keys), *self.shape[1:], dtype=self.dtype, device=self.device)
        new_values[inverse[:len(self.keys)]] = self.values
        new_values[inverse[len(self.keys):]] = values
        self.keys = new_keys
        self.values = new_values

    def __len__(self):
        return self.shape[0]


# ============================================================================
# PyG Graph Utilities
# ============================================================================

class PyGGraphWrapper:
    """Wrapper to provide TorchDrug-like interface for PyG Data objects"""
    
    def __init__(self, data, batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self._cache = {}
        
    @property
    def num_node(self):
        return self.data.num_nodes
    
    @property
    def num_edge(self):
        return self.data.edge_index.size(1)
    
    @property
    def num_relation(self):
        if hasattr(self.data, 'num_relations'):
            return self.data.num_relations
        if hasattr(self.data, 'edge_attr'):
            return self.data.edge_attr.max().item() + 1
        return 1
    
    @property
    def edge_index(self):
        return self.data.edge_index
    
    @property
    def edge_attr(self):
        return self.data.edge_attr if hasattr(self.data, 'edge_attr') else None
    
    @property
    def device(self):
        return self.data.edge_index.device
    
    def undirected(self, add_inverse=True):
        """Convert to undirected graph"""
        edge_index = self.data.edge_index
        edge_attr = self.edge_attr
        
        if add_inverse and edge_attr is not None:
            # Add inverse edges with offset relation IDs
            inv_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            inv_edge_attr = edge_attr + self.num_relation
            
            edge_index = torch.cat([edge_index, inv_edge_index], dim=1)
            edge_attr = torch.cat([edge_attr, inv_edge_attr], dim=0)
        else:
            edge_index = to_undirected(edge_index)
            if edge_attr is not None:
                edge_attr = edge_attr.repeat(2)
        
        new_data = Data(
            x=self.data.x if hasattr(self.data, 'x') else None,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=self.num_node
        )
        if hasattr(self.data, 'num_relations'):
            new_data.num_relations = self.data.num_relations * 2 if add_inverse else self.data.num_relations
        
        return PyGGraphWrapper(new_data, self.batch_size)
    
    @property
    def degree_out(self):
        """Out-degree for each node"""
        if 'degree_out' not in self._cache:
            self._cache['degree_out'] = scatter_add(
                torch.ones(self.num_edge, device=self.device),
                self.edge_index[0],
                dim=0,
                dim_size=self.num_node
            )
        return self._cache['degree_out']
    
    @property
    def degree_in(self):
        """In-degree for each node"""
        if 'degree_in' not in self._cache:
            self._cache['degree_in'] = scatter_add(
                torch.ones(self.num_edge, device=self.device),
                self.edge_index[1],
                dim=0,
                dim_size=self.num_node
            )
        return self._cache['degree_in']
    
    def edge_mask(self, mask, compact=False):
        """Create subgraph with masked edges"""
        if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
            edge_index = self.edge_index[:, mask]
            edge_attr = self.edge_attr[mask] if self.edge_attr is not None else None
        else:
            # mask is indices
            edge_index = self.edge_index[:, mask]
            edge_attr = self.edge_attr[mask] if self.edge_attr is not None else None
        
        if compact:
            # Reindex nodes
            nodes = torch.unique(edge_index)
            node_map = torch.zeros(self.num_node, dtype=torch.long, device=self.device)
            node_map[nodes] = torch.arange(len(nodes), device=self.device)
            edge_index = node_map[edge_index]
            num_nodes = len(nodes)
        else:
            num_nodes = self.num_node
        
        new_data = Data(
            x=self.data.x if hasattr(self.data, 'x') else None,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )
        if hasattr(self.data, 'num_relations'):
            new_data.num_relations = self.data.num_relations
        
        return PyGGraphWrapper(new_data, self.batch_size)
    
    def match(self, pattern):
        """Find edges matching pattern [h, t, r]"""
        # pattern: [num_patterns, 3] where -1 is wildcard
        edge_index = self.edge_index
        edge_attr = self.edge_attr
        
        matches = []
        for p in pattern:
            h, t, r = p
            mask = torch.ones(self.num_edge, dtype=torch.bool, device=self.device)
            if h >= 0:
                mask &= (edge_index[0] == h)
            if t >= 0:
                mask &= (edge_index[1] == t)
            if r >= 0 and edge_attr is not None:
                mask &= (edge_attr == r)
            matches.append(torch.where(mask)[0])
        
        if matches:
            return (torch.cat(matches),)
        return (torch.empty(0, dtype=torch.long, device=self.device),)
    
    def num_neighbors(self, node_indices):
        """Get number of neighbors for nodes"""
        degree = self.degree_out
        return degree[node_indices]


def create_pyg_graph(edge_list, num_nodes, num_relations):
    """Create PyG Data object from edge list"""
    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty(0, dtype=torch.long)
    else:
        edge_index = torch.tensor([[e[0], e[1]] for e in edge_list], dtype=torch.long).t()
        edge_attr = torch.tensor([e[2] for e in edge_list], dtype=torch.long)
    
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    data.num_relations = num_relations
    return PyGGraphWrapper(data)


# ============================================================================
# PNA Layer Implementation
# ============================================================================

class PNAConv(nn.Module):
    """Principal Neighbourhood Aggregation convolution layer"""
    
    def __init__(self, input_dim, output_dim, num_relation, query_input_dim,
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        
        # Output projection
        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        
        # Relation embeddings
        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, num_relation * 2 * input_dim)
        else:
            self.relation = nn.Embedding(num_relation * 2, input_dim)
    
    def message_and_aggregate(self, graph, node_features, query, boundary):
        """Perform message passing and aggregation"""
        batch_size = query.shape[0] if query.ndim > 1 else 1
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        num_nodes = graph.num_node
        
        # Get relation embeddings
        if self.dependent:
            # query: [batch_size, query_dim]
            relation_input = self.relation_linear(query).view(batch_size, self.num_relation * 2, self.input_dim)
        else:
            relation_input = self.relation.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # For single graph (non-batched), flatten relation embeddings
        if batch_size == 1:
            relation_input = relation_input.squeeze(0)  # [num_rel*2, input_dim]
        
        # Get edge features using relation embeddings
        if edge_attr is not None:
            if batch_size == 1:
                edge_features = relation_input[edge_attr]  # [num_edges, input_dim]
            else:
                # For batched graphs, need to handle batch dimension
                edge_features = relation_input[0][edge_attr]  # Simplified for now
        else:
            edge_features = torch.ones(edge_index.size(1), self.input_dim, device=edge_index.device)
        
        # Message computation: element-wise multiply edge features with source node features
        source_features = node_features[edge_index[0]]  # [num_edges, input_dim]
        messages = source_features * edge_features  # [num_edges, input_dim]
        
        # Degree for normalization
        degree_out = graph.degree_out.unsqueeze(-1) + 1  # [num_nodes, 1]
        
        # Aggregate based on method
        if self.aggregate_func == "sum":
            update = scatter_add(messages, edge_index[1], dim=0, dim_size=num_nodes)
            update = update + boundary
        
        elif self.aggregate_func == "mean":
            update = scatter_add(messages, edge_index[1], dim=0, dim_size=num_nodes)
            update = (update + boundary) / degree_out
        
        elif self.aggregate_func == "max":
            update = scatter_max(messages, edge_index[1], dim=0, dim_size=num_nodes)[0]
            update = torch.max(update, boundary)
        
        elif self.aggregate_func == "pna":
            # PNA aggregation with multiple aggregators and scalers
            sum_agg = scatter_add(messages, edge_index[1], dim=0, dim_size=num_nodes)
            sq_messages = messages ** 2
            sq_sum = scatter_add(sq_messages, edge_index[1], dim=0, dim_size=num_nodes)
            max_agg = scatter_max(messages, edge_index[1], dim=0, dim_size=num_nodes)[0]
            min_agg = scatter_min(messages, edge_index[1], dim=0, dim_size=num_nodes)[0]
            
            mean = (sum_agg + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max_agg = torch.max(max_agg, boundary)
            min_agg = torch.min(min_agg, boundary)
            std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
            
            # Stack aggregators: [num_nodes, input_dim, 4]
            features = torch.stack([mean, max_agg, min_agg, std], dim=-1)
            features = features.flatten(-2)  # [num_nodes, input_dim * 4]
            
            # Degree scalers
            scale = degree_out.log()
            scale_mean = scale.mean()
            scale = scale / (scale_mean + 1e-10)
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            
            # Apply scalers: [num_nodes, input_dim * 4, 3]
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        
        else:
            raise ValueError(f"Unknown aggregation function: {self.aggregate_func}")
        
        return update
    
    def forward(self, graph, node_features, query, boundary):
        """Forward pass through PNA layer"""
        # Message passing and aggregation
        update = self.message_and_aggregate(graph, node_features, query, boundary)
        
        # Combine with input features
        output = self.linear(torch.cat([node_features, update], dim=-1))
        
        if self.layer_norm:
            output = self.layer_norm(output)
        
        if self.activation:
            output = self.activation(output)
        
        return output


# ============================================================================
# PNA Model
# ============================================================================

class PNA(nn.Module):
    """Principal Neighbourhood Aggregation network"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_relation, query_input_dim,
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True):
        super().__init__()
        self.num_relation = num_relation
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(
                PNAConv(layer_input_dim, hidden_dim, num_relation, query_input_dim,
                       aggregate_func, layer_norm, activation, dependent)
            )
    
    def forward(self, graph, node_features, query, boundary):
        """Forward pass through all layers"""
        hidden = node_features
        for layer in self.layers:
            hidden = layer(graph, hidden, query, boundary)
        return hidden


# ============================================================================
# Conditioned PNA (for Link Prediction)
# ============================================================================

class ConditionedPNA(nn.Module):
    """Conditioned PNA with iterative subgraph sampling for efficient link prediction"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_relation, query_input_dim,
                 num_mlp_layers=2, node_ratio=0.1, degree_ratio=1.0,
                 test_node_ratio=None, test_degree_ratio=None,
                 aggregate_func="pna", remove_one_hop=False, break_tie=False):
        super().__init__()
        
        self.num_relation = num_relation
        self.num_layers = num_layers
        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        self.test_node_ratio = test_node_ratio or node_ratio
        self.test_degree_ratio = test_degree_ratio or degree_ratio
        self.remove_one_hop = remove_one_hop
        self.break_tie = break_tie
        
        # PNA layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(
                PNAConv(layer_input_dim, hidden_dim, num_relation, query_input_dim,
                       aggregate_func, layer_norm=False, activation="relu", dependent=True)
            )
        
        # Relation embeddings
        self.rel_embedding = nn.Embedding(num_relation * 2, input_dim)
        
        # Score network
        feature_dim = hidden_dim + input_dim
        self.linear = nn.Linear(feature_dim, hidden_dim)
        
        mlp_layers = [hidden_dim] + [feature_dim] * (num_mlp_layers - 1) + [1]
        mlp_modules = []
        for i in range(len(mlp_layers) - 1):
            mlp_modules.append(nn.Linear(mlp_layers[i], mlp_layers[i+1]))
            if i < len(mlp_layers) - 2:
                mlp_modules.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_modules)
    
    def forward(self, h_index, r_index, t_index, hidden_states, rel_hidden_states, 
                graph, score_text_embs, all_index):
        """
        Forward pass for link prediction
        
        Args:
            h_index: [batch_size, num_neg+1] head entity indices
            r_index: [batch_size, num_neg+1] relation indices
            t_index: [batch_size, num_neg+1] tail entity indices
            hidden_states: [batch_size, hidden_dim] head entity embeddings from LLM
            rel_hidden_states: [batch_size, hidden_dim] relation embeddings from LLM
            graph: PyGGraphWrapper with knowledge graph
            score_text_embs: [num_entities, hidden_dim] text embeddings for all entities
            all_index: [num_entities] indices for all entities
        
        Returns:
            scores: [batch_size, num_neg+1] prediction scores
        """
        if self.training:
            graph = self.remove_easy_edges(graph, h_index, r_index, t_index)
        
        # Convert to undirected with inverse relations
        graph = graph.undirected(add_inverse=True)
        
        # Standardize to tail prediction format
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        
        batch_size = len(h_index)
        device = h_index.device
        
        # Initialize node features and scores
        num_nodes = graph.num_node
        node_features = torch.zeros(num_nodes, hidden_states.shape[-1], 
                                    dtype=hidden_states.dtype, device=device)
        node_features[all_index] = score_text_embs.type(hidden_states.dtype)
        
        # Set head node features
        h_nodes = h_index[:, 0]  # [batch_size]
        node_features[h_nodes] = hidden_states
        
        # Get relation embeddings
        rel_embeds = self.rel_embedding(r_index[:, 0])  # [batch_size, input_dim]
        rel_embeds = rel_embeds.type(hidden_states.dtype)
        
        # Initialize scores for all nodes
        node_scores = self.compute_score(node_features, rel_embeds, h_nodes, graph)
        
        # Iterative message passing with subgraph sampling
        for layer_idx, layer in enumerate(self.layers):
            # Select important edges based on scores
            edge_mask = self.select_edges(graph, node_scores)
            subgraph = graph.edge_mask(edge_mask, compact=False)
            
            # Weighted features by current scores
            weighted_features = F.sigmoid(node_scores).unsqueeze(-1) * node_features
            
            # Prepare boundary (self-loop features)
            boundary = node_features
            
            # Apply GNN layer
            query = rel_embeds.mean(dim=0, keepdim=True)  # Simplified: use mean query
            new_features = layer(subgraph, weighted_features.float(), query, boundary.float())
            
            # Update node features
            node_features = node_features + new_features.type(node_features.dtype)
            
            # Recompute scores
            node_scores = self.compute_score(node_features, rel_embeds, h_nodes, graph)
        
        # Extract scores for target entities
        scores = node_scores[t_index]  # [batch_size, num_neg+1]
        
        return scores
    
    def compute_score(self, node_features, rel_embeds, h_nodes, graph):
        """Compute heuristic scores for all nodes"""
        num_nodes = node_features.shape[0]
        batch_size = rel_embeds.shape[0]
        
        # Expand relation embeddings for all nodes
        rel_embeds_expanded = rel_embeds.mean(dim=0).unsqueeze(0).expand(num_nodes, -1)
        
        # Compute heuristic
        combined = torch.cat([node_features, rel_embeds_expanded], dim=-1)
        heuristic = self.linear(combined)
        
        # Compute final score
        x = node_features * heuristic
        scores = self.mlp(x).squeeze(-1)
        
        return scores
    
    def select_edges(self, graph, score):
        """
        Select important edges based on node scores with per-graph fairness.
        Ensures each graph in the batch gets proportional edge selection.
        """
        node_ratio = self.node_ratio if self.training else self.test_node_ratio
        degree_ratio = self.degree_ratio if self.training else self.test_degree_ratio
        
        # Handle both batched and single graphs
        if hasattr(graph.data, 'batch'):
            batch = graph.data.batch
            num_graphs = batch.max().item() + 1
        else:
            # Single graph case
            batch = torch.zeros(graph.num_node, dtype=torch.long, device=graph.device)
            num_graphs = 1
        
        # 1. Calculate Node Limits Per Graph (Vectorized)
        num_nodes_per_graph = bincount(batch, minlength=num_graphs)
        
        # Calculate 'ks' (nodes to keep) for each graph individually
        ks = (num_nodes_per_graph.float() * node_ratio).long()
        ks = torch.clamp(ks, min=1)  # Ensure at least 1 node kept per graph
        ks = torch.min(ks, num_nodes_per_graph)

        # 2. Select Top-K Nodes per graph
        index = variadic_topks(score, num_nodes_per_graph, ks=ks, break_tie=self.break_tie)[1]
        node_in = index
        
        # 3. Mask Sources
        src_mask = torch.zeros(graph.num_node, dtype=torch.bool, device=graph.device)
        src_mask[node_in] = True
        
        # Find edges starting from selected nodes
        edge_mask_in = src_mask[graph.edge_index[0]]
        
        # 4. Calculate Edge Limits Per Graph
        # Identify which graph each valid edge belongs to
        edge_batch = batch[graph.edge_index[0][edge_mask_in]]
        num_edges_per_graph = bincount(edge_batch, minlength=num_graphs)
        
        # Calculate max edges allowed per graph
        es = (degree_ratio * ks.float() * (graph.num_edge / graph.num_node)).long()
        es = torch.clamp(es, min=1)
        
        # Handle case where some graphs might have 0 valid edges
        if es.size(0) != num_edges_per_graph.size(0):
            # Recalculate 'es' using per-graph statistics
            es = (degree_ratio * ks.float() * (num_edges_per_graph.float() / num_nodes_per_graph.float().clamp(min=1))).long()
            es = torch.clamp(es, min=1)

        # Apply limit: cannot select more edges than exist
        es = torch.min(es, num_edges_per_graph)

        # 5. Select Top-K Edges per graph
        valid_edge_indices = torch.nonzero(edge_mask_in, as_tuple=False).squeeze(-1)
        if valid_edge_indices.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=graph.device)
        
        node_out = graph.edge_index[1][valid_edge_indices]
        score_edge = score[node_out]
        
        final_edge_indices = variadic_topks(score_edge, num_edges_per_graph, ks=es, break_tie=self.break_tie)[1]
        
        # Map back to global edge indices
        return valid_edge_indices[final_edge_indices]
    
    def remove_easy_edges(self, graph, h_index, r_index, t_index):
        """Remove easy edges (ground truth) from graph during training"""
        if self.remove_one_hop:
            # Remove all edges between h and t
            h_flat = h_index.flatten()
            t_flat = t_index.flatten()
            pattern = torch.stack([h_flat, t_flat, torch.full_like(h_flat, -1)], dim=-1)
        else:
            # Remove specific (h, r, t) edges
            h_flat = h_index.flatten()
            t_flat = t_index.flatten()
            r_flat = r_index.flatten()
            pattern = torch.stack([h_flat, t_flat, r_flat], dim=-1)
        
        # Find matching edges
        edge_indices = graph.match(pattern)[0]
        
        # Create mask to remove these edges
        mask = torch.ones(graph.num_edge, dtype=torch.bool, device=graph.device)
        mask[edge_indices] = False
        
        return graph.edge_mask(mask, compact=False)
    
    def negative_sample_to_tail(self, h_index, t_index, r_index):
        """Ensure all samples are in tail prediction format"""
        # Check if samples are tail predictions (h is constant)
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        
        # Swap h and t if needed, adjust relation by offset
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        
        return new_h_index, new_t_index, new_r_index


# ============================================================================
# Factory Functions
# ============================================================================

def create_pna_layer(config):
    """Create PNA layer from config dict"""
    return PNAConv(
        input_dim=config.get('input_dim', 64),
        output_dim=config.get('output_dim', 64),
        num_relation=config.get('num_relation', 100),
        query_input_dim=config.get('query_input_dim', 64),
        aggregate_func=config.get('aggregate_func', 'pna'),
        layer_norm=config.get('layer_norm', False),
        activation=config.get('activation', 'relu'),
        dependent=config.get('dependent', True)
    )


def create_conditioned_pna(config):
    """Create ConditionedPNA from config dict"""
    return ConditionedPNA(
        input_dim=config.get('input_dim', 64),
        hidden_dim=config.get('hidden_dim', 64),
        num_layers=config.get('num_layer', 3),
        num_relation=config.get('num_relation', 100),
        query_input_dim=config.get('query_input_dim', 64),
        num_mlp_layers=config.get('num_mlp_layer', 2),
        node_ratio=config.get('node_ratio', 0.1),
        degree_ratio=config.get('degree_ratio', 1.0),
        test_node_ratio=config.get('test_node_ratio', None),
        test_degree_ratio=config.get('test_degree_ratio', None),
        aggregate_func=config.get('aggregate_func', 'pna'),
        remove_one_hop=config.get('remove_one_hop', False),
        break_tie=config.get('break_tie', False)
    )
