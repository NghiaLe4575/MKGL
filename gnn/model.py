import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import  scatter_add

from torchdrug import core, layers
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from .util import VirtualTensor, Range, RepeatGraph
from .util import bincount, variadic_topks
from .layer import *

def print_stat(name, tensor):
    if tensor is None:
        print(f"DEBUG: {name} is None")
        return
    
    # Handle VirtualTensor by materializing it
    if hasattr(tensor, '__class__') and 'VirtualTensor' in tensor.__class__.__name__:
        # VirtualTensor - materialize by accessing all elements
        try:
            indices = torch.arange(len(tensor), device=tensor.device)
            t = tensor[indices]
        except:
            # If that fails, try to access the underlying values/input directly
            if hasattr(tensor, 'values') and len(tensor.values) > 0:
                t = tensor.values
            elif hasattr(tensor, 'input'):
                t = tensor.input
            else:
                print(f"DEBUG: {name} | Cannot materialize VirtualTensor")
                return
    else:
        t = tensor
    
    # Ensure it's float for statistics
    if hasattr(t, 'float'):
        t = t.float() if t.dtype != torch.float32 else t
    
    print(f"DEBUG: {name} | Shape: {list(t.shape)} | Min: {t.min().item():.4f} | Max: {t.max().item():.4f} | Mean: {t.mean().item():.4f} | NaNs: {torch.isnan(t).sum().item()}")

@R.register("PNA")
class PNA(nn.Module, core.Configurable):

    def __init__(self, base_layer, num_layer, num_mlp_layer=2, remove_one_hop=False):
        super(PNA, self).__init__()

        self.num_relation = base_layer.num_relation
        self.remove_one_hop = remove_one_hop
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(core.Configurable.load_config_dict(base_layer.config_dict()))
        feature_dim = base_layer.output_dim + base_layer.input_dim
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

    def aggregate(self, graph, input_embeds):
        layer_input = input_embeds
        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut:
                hidden = hidden + layer_input
            layer_input = hidden
            
        return hidden

    def init_input_embeds(self, graph, input_embeds, input_index):
        input_embeds = torch.zeros(graph.num_node, input_embeds.shape[-1], device=input_embeds.device)
        input_embeds[input_index] = input_embeds
        return input_embeds


    def forward(self, graph, input_embeds, input_index):
        graph = graph.undirected(add_inverse=True)
        input_embeds = self.init_input_embeds(input_embeds, input_index)
        output = self.aggregate(graph, input_embeds)
        return output



@R.register("ConditionedPNA")
class ConditionedPNA(PNA, core.Configurable):

    def __init__(self, base_layer, num_layer, num_mlp_layer=2, node_ratio=0.1, degree_ratio=1, test_node_ratio=None, test_degree_ratio=None,
                 break_tie=False, **kwargs):
        
        super().__init__(base_layer, num_layer, num_mlp_layer=num_mlp_layer, **kwargs)

        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        self.test_node_ratio = test_node_ratio or node_ratio
        self.test_degree_ratio = test_degree_ratio or degree_ratio
        self.break_tie = break_tie

        feature_dim = base_layer.output_dim + base_layer.input_dim
        self.rel_embedding = nn.Embedding(base_layer.num_relation * 2, base_layer.input_dim)
        self.linear = nn.Linear(feature_dim, base_layer.output_dim)
        self.mlp = layers.MLP(base_layer.output_dim, [feature_dim] * (num_mlp_layer - 1) + [1])


    def forward(self, h_index, r_index, t_index, hidden_states, rel_hidden_states, graph, score_text_embs, all_index):
        print(f"DEBUG: START FORWARD | h_max={h_index.max()} t_max={t_index.max()} r_max={r_index.max()}")
        if r_index.max() >= self.num_relation * 2:
            print(f"CRASH PENDING: r_index {r_index.max()} >= limit {self.num_relation * 2}")
        
        if self.training:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)
        graph = graph.undirected(add_inverse=True)
        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index, t_index, r_index)
        
        batch_size = len(h_index)
        graph = RepeatGraph(graph, batch_size)
        offset = graph.num_cum_nodes - graph.num_nodes
        h_index = h_index + offset.unsqueeze(-1).to(h_index.device)
        t_index = t_index + offset.unsqueeze(-1).to(t_index.device)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        if r_index[:, 0].max() >= self.rel_embedding.num_embeddings:
            print(f"CRASH PENDING: Rel Embedding Index {r_index[:, 0].max()} >= {self.rel_embedding.num_embeddings}")

        rel_embeds = self.rel_embedding(r_index[:, 0]) 
        rel_embeds = rel_embeds.type(hidden_states.dtype) #+ rel_hidden_states
        
        # DEBUG: Check initial embeddings
        print_stat("Forward: Initial hidden_states", hidden_states)
        print_stat("Forward: Initial rel_embeds", rel_embeds)

        input_embeds, init_score = self.init_input_embeds(graph, hidden_states, h_index[:, 0], score_text_embs, all_index, rel_embeds)
        score = self.aggregate(graph, h_index[:, 0], r_index[:, 0], input_embeds, rel_embeds, init_score)
        score = score[t_index]
        return score

    def aggregate(self, graph, h_index, r_index, input_embeds, rel_embeds, init_score):
        
        query = rel_embeds
        boundary, score = input_embeds, init_score
        hidden = boundary.clone()
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary
            graph.hidden = hidden
            graph.score = score
            graph.node_id = Range(graph.num_node, device=h_index.device)
            graph.pna_degree_out = graph.degree_out
        with graph.edge():
            graph.edge_id = Range(graph.num_edge, device=h_index.device)
        pna_degree_mean = (graph[0].degree_out + 1).log().mean()
        
        print("\n--- START AGGREGATE ---")
        print_stat("Aggregate: Init Score", graph.score)
        
        for i, layer in enumerate(self.layers):
            print(f"\n--- LAYER {i} START ---")
            print_stat(f"Layer {i}: graph.score (Start of Loop)", graph.score)
            
            edge_index = self.select_edges(graph, graph.score)
            subgraph = graph.edge_mask(edge_index, compact=True)
            subgraph.pna_degree_mean = pna_degree_mean

            # --- INSERT THIS DEBUG BLOCK ---
            sub_edge_attr = getattr(subgraph, 'edge_attr', None)
            if sub_edge_attr is not None:
                max_val = sub_edge_attr.max().item()
                limit = self.num_relation * 2
                print(f"DEBUG: Layer {i} | Edge Attr Max: {max_val} | Limit: {limit}")
                
                if max_val >= limit:
                    print(f"!!! CRASH DETECTED !!!")
                    print(f"You have a Relation ID {max_val} but only configured {limit} slots.")
                    print(f"Your 'sub_edge_attr' logic is correct, but the DATA is out of bounds.")
                    # We exit explicitly to avoid the confusing CUDA error
                    import sys; sys.exit(1)
            # -------------------------------

            # Gating mechanism: check if sigmoid is saturating due to high score
            gate = F.sigmoid(subgraph.score).unsqueeze(-1)
            print_stat(f"Layer {i}: Gate (Sigmoid output)", gate)
            
            layer_input = gate * subgraph.hidden
            hidden = layer(subgraph, layer_input.type(torch.float32))
            out_mask = subgraph.degree_out > 0
            node_out = subgraph.node_id[out_mask]

            # Update Hidden
            prev_hidden = graph.hidden[node_out]
            update_delta = hidden[out_mask]
            
            # Check for explosion in hidden states (often causes score explosion next)
            if update_delta.abs().max() > 100:
                print(f"WARNING: Layer {i} hidden update delta is large!")
                print_stat(f"Layer {i}: Update Delta", update_delta)
            
            graph.hidden[node_out] = (prev_hidden + update_delta).type(graph.hidden[node_out].dtype)
            print_stat(f"Layer {i}: Updated Hidden (Subset)", graph.hidden[node_out])

            index = graph.node2graph[node_out]
            
            # Update Score
            print(f"DEBUG: Layer {i} | Calculating new scores...")
            new_scores = self.score(graph.hidden[node_out], query[index])
            
            # Track the new scores BEFORE they go back into the graph
            print_stat(f"Layer {i}: New Scores Calculated", new_scores)
            
            graph.score[node_out] = new_scores.type(graph.score[node_out].dtype)

            data_dict, meta_dict = subgraph.data_by_meta("graph")
            graph.meta_dict.update(meta_dict)
            graph.__dict__.update(data_dict)

        print("--- END AGGREGATE ---\n")
        return graph.score

    def init_input_embeds(self, graph, head_embeds, head_index, tail_embeds, tail_index,  rel_embeds):
        if tail_embeds.dtype != head_embeds.dtype:
            tail_embeds = tail_embeds.to(head_embeds.dtype)
        
        input_embeds = VirtualTensor.zeros(graph.num_node, rel_embeds.shape[1], device=rel_embeds.device, dtype=rel_embeds.dtype)
        
        input_embeds[tail_index] = tail_embeds.type(head_embeds.dtype)
        input_embeds[head_index] = head_embeds

        print("\nDEBUG: init_input_embeds calc start")
        zero_scores = self.score(torch.zeros_like(rel_embeds), rel_embeds)
        print_stat("init_input_embeds: Raw Score (Zero Embeds)", zero_scores)
        
        score = VirtualTensor.gather(zero_scores, graph.node2graph) # zero all
        
        score_head = self.score(head_embeds, rel_embeds)
        print_stat("init_input_embeds: Raw Score (Head Embeds)", score_head)
        
        score[head_index] = score_head
        
        # Check before clamp
        print_stat("init_input_embeds: Score All (Pre-Clamp)", score)
        
        # Apply clamping to prevent extreme values
        score = torch.clamp(score, min=-15, max=15)
        
        # Check after clamp
        print_stat("init_input_embeds: Score All (Post-Clamp)", score)
            
        return input_embeds, score

    def score(self, hidden, rel_embeds):
        heuristic = self.linear(torch.cat([hidden, rel_embeds], dim=-1))
        x = hidden * heuristic
        raw_score = self.mlp(x).squeeze(-1)
        if raw_score.abs().max() > 50 or torch.isnan(raw_score).any():
            print("  DEBUG: score() internal tracking:")
            print_stat("    score input: hidden", hidden)
            print_stat("    score input: heuristic", heuristic)
            print_stat("    score input: x (hidden*heuristic)", x)
            print_stat("    score output", raw_score)
        return raw_score


    def select_edges(self, graph, score):
        node_ratio = self.node_ratio if self.training else self.test_node_ratio
        degree_ratio = self.degree_ratio if self.training else self.test_degree_ratio
        ks = (node_ratio * graph.num_nodes).long()
        es = (degree_ratio * ks * graph.num_edges / graph.num_nodes).long()

        node_in = score.keys
        num_nodes = bincount(graph.node2graph[node_in], minlength=len(graph))
        ks = torch.min(ks, num_nodes)
        score_in = score[node_in]
        index = variadic_topks(score_in, num_nodes, ks=ks, break_tie=self.break_tie)[1]
        node_in = node_in[index]
        num_nodes = ks

        num_neighbors = graph.num_neighbors(node_in)
        num_edges = scatter_add(num_neighbors, graph.node2graph[node_in], dim_size=len(graph))
        es = torch.min(es, num_edges)

        num_edge_mean = num_edges.float().mean().clamp(min=1)
        chunk_size = max(int(1e7 / num_edge_mean), 1)
        num_nodes = num_nodes.split(chunk_size)
        num_edges = num_edges.split(chunk_size)
        es = es.split(chunk_size)
        num_chunk_nodes = [num_node.sum() for num_node in num_nodes]
        node_ins = node_in.split(num_chunk_nodes)

        edge_indexes = []
        for node_in, num_node, num_edge, e in zip(node_ins, num_nodes, num_edges, es):
            edge_index, node_out = graph.neighbors(node_in)
            score_edge = score[node_out]
            index = variadic_topks(score_edge, num_edge, ks=e, break_tie=self.break_tie)[1]
            edge_index = edge_index[index]
            edge_indexes.append(edge_index)
        edge_index = torch.cat(edge_indexes)

        return edge_index
    
    def remove_easy_edges(self, graph, h_index, t_index, r_index):
        if self.remove_one_hop:
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
            any = -torch.ones_like(h_index_ext)
            pattern = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
        else:
            pattern = torch.stack([h_index, t_index, r_index], dim=-1)
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index