"""
PyG-based retriever implementations for MKGL
Replaces TorchDrug core.Configurable with direct instantiation
"""
import torch
from torch import nn
import torch.nn.functional as F
from gnn.model_new import ConditionedPNA, PNA


class BasePNARetriever(nn.Module): 
    """
    Base retriever for text information aggregation
    """
    
    def __init__(self, config, text_embeddings, kgl2token, orig_vocab_size):
        super().__init__()
        self.config = config
        self.text_embeddings = text_embeddings
        self.kgl2token = kgl2token
        self.orig_vocab_size = orig_vocab_size
        
        self.down_scaling = nn.Linear(
            self.config.llm_hidden_dim, self.config.r, bias=False, dtype=torch.float)
        
        if self.config.text_encoder == 'pna':
            self.re_scaling = nn.Linear(config.r * 12, self.config.r)
    
    def aggregate_text(self, token_ids, text_embeddings, method='pna'):
        """
        Aggregate text embeddings using PNA-style aggregation
        
        Args:
            token_ids: [batch_size, seq_len] token indices
            text_embeddings: [vocab_size, hidden_dim] embedding table
            method: aggregation method ('mean' or 'pna')
        
        Returns:
            aggregated: [batch_size, hidden_dim] aggregated embeddings
        """
        device = text_embeddings.device
        
        token_ids = token_ids.to(device)  # Batch x Length
        token_mask = (token_ids > 0).unsqueeze(-1).to(device)  # B x L x 1
        token_lengths = token_mask.half().sum(axis=1).to(device)  # B x 1
        degree = token_lengths
        token_embs = text_embeddings[token_ids]  # B x L x Hidden
        
        mean = (token_embs * token_mask).sum(axis=1) / (token_lengths + 1e-10)
        
        if method == 'mean':
            result = mean
        else:
            # PNA aggregation: mean, max, min, std
            sq_mean = (token_embs ** 2 * token_mask).sum(axis=1) / (token_lengths + 1e-10)
            max_val, _ = (token_embs * token_mask + (1 - token_mask) * -1e10).max(axis=1)
            min_val, _ = (token_embs * token_mask + (1 - token_mask) * 1e10).min(axis=1)
            std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
            
            features = torch.cat([mean, max_val, min_val, std], dim=-1)
            
            # Degree scalers
            scale = degree.log()
            scale = scale / (scale.mean() + 1e-10)
            scales = torch.cat(
                [torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            
            result = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)

        return result 
    
    def retrieve_text(self, token_ids):
        """
        Retrieve and aggregate text representations
        
        Args:
            token_ids: [num_tokens, seq_len] token indices
        
        Returns:
            text_embs: [num_tokens, r] aggregated text embeddings
        """
        R = self.down_scaling(self.text_embeddings)
        
        result = self.aggregate_text(token_ids, R, self.config.text_encoder)
        
        if self.config.text_encoder == 'pna':
            result = self.re_scaling(result)
        
        return self.norm(result)

    def norm(self, x):
        """L2 normalize embeddings"""
        return F.normalize(x, p=2, dim=1)
    
    def forward(self, kgl_ids=None):
        """
        Forward pass to retrieve text embeddings
        
        Args:
            kgl_ids: Optional[Tensor] KGL token IDs (if None, retrieve all)
        
        Returns:
            text_embs: [num_tokens, r] text embeddings
        """
        if kgl_ids is not None:
            kgl_ids = kgl_ids - self.orig_vocab_size
            token_ids = self.kgl2token[kgl_ids.cpu()]
        else:
            token_ids = self.kgl2token
        return self.retrieve_text(token_ids)


class ContextRetriever(BasePNARetriever):
    """
    Context retriever that produces contextualized embeddings for entities
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.up_scaling = nn.Linear(
            self.config.r, self.config.llm_hidden_dim, bias=False, dtype=torch.float)

    def forward(self, kgl_ids, graph, all_index, all_kgl_index):
        """
        Forward pass to retrieve context embeddings
        
        Args:
            kgl_ids: [batch_size] KGL token IDs
            graph: PyGGraphWrapper knowledge graph
            all_index: [num_entities] all entity indices
            all_kgl_index: [num_entities] all KGL token indices
        
        Returns:
            context: [batch_size, llm_hidden_dim] context embeddings
        """
        text_embs = super().forward(kgl_ids)
        context = self.up_scaling(text_embs)
        return context


class ScoreRetriever(BasePNARetriever):
    """
    Score retriever that uses GNN to compute link prediction scores
    """
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        # Create GNN encoder from config
        self.kg_retriever = self._create_kg_encoder(config.kg_encoder)
        
        self.h_down_scaling = nn.Linear(
            self.config.llm_hidden_dim, self.config.r, bias=False, dtype=torch.float)
        self.r_down_scaling = nn.Linear(
            self.config.llm_hidden_dim, self.config.r, bias=False, dtype=torch.float)

    def _create_kg_encoder(self, encoder_config):
        """
        Create GNN encoder from config dict (replaces core.Configurable)
        
        Args:
            encoder_config: Config dict with class name and parameters
        
        Returns:
            encoder: GNN model instance
        """
        encoder_class = encoder_config.get('class', 'ConditionedPNA')
        
        # Extract base layer config
        base_layer_config = encoder_config.get('base_layer', {})
        input_dim = base_layer_config.get('input_dim', 64)
        output_dim = base_layer_config.get('output_dim', 64)
        query_input_dim = base_layer_config.get('query_input_dim', 64)
        num_relation = base_layer_config.get('num_relation', 100)
        aggregate_func = base_layer_config.get('aggregate_func', 'pna')
        layer_norm = base_layer_config.get('layer_norm', False)
        dependent = base_layer_config.get('dependent', True)
        
        # Extract model-level config
        num_layer = encoder_config.get('num_layer', 3)
        num_mlp_layer = encoder_config.get('num_mlp_layer', 2)
        node_ratio = encoder_config.get('node_ratio', 0.1)
        degree_ratio = encoder_config.get('degree_ratio', 1.0)
        test_node_ratio = encoder_config.get('test_node_ratio', None)
        test_degree_ratio = encoder_config.get('test_degree_ratio', None)
        remove_one_hop = encoder_config.get('remove_one_hop', False)
        break_tie = encoder_config.get('break_tie', False)
        
        if encoder_class == 'ConditionedPNA':
            return ConditionedPNA(
                input_dim=input_dim,
                hidden_dim=output_dim,
                num_layers=num_layer,
                num_relation=num_relation,
                query_input_dim=query_input_dim,
                num_mlp_layers=num_mlp_layer,
                node_ratio=node_ratio,
                degree_ratio=degree_ratio,
                test_node_ratio=test_node_ratio,
                test_degree_ratio=test_degree_ratio,
                aggregate_func=aggregate_func,
                remove_one_hop=remove_one_hop,
                break_tie=break_tie
            )
        elif encoder_class == 'PNA':
            return PNA(
                input_dim=input_dim,
                hidden_dim=output_dim,
                num_layers=num_layer,
                num_relation=num_relation,
                query_input_dim=query_input_dim,
                aggregate_func=aggregate_func,
                layer_norm=layer_norm,
                activation='relu',
                dependent=dependent
            )
        else:
            raise ValueError(f"Unknown encoder class: {encoder_class}")

    def forward(self, h_id, r_id, t_id, hidden_states, rel_hidden_states, 
                graph, all_index, all_kgl_index):
        """
        Forward pass to compute link prediction scores
        
        Args:
            h_id: [batch_size, num_neg+1] head entity IDs
            r_id: [batch_size, num_neg+1] relation IDs
            t_id: [batch_size, num_neg+1] tail entity IDs
            hidden_states: [batch_size, llm_hidden_dim] head embeddings from LLM
            rel_hidden_states: [batch_size, llm_hidden_dim] relation embeddings from LLM
            graph: PyGGraphWrapper knowledge graph
            all_index: [num_entities] all entity indices
            all_kgl_index: [num_entities] all KGL token indices
        
        Returns:
            score: [batch_size, num_neg+1] prediction scores
        """
        # Get text embeddings for all entities
        score_text_embs = super().forward(all_kgl_index)
        
        # Project LLM embeddings to GNN space
        head_embeds = self.h_down_scaling(hidden_states) 
        rel_embeds = self.r_down_scaling(rel_hidden_states)
        
        # Compute scores using GNN
        score = self.kg_retriever(
            h_id, r_id, t_id, 
            head_embeds, rel_embeds, 
            graph, score_text_embs, all_index
        )
        
        return score


class RelScoreRetriever(BasePNARetriever):
    """
    Relation score retriever for relation prediction tasks
    """
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.r_down_scaling = nn.Linear(
            self.config.llm_hidden_dim, self.config.r, bias=False, dtype=torch.float)

    def forward(self, rel_hidden_states, all_rel_kgl_index):
        """
        Forward pass to compute relation scores
        
        Args:
            rel_hidden_states: [batch_size, llm_hidden_dim] relation embeddings
            all_rel_kgl_index: [num_relations] all relation KGL indices
        
        Returns:
            score: [batch_size, num_relations] relation scores
        """
        score_text_embs = super().forward(all_rel_kgl_index)  # [num_rel, r]
        rel_embeds = self.r_down_scaling(rel_hidden_states)  # [batch_size, r]
        score = F.linear(rel_embeds, score_text_embs)
        return score
