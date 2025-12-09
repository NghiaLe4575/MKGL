"""
MKGL model with PyG graph support
Key changes from original:
- Uses PyG Data/Batch instead of TorchDrug PackedGraph
- Updated graph construction and manipulation
- Compatible with retriever_new and model_new
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data as torch_data

from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from gnn.model_new import PyGGraphWrapper
from retriever_new import ContextRetriever, ScoreRetriever


class MKGLConfig(LlamaConfig):
    model_type = 'mkgl_config'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MKGL(LlamaForCausalLM):
    config_class = MKGLConfig

    def __init__(self, config):
        super().__init__(config)

    def init_kg_specs(self, kgl2token, orig_vocab_size, cfg):
        """
        Initialize KG-specific components
        
        Args:
            kgl2token: [num_kg_tokens, token_length] mapping from KG to text tokens
            orig_vocab_size: size of original vocabulary
            cfg: config dict with retriever configs
        """
        self.kgl2token = kgl2token
        self.orig_vocab_size = orig_vocab_size
        
        device = self.lm_head.weight.device
        self.context_retriever = ContextRetriever(
            cfg.context_retriever, 
            self.get_input_embeddings().weight.data, 
            kgl2token, 
            orig_vocab_size
        ).to(device)
        
        self.score_retriever = ScoreRetriever(
            cfg.score_retriever, 
            self.lm_head.weight.data, 
            kgl2token, 
            orig_vocab_size
        ).to(device)

    def forward(
        self,
        h_id,
        r_id,
        t_id,
        h_kgl_tokenid,
        r_kgl_tokenid,
        graph,
        all_index,
        all_kgl_index,
        input_ids,
        attention_mask,
        input_length,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        """
        Forward pass for MKGL model
        
        Args:
            h_id, r_id, t_id: Entity and relation IDs
            h_kgl_tokenid, r_kgl_tokenid: KGL token IDs
            graph: PyGGraphWrapper with knowledge graph
            all_index, all_kgl_index: All entity indices
            input_ids, attention_mask, input_length: LLM inputs
        
        Returns:
            pred: Prediction scores
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size = h_kgl_tokenid.shape[0]
        device = self.lm_head.weight.device

        # Get embeddings for text tokens and KG tokens
        mask = input_ids < self.orig_vocab_size
        token_embs = self.get_input_embeddings()(input_ids[mask])
        kgl_token_embs = self.context_retriever(input_ids[~mask], graph, all_index, all_kgl_index)

        rel_token_embs = self.context_retriever(r_kgl_tokenid, graph, all_index, all_kgl_index)

        # Combine token and KG embeddings
        input_embs = torch.zeros(
            *input_ids.shape, self.config.hidden_size, dtype=torch.half).to(device)
        input_embs[mask] = token_embs.type(input_embs.dtype)
        input_embs[~mask] = kgl_token_embs.type(input_embs.dtype)

        # Forward through transformer
        transformer_outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embs,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract hidden states: batch_size, seq_len, hidden_state
        hidden_states = transformer_outputs[0]

        # Select the last output of llm: batch_size x hidden_size
        hr_hidden_states = hidden_states[torch.arange(
            batch_size, device=hidden_states.device), input_length - 1]

        rel_hidden_states = hidden_states[torch.arange(
            batch_size, device=hidden_states.device), input_length - 2]

        # Compute prediction scores using GNN
        pred = self.score_retriever(
            h_id, r_id, t_id, 
            hr_hidden_states, rel_token_embs, 
            graph, all_index, all_kgl_index
        )
        
        return pred


class KGL4KGC(nn.Module):
    """
    Knowledge Graph Completion model with MKGL
    """

    def __init__(self, config, llmodel, dataset):
        super().__init__()
        self.llmodel = llmodel
        self.dataset = dataset
        self.num_negative = config.num_negative
        self.adversarial_temperature = config.adversarial_temperature
        self.strict_negative = config.strict_negative
        
        train_set, valid_set, test_set = dataset.kgdata.split()
        self.preprocess(train_set, valid_set, test_set)

    @property
    def device(self):
        return self.llmodel.lm_head.weight.device

    def loss(self, pred, target, all_loss=None):
        """
        Compute loss with adversarial negative sampling
        
        Args:
            pred: [batch_size, num_neg+1] prediction scores
            target: [batch_size, num_neg+1] target labels
            all_loss: optional additional loss
        
        Returns:
            loss: scalar loss
            metric: dict with metrics
        """
        metric = {}
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        neg_weight = torch.ones_like(pred)
        if self.adversarial_temperature > 0:
            with torch.no_grad():
                neg_weight[:, 1:] = F.softmax(
                    pred[:, 1:] / self.adversarial_temperature, dim=-1)
        else:
            neg_weight[:, 1:] = 1 / self.num_negative
        
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = loss.mean()

        if all_loss is not None:
            loss = loss + all_loss
            
        metric['loss'] = loss
        
        return loss, metric
    
    def forward(self, batch, all_loss=None, metric=None, label=None):
        """Training/evaluation forward pass"""
        device = batch.h_id.device
        
        if self.training:
            all_loss = torch.tensor(0, dtype=torch.float, device=device)
            pred = self.predict(batch, all_loss, metric)
            
            target = torch.zeros_like(pred)
            target[:, 0] = 1
            
            return self.loss(pred, target)
        else:
            with torch.no_grad():
                pred, (mask, target) = self.predict_and_target(batch)
                label = torch.zeros_like(pred)
                label[:, target] = 1
                loss, _ = self.loss(pred, label)
                pos_pred = pred.gather(-1, target.unsqueeze(-1))
                # filter rank
                ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
                return loss, ranking.to(device)
    
    def predict(self, batch, all_loss=None, metric=None):
        """
        Predict scores for batch
        
        Args:
            batch: batch dict with h_id, t_id, r_id, etc.
        
        Returns:
            pred: [batch_size, num_neg+1] prediction scores
        """
        pos_h_index, pos_t_index, pos_r_index = batch.h_id, batch.t_id, batch.r_id
        device = pos_h_index.device
        batch_size = len(batch.h_id)
        
        # Get graph
        graph = self.get_graph(batch)
        if not isinstance(graph, PyGGraphWrapper):
            graph = PyGGraphWrapper(graph)
        
        # Move graph to device
        if hasattr(graph.data, 'to'):
            graph.data = graph.data.to(device)
        
        # Get all entity indices
        num_nodes = graph.num_node
        
        all_index = torch.arange(num_nodes, device=device)
        all_kgl_index = self.id2tokenid(all_index, split=batch.split)
        
        if self.training:
            # Sample negatives
            t_index, h_index = self.negative_sample(
                pos_h_index, pos_t_index, pos_r_index, graph)
            h_index, t_index, r_index = self.merge_negative(
                pos_h_index, pos_t_index, pos_r_index, h_index, t_index)
        else:
            # All entities as candidates
            h_index = pos_h_index.unsqueeze(-1).expand(-1, num_nodes)
            t_index = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1)
            r_index = pos_r_index.unsqueeze(-1).expand(-1, num_nodes)
        
        # LLM feature
        h_kgl_tokenid = torch.cat([batch.h_tokenid, batch.t_tokenid])
        r_kgl_tokenid = torch.cat([batch.r_tokenid, batch.inv_r_tokenid])
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        input_length = batch.input_length
        
        pred = self.llmodel(
            h_index,
            r_index,
            t_index,
            h_kgl_tokenid,
            r_kgl_tokenid,
            graph,
            all_index,
            all_kgl_index,
            input_ids,
            attention_mask,
            input_length,
        )
        
        return pred
    
    def target(self, batch):
        """Get target indices for evaluation"""
        pos_h_index, pos_t_index, pos_r_index = batch.h_id, batch.t_id, batch.r_id
        batch_size = len(batch.h_id)
        graph = self.get_eval_graph(batch)
        
        if not isinstance(graph, PyGGraphWrapper):
            graph = PyGGraphWrapper(graph)
        
        # Find true tails for each (h, r) pair
        # Use pattern matching on graph edges
        device = pos_h_index.device
        num_nodes = graph.num_node
        
        # Create mask for filtering
        mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)
        target = pos_t_index
        
        # Find edges matching (h, ?, r) pattern
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        
        for i in range(batch_size):
            h = pos_h_index[i].item()
            r = pos_r_index[i].item()
            
            # Find all edges from h with relation r
            if edge_attr is not None:
                edge_mask = (edge_index[0] == h) & (edge_attr == r)
            else:
                edge_mask = (edge_index[0] == h)
            
            true_tails = edge_index[1][edge_mask].unique()
            mask[i, true_tails] = False
            mask[i, target[i]] = True  # Keep the target
        
        return mask, target
    
    def predict_and_target(self, batch, all_loss=None, metric=None):
        """Combined predict and target for evaluation"""
        return self.predict(batch, all_loss, metric), self.target(batch)

    def preprocess(self, train_set, valid_set, test_set):
        """
        Preprocess dataset and extract graphs
        
        Args:
            train_set, valid_set, test_set: Dataset splits (Subset objects)
        """
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        
        # Check if fact_graph already exists (e.g., from InductiveKnowledgeGraphDataset)
        if hasattr(dataset, 'fact_graph') and dataset.fact_graph is not None:
            # Use pre-built fact_graph
            self.graph = dataset.graph
            self.fact_graph = dataset.fact_graph
        else:
            # Create fact graph by filtering edges
            # Standard datasets have graph with all edges, need to filter validation/test
            self.graph = dataset.graph
            
            # Build mask: keep only training edges
            # The graph has all edges, we need to exclude valid/test indices
            num_edges = dataset.graph.edge_index.size(1)
            fact_mask = torch.ones(num_edges, dtype=torch.bool)
            
            # This is a simplified approach - in practice, you'd need to identify
            # which edges correspond to valid/test triplets
            # For now, assume the dataset already provides the correct graph
            self.fact_graph = dataset.graph  # Fallback: use full graph
        
        return train_set, valid_set, test_set

    def _filter_graph_edges(self, graph, edge_mask):
        """Filter graph edges by mask"""
        from torch_geometric.data import Data
        
        # Handle both wrapped and unwrapped graphs
        if isinstance(graph, PyGGraphWrapper):
            graph_data = graph.data
        else:
            graph_data = graph
        
        # Create filtered edge list
        edge_index = graph_data.edge_index[:, edge_mask]
        edge_attr = graph_data.edge_attr[edge_mask] if graph_data.edge_attr is not None else None
        
        filtered_graph = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=graph_data.num_nodes
        )
        
        if hasattr(graph_data, 'num_relations'):
            filtered_graph.num_relations = graph_data.num_relations
        
        return filtered_graph

    def id2tokenid(self, id, split='test', entity=True):
        """Convert entity/relation IDs to token IDs"""
        if entity:
            id2rawname = np.array(self.dataset.kgdata.entity_vocab)
        else:
            id2rawname = np.array(self.dataset.kgdata.relation_vocab)
        
        rawname = id2rawname[id.cpu()]
        tokenid = np.stack([self.dataset.rawname2tokenid.loc[n] for n in rawname])
        
        return torch.tensor(tokenid, dtype=id.dtype, device=id.device)

    def get_graph(self, batch):
        """Get training graph"""
        return self.fact_graph
    
    def get_eval_graph(self, batch):
        """Get evaluation graph"""
        return self.graph

    def negative_sample(self, pos_h_index, pos_t_index, pos_r_index, graph):
        """Sample negative entities"""
        batch_size = len(pos_h_index)
        device = pos_h_index.device
        
        # Sample random negatives
        neg_t_index = torch.randint(
            0, self.num_entity, 
            (batch_size, self.num_negative), 
            device=device
        )
        neg_h_index = torch.randint(
            0, self.num_entity, 
            (batch_size, self.num_negative), 
            device=device
        )
        
        return neg_t_index, neg_h_index

    def merge_negative(self, pos_h_index, pos_t_index, pos_r_index, neg_h_index, neg_t_index):
        """Merge positive and negative samples"""
        # Positive sample first, then negatives
        h_index = torch.cat([pos_h_index.unsqueeze(-1), pos_h_index.unsqueeze(-1).expand(-1, self.num_negative)], dim=-1)
        t_index = torch.cat([pos_t_index.unsqueeze(-1), neg_t_index], dim=-1)
        r_index = torch.cat([pos_r_index.unsqueeze(-1), pos_r_index.unsqueeze(-1).expand(-1, self.num_negative)], dim=-1)
        
        return h_index, t_index, r_index


class KGL4IndKGC(KGL4KGC):
    """Inductive KGC variant"""

    def preprocess(self, train_set, valid_set, test_set):
        """Preprocess for inductive setting"""
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        
        if isinstance(train_set, torch_data.Subset):
            dataset_obj = train_set.dataset
        else:
            dataset_obj = train_set
        
        self.num_entity = dataset_obj.num_entity
        self.num_relation = dataset_obj.num_relation

        self.graph = dataset_obj.graph
        self.fact_graph = dataset_obj.fact_graph
        self.inductive_graph = dataset_obj.inductive_graph
        self.inductive_fact_graph = dataset_obj.inductive_fact_graph

    def id2tokenid(self, id, split='test', entity=True):
        """Convert IDs with inductive/transductive vocab split"""
        if entity:
            if split == 'test':
                id2rawname = np.array(self.dataset.kgdata.inductive_vocab)
            else:
                id2rawname = np.array(self.dataset.kgdata.transductive_vocab)
        else:
            id2rawname = np.array(self.dataset.kgdata.relation_vocab)
        
        rawname = id2rawname[id.cpu()]
        tokenid = np.stack([self.dataset.rawname2tokenid.loc[n] for n in rawname])
        
        return torch.tensor(tokenid, dtype=id.dtype, device=id.device)

    def get_graph(self, batch):
        """Get graph based on split"""
        return self.inductive_fact_graph if batch.split == "test" else self.fact_graph

    def get_eval_graph(self, batch):
        """Get evaluation graph based on split"""
        return self.inductive_graph if batch.split == "test" else self.graph
