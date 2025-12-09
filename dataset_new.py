import os
import csv
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data, download_url
from tqdm import tqdm

class InductiveKnowledgeGraphDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.fact_graph = None
        self.graph = None
        self.inductive_fact_graph = None
        self.inductive_graph = None
        self.triplets = None
        self.num_samples = []
        
        # Vocabularies
        self.transductive_vocab = []
        self.inductive_vocab = []
        self.relation_vocab = []
        self.inv_transductive_vocab = {}
        self.inv_inductive_vocab = {}
        self.inv_relation_vocab = {}
        
        # Counts (Initialize to 0)
        self._num_transductive_nodes = 0
        self._num_inductive_nodes = 0
        self._num_relations = 0

    @property
    def num_relation(self):
        return self._num_relations
        
    @property
    def num_entity(self):
        return self._num_transductive_nodes
    
    @property
    def entity_vocab(self):
        """Alias for compatibility - returns transductive entities for standard datasets"""
        return self.transductive_vocab

    def _create_pyg_graph(self, triplets, num_nodes, num_relations):
        if isinstance(triplets, torch.Tensor):
        # Keep only rows where relation ID is valid
            mask = (triplets[:, 2] < num_relations) & (triplets[:, 2] >= 0)
            triplets = triplets[mask]
        if len(triplets) == 0:
            data = Data(edge_index=torch.empty((2, 0), dtype=torch.long),
                        edge_attr=torch.empty(0, dtype=torch.long),
                        num_nodes=num_nodes)
            data.num_relations = num_relations
            return data
            
        tensor_triplets = torch.tensor(triplets, dtype=torch.long)
        edge_index = torch.stack([tensor_triplets[:, 0], tensor_triplets[:, 1]], dim=0)
        edge_attr = tensor_triplets[:, 2]
        x = torch.arange(num_nodes, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        data.num_relations = num_relations
        return data

    def _finalize_vocab(self, inv_vocab):
        sorted_items = sorted(inv_vocab.items(), key=lambda x: x[1])
        vocab_list = [k for k, v in sorted_items]
        return vocab_list, inv_vocab

    def load_inductive_tsvs(self, transductive_files, inductive_files, verbose=0):
        inv_transductive_vocab = {}
        inv_inductive_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        # 1. Load Transductive
        for txt_file in transductive_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(list(reader), desc=f"Loading {os.path.basename(txt_file)}")
                else:
                    reader = list(reader)

                num_sample = 0
                for tokens in reader:
                    if len(tokens) < 3: continue
                    h_token, r_token, t_token = tokens[:3]
                    
                    if h_token not in inv_transductive_vocab:
                        inv_transductive_vocab[h_token] = len(inv_transductive_vocab)
                    h = inv_transductive_vocab[h_token]
                    
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    
                    if t_token not in inv_transductive_vocab:
                        inv_transductive_vocab[t_token] = len(inv_transductive_vocab)
                    t = inv_transductive_vocab[t_token]
                    
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        # 2. Load Inductive
        for txt_file in inductive_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(list(reader), desc=f"Loading {os.path.basename(txt_file)}")
                else:
                    reader = list(reader)

                num_sample = 0
                for tokens in reader:
                    if len(tokens) < 3: continue
                    h_token, r_token, t_token = tokens[:3]
                    
                    if h_token not in inv_inductive_vocab:
                        inv_inductive_vocab[h_token] = len(inv_inductive_vocab)
                    h = inv_inductive_vocab[h_token]
                    
                    if r_token not in inv_relation_vocab:
                        # Should exist in transductive, but if not, add it
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    
                    if t_token not in inv_inductive_vocab:
                        inv_inductive_vocab[t_token] = len(inv_inductive_vocab)
                    t = inv_inductive_vocab[t_token]
                    
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        # 3. Finalize Vocabs
        self.transductive_vocab, self.inv_transductive_vocab = self._finalize_vocab(inv_transductive_vocab)
        self.inductive_vocab, self.inv_inductive_vocab = self._finalize_vocab(inv_inductive_vocab)
        self.relation_vocab, self.inv_relation_vocab = self._finalize_vocab(inv_relation_vocab)

        self._num_transductive_nodes = len(self.transductive_vocab)
        self._num_inductive_nodes = len(self.inductive_vocab)
        self._num_relations = len(self.relation_vocab)

        # 4. Create Graphs
        idx_trans_train = num_samples[0]
        idx_trans_all = sum(num_samples[:3])
        idx_ind_train_start = sum(num_samples[:3])
        idx_ind_train_end = sum(num_samples[:4])
        
        self.fact_graph = self._create_pyg_graph(
            triplets[:idx_trans_train], self._num_transductive_nodes, self._num_relations
        )
        self.graph = self._create_pyg_graph(
            triplets[:idx_trans_all], self._num_transductive_nodes, self._num_relations
        )
        self.inductive_fact_graph = self._create_pyg_graph(
            triplets[idx_ind_train_start:idx_ind_train_end], self._num_inductive_nodes, self._num_relations
        )
        self.inductive_graph = self._create_pyg_graph(
            triplets[idx_ind_train_start:], self._num_inductive_nodes, self._num_relations
        )

        slice_1 = triplets[:sum(num_samples[:2])] 
        slice_2 = triplets[sum(num_samples[:4]):] 
        self.triplets = torch.tensor(slice_1 + slice_2, dtype=torch.long)
        self.num_samples = num_samples[:2] + [sum(num_samples[4:])]

    def __getitem__(self, index):
        return self.triplets[index]

    def __len__(self):
        return len(self.triplets)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


class StandardKGCDataset(InductiveKnowledgeGraphDataset):
    """Base class for standard (transductive) datasets."""
    
    def load_standard_tsvs(self, files, verbose=0):
        inv_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(list(reader), desc=f"Loading {os.path.basename(txt_file)}")
                else:
                    reader = list(reader)

                num_sample = 0
                for tokens in reader:
                    if len(tokens) < 3: continue
                    h_token, r_token, t_token = tokens[:3]

                    if h_token not in inv_entity_vocab:
                        inv_entity_vocab[h_token] = len(inv_entity_vocab)
                    h = inv_entity_vocab[h_token]

                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]

                    if t_token not in inv_entity_vocab:
                        inv_entity_vocab[t_token] = len(inv_entity_vocab)
                    t = inv_entity_vocab[t_token]

                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        self.transductive_vocab, self.inv_transductive_vocab = self._finalize_vocab(inv_entity_vocab)
        self.relation_vocab, self.inv_relation_vocab = self._finalize_vocab(inv_relation_vocab)
        self.inductive_vocab = []  # Empty for standard datasets
        self.inv_inductive_vocab = {}  # Empty for standard datasets
        
        self._num_transductive_nodes = len(self.transductive_vocab)
        self._num_relations = len(self.relation_vocab)

        self.fact_graph = self._create_pyg_graph(
            triplets[:num_samples[0]], self._num_transductive_nodes, self._num_relations
        )
        self.graph = self._create_pyg_graph(
            triplets, self._num_transductive_nodes, self._num_relations
        )
        # No inductive graphs
        self.inductive_fact_graph = None
        self.inductive_graph = None

        self.triplets = torch.tensor(triplets, dtype=torch.long)
        self.num_samples = num_samples


# --- Inductive Classes ---

class FB15k237Inductive(InductiveKnowledgeGraphDataset):
    transductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/test.txt",
    ]
    inductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
    ]

    def __init__(self, path="data/datasets", version="v1", verbose=1):
        super().__init__()
        if not os.path.exists(path): os.makedirs(path)
        
        trans_files, ind_files = [], []
        for url in self.transductive_urls:
            url = url % version
            save_file = f"fb15k237_{version}_{os.path.basename(url)}"
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                print(f"Downloading {url}...")
                download_url(url, path, filename=save_file)
            trans_files.append(txt_file)
            
        for url in self.inductive_urls:
            url = url % version
            save_file = f"fb15k237_{version}_ind_{os.path.basename(url)}"
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                print(f"Downloading {url}...")
                download_url(url, path, filename=save_file)
            ind_files.append(txt_file)

        self.load_inductive_tsvs(trans_files, ind_files, verbose=verbose)


class WN18RRInductive(InductiveKnowledgeGraphDataset):
    transductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/test.txt",
    ]
    inductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
    ]

    def __init__(self, path="data/datasets", version="v1", verbose=1):
        super().__init__()
        if not os.path.exists(path): os.makedirs(path)
        
        trans_files, ind_files = [], []
        for url in self.transductive_urls:
            url = url % version
            save_file = f"wn18rr_{version}_{os.path.basename(url)}"
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                download_url(url, path, filename=save_file)
            trans_files.append(txt_file)
            
        for url in self.inductive_urls:
            url = url % version
            save_file = f"wn18rr_{version}_ind_{os.path.basename(url)}"
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                download_url(url, path, filename=save_file)
            ind_files.append(txt_file)

        self.load_inductive_tsvs(trans_files, ind_files, verbose=verbose)


# --- Standard Classes (The missing ones) ---

class FB15k237(StandardKGCDataset):
    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/test.txt",
    ]

    def __init__(self, path="data/datasets", version="v1", verbose=1):
        super().__init__()
        if not os.path.exists(path): os.makedirs(path)
        
        files = []
        for url in self.urls:
            url = url % version
            save_file = f"fb15k237_{version}_{os.path.basename(url)}"
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                print(f"Downloading {url}...")
                download_url(url, path, filename=save_file)
            files.append(txt_file)

        self.load_standard_tsvs(files, verbose=verbose)


class WN18RR(StandardKGCDataset):
    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/test.txt",
    ]

    def __init__(self, path="data/datasets", version="v1", verbose=1):
        super().__init__()
        if not os.path.exists(path): os.makedirs(path)
        
        files = []
        for url in self.urls:
            url = url % version
            save_file = f"wn18rr_{version}_{os.path.basename(url)}"
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                print(f"Downloading {url}...")
                download_url(url, path, filename=save_file)
            files.append(txt_file)

        self.load_standard_tsvs(files, verbose=verbose)