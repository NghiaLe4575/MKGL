import argparse
import json
import os
import os.path as osp
import pickle
import yaml
import easydict
import numpy as np
import pandas as pd
import pprint
import swifter
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from dataset_new import FB15k237Inductive, WN18RRInductive, FB15k237, WN18RR
# Import the PyG datasets we migrated
class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template['description']}")

    def generate_prompt(
        self,
        instruction: str,
        input: str = None,
        label: str = None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


class InductiveKGCDataset(object):
    """
    Wrapper class that takes a raw PyG dataset (kgdata), tokenizes it, 
    and adds the instruction tuning text prompts.
    """
    def __init__(self, args, kgdata, tokenizer):
        self.args = args
        self.kgdata = kgdata
        self.tokenizer = tokenizer
        self.prompter = Prompter('alpaca_short', verbose=False)
        self.inv_prefix = '/inv'
        self.inv_fine_prefix = 'inverse of '

        self.read_vocab()
        self.read_data()
        self.add_input_text()
        self.post_process()

        self.saved_dir = 'data/preprocessed/'
        self.save()

    def read_vocab(self):
        kgdata = self.kgdata

        # Determine name prefix based on config name
        if 'fb15' in self.args.config_name:
            name_prefix = './data/names/fb15k237/'
        elif 'wn18' in self.args.config_name:
            name_prefix = './data/names/wn18rr/'
        else:
            # Fallback or error
            name_prefix = './data/names/fb15k237/'

        # Read fine-grained descriptions from text files
        ent_name = pd.read_csv(name_prefix+'entity.txt',
                               sep='\t', header=None, names=['raw_name', 'fine_name'], dtype=str)
        ent2text = pd.Series(ent_name['fine_name'].values,
                             index=ent_name['raw_name'].values)
        
        rel_name = pd.read_csv(name_prefix+'relation.txt',
                               sep='\t', header=None, names=['raw_name', 'fine_name'])
        rel2text = pd.Series(rel_name['fine_name'].values,
                             index=rel_name['raw_name'].values)

        # Build DataFrames using vocab lists from the PyG dataset
        # Note: kgdata.transductive_vocab is a list of raw strings
        trans_ent_vocab_df = pd.DataFrame({
            'kg_id': range(len(kgdata.transductive_vocab)), 
            'raw_name': kgdata.transductive_vocab, 
            'transductive': 1
        })
        
        ind_ent_vocab_df = pd.DataFrame({
            'kg_id': range(len(kgdata.inductive_vocab)), 
            'raw_name': kgdata.inductive_vocab, 
            'transductive': 0
        })
        
        ent_vocab_df = pd.concat([trans_ent_vocab_df, ind_ent_vocab_df], ignore_index=True)
        
        # Map raw names to fine-grained descriptions
        # Using .get to avoid crashes if a name is missing in entity.txt
        ent_vocab_df['fine_name'] = ent_vocab_df['raw_name'].map(lambda x: ent2text.get(x, x))

        rel_vocab_df = pd.DataFrame({
            'kg_id': range(len(kgdata.relation_vocab)), 
            'raw_name': kgdata.relation_vocab, 
            'transductive': 0
        })
        rel_vocab_df['fine_name'] = rel_vocab_df['raw_name'].map(lambda x: rel2text.get(x, x))

        # Create Inverse Relations
        inv_rel_vocab_df = rel_vocab_df.iloc[:]
        inv_rel_vocab_df = inv_rel_vocab_df.copy() # Avoid SettingWithCopy warning
        inv_rel_vocab_df['kg_id'] += len(inv_rel_vocab_df)
        inv_rel_vocab_df['raw_name'] = self.inv_prefix + inv_rel_vocab_df['raw_name']
        inv_rel_vocab_df['fine_name'] = self.inv_fine_prefix + inv_rel_vocab_df['fine_name']

        rel_vocab_df = pd.concat([rel_vocab_df, inv_rel_vocab_df], ignore_index=True)

        # Handle overlapped names (same description for different IDs)
        def process_overlapped_name(rows):
            if len(rows) > 1:
                rows.loc[:, 'fine_name'] = rows.loc[:, 'fine_name'] + \
                    [' #%i' % i for i in range(1, len(rows)+1)]
            return rows

        ent_vocab_df = ent_vocab_df.groupby('fine_name', group_keys=False).apply(process_overlapped_name)
        # ent_vocab_df = ent_vocab_df.droplevel('fine_name').sort_index() # Not needed with group_keys=False usually, but keeping safe
        
        rel_vocab_df = rel_vocab_df.groupby('fine_name', group_keys=False).apply(process_overlapped_name)
        # rel_vocab_df = rel_vocab_df.droplevel('fine_name').sort_index()

        ent_vocab_df['entity'] = 1
        rel_vocab_df['entity'] = 0
        vocab_df = pd.concat([ent_vocab_df, rel_vocab_df], ignore_index=True)
        vocab_df['token_name'] = '<rdf: ' + vocab_df['fine_name'] + '>'

        # Tokenize
        def tokenize_vocab(df):
            new_tokens = df['token_name'].values.tolist()
            self.tokenizer.add_tokens(new_tokens)
            
            # Get added tokens map
            vocab_map = self.tokenizer.get_added_vocab()
            # FIX: Use get_vocab() instead of .vocab attribute
            base_vocab = self.tokenizer.get_vocab()
            
            # Look up tokens in added vocab first, then base vocab, default to 0
            df['token_index'] = [
                vocab_map.get(tn, base_vocab.get(tn, 0)) 
                for tn in df['token_name'].values
            ]

            rawname2tokenid = pd.Series(
                df['token_index'].values, index=df['raw_name'].values)

            df.set_index('token_index', inplace=True)
            fine_names = [str(n).strip() for n in df['fine_name'].values]
            
            tokenized = self.tokenizer(
                fine_names, add_special_tokens=False, truncation=True, padding=True
            )
            df['text_token_ids'] = tokenized.input_ids
            return df, rawname2tokenid

        self.vocab_df, self.rawname2tokenid = tokenize_vocab(vocab_df)

    def read_data(self):
        kgdata = self.kgdata
        # PyG Dataset.split() returns a list of Subsets
        train_set, valid_set, test_set = kgdata.split()

        def convert_to_df(subset, ent_vocab, rel_vocab, is_inductive=False):
            ev = pd.Series(ent_vocab)
            rv = pd.Series(rel_vocab)

            # Optimizing for PyG: extract tensors directly using indices
            # subset.dataset is the underlying dataset object
            # subset.indices are the indices for this split
            indices = subset.indices
            triplets = subset.dataset.triplets[indices]
            
            # Convert to numpy for DataFrame creation
            data_np = triplets.cpu().numpy()

            df = pd.DataFrame(data_np, columns=['h_id', 't_id', 'r_id'])
            
            df['h_raw'] = ev[df['h_id'].values].values
            df['t_raw'] = ev[df['t_id'].values].values
            df['r_raw'] = rv[df['r_id'].values].values

            # Map raw names to token IDs
            # Using .values ensures we pass numpy arrays to map, which is faster
            df['h_tokenid'] = self.rawname2tokenid[df['h_raw'].values].values
            df['t_tokenid'] = self.rawname2tokenid[df['t_raw'].values].values
            df['r_tokenid'] = self.rawname2tokenid[df['r_raw'].values].values
            
            # Inverse relation logic
            inv_r_raw = self.inv_prefix + df['r_raw'].values
            df['inv_r_tokenid'] = self.rawname2tokenid[inv_r_raw].values

            # Map token IDs to fine descriptions
            # Using reindex or loc
            df['h_fine'] = self.vocab_df.loc[df['h_tokenid'].values, 'fine_name'].values
            df['t_fine'] = self.vocab_df.loc[df['t_tokenid'].values, 'fine_name'].values
            df['r_fine'] = self.vocab_df.loc[df['r_tokenid'].values, 'fine_name'].values
            df['inv_r_fine'] = self.vocab_df.loc[df['inv_r_tokenid'].values, 'fine_name'].values

            return df

        train_df = convert_to_df(train_set, kgdata.transductive_vocab, kgdata.relation_vocab, is_inductive=False)
        valid_df = convert_to_df(valid_set, kgdata.transductive_vocab, kgdata.relation_vocab, is_inductive=False)
        test_df = convert_to_df(test_set, kgdata.inductive_vocab, kgdata.relation_vocab, is_inductive=True)

        train_df['split'] = 'train'
        valid_df['split'] = 'valid'
        test_df['split'] = 'test'
        self.train_df, self.valid_df, self.test_df = train_df, valid_df, test_df

    def add_input_text(self):
        print('##########Add input text##########')

        train_df, valid_df, test_df = self.train_df, self.valid_df, self.test_df
        vocab_df = self.vocab_df

        def produce_input_text(row):
            h_info = vocab_df.loc[row['h_tokenid']]
            t_info = vocab_df.loc[row['t_tokenid']]
            r_info = vocab_df.loc[row['r_tokenid']]
            inv_r_info = vocab_df.loc[row['inv_r_tokenid']]

            h = h_info['token_name']
            t = t_info['token_name']
            r = r_info['token_name']
            inv_r = inv_r_info['token_name']

            h_des = h_info['fine_name']
            t_des = t_info['fine_name']
            r_des = r_info['fine_name']
            inv_r_des = inv_r_info['fine_name']

            instruction = f'Suppose that you are an excellent linguist studying a three-word language. Given the following dictionary:\n\n Input\tType\tDescription\n{h}\tHead entity\t{h_des}\n{r}\tRelation\t{r_des}\n\nPlease complete the last word (?) of the sentence: {h}{r}?'
            inv_instruction = f'Suppose that you are an excellent linguist studying a three-word language. Given the following dictionary:\n\n Input\tType\tDescription\n{t}\tHead entity\t{t_des}\n{inv_r}\tRelation\t{inv_r_des}\n\nPlease complete the last word (?) of the sentence: {t}{inv_r}?'

            row['input_text'] = self.prompter.generate_prompt(instruction, label=f'{h}{r}')
            row['inv_input_text'] = self.prompter.generate_prompt(
                inv_instruction, label=f'{t}{inv_r}')

            return row

        test_df = test_df.swifter.apply(produce_input_text, axis=1)
        valid_df = valid_df.swifter.apply(produce_input_text, axis=1)
        train_df = train_df.swifter.apply(produce_input_text, axis=1)

        self.train_df, self.valid_df, self.test_df = train_df, valid_df, test_df

    def _to_hf_dataset(self, df):
        return Dataset.from_pandas(df)

    def post_process(self):
        print('##########Post process: convert to hf datasets##########')
        self.train_data = self._to_hf_dataset(self.train_df)
        self.valid_data = self._to_hf_dataset(self.valid_df)
        self.test_data = self._to_hf_dataset(self.test_df)

    def save(self):
        saved_dir = self.saved_dir
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        file_path = saved_dir + self.args.config_name + '.pkl'
        print('##########Save dataset in %s############' % file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path):
        print('##########Load dataset from %s############' % file_path)
        with open(file_path, 'rb') as f:
            return pickle.load(f)


class KGCDataset(InductiveKGCDataset):
    """
    Standard KGC Dataset Wrapper (Transductive).
    Adaptation for PyG compatibility, though main focus is Inductive.
    """
    def read_vocab(self):
        kgdata = self.kgdata
        
        # Similar name prefix logic
        if 'fb15' in self.args.config_name:
            name_prefix = './data/names/fb15k237/'
        elif 'wn18' in self.args.config_name:
            name_prefix = './data/names/wn18rr/'
        else:
            name_prefix = './data/names/fb15k237/'

        ent_name = pd.read_csv(name_prefix+'entity.txt',
                               sep='\t', header=None, names=['raw_name', 'fine_name'], dtype=str)
        ent2text = pd.Series(ent_name['fine_name'].values, index=ent_name['raw_name'].values)
        rel_name = pd.read_csv(name_prefix+'relation.txt',
                               sep='\t', header=None, names=['raw_name', 'fine_name'])
        rel2text = pd.Series(rel_name['fine_name'].values, index=rel_name['raw_name'].values)

        # For standard KGC, we use 'transductive_vocab' as the only entity vocab
        # If your dataset.py uses 'entity_vocab' for standard KGC, use that.
        # Assuming we reuse InductiveKnowledgeGraphDataset for simplicity where transductive_vocab == entity_vocab
        vocab_source = getattr(kgdata, 'transductive_vocab', [])
        
        ent_vocab_df = pd.DataFrame({
            'kg_id': range(len(vocab_source)), 
            'raw_name': vocab_source, 
            'transductive': 1
        })
        ent_vocab_df['fine_name'] = ent_vocab_df['raw_name'].map(lambda x: ent2text.get(x, x))

        rel_vocab_df = pd.DataFrame({
            'kg_id': range(len(kgdata.relation_vocab)), 
            'raw_name': kgdata.relation_vocab, 
            'transductive': 0
        })
        rel_vocab_df['fine_name'] = rel_vocab_df['raw_name'].map(lambda x: rel2text.get(x, x))

        inv_rel_vocab_df = rel_vocab_df.iloc[:].copy()
        inv_rel_vocab_df['kg_id'] += len(inv_rel_vocab_df)
        inv_rel_vocab_df['raw_name'] = self.inv_prefix + inv_rel_vocab_df['raw_name']
        inv_rel_vocab_df['fine_name'] = self.inv_fine_prefix + inv_rel_vocab_df['fine_name']

        rel_vocab_df = pd.concat([rel_vocab_df, inv_rel_vocab_df], ignore_index=True)

        def process_overlapped_name(rows):
            if len(rows) > 1:
                rows.loc[:, 'fine_name'] = rows.loc[:, 'fine_name'] + \
                    [' #%i' % i for i in range(1, len(rows)+1)]
            return rows

        ent_vocab_df = ent_vocab_df.groupby('fine_name', group_keys=False).apply(process_overlapped_name)
        rel_vocab_df = rel_vocab_df.groupby('fine_name', group_keys=False).apply(process_overlapped_name)

        ent_vocab_df['entity'] = 1
        rel_vocab_df['entity'] = 0
        vocab_df = pd.concat([ent_vocab_df, rel_vocab_df], ignore_index=True)
        vocab_df['token_name'] = '<rdf: ' + vocab_df['fine_name'] + '>'

        def tokenize_vocab(df):
            new_tokens = df['token_name'].values.tolist()
            self.tokenizer.add_tokens(new_tokens)
            
            # Get added tokens map
            vocab_map = self.tokenizer.get_added_vocab()
            # FIX: Use get_vocab() instead of .vocab attribute
            base_vocab = self.tokenizer.get_vocab()
            
            # Look up tokens in added vocab first, then base vocab, default to 0
            df['token_index'] = [
                vocab_map.get(tn, base_vocab.get(tn, 0)) 
                for tn in df['token_name'].values
            ]

            rawname2tokenid = pd.Series(
                df['token_index'].values, index=df['raw_name'].values)

            df.set_index('token_index', inplace=True)
            fine_names = [str(n).strip() for n in df['fine_name'].values]
            
            tokenized = self.tokenizer(
                fine_names, add_special_tokens=False, truncation=True, padding=True
            )
            df['text_token_ids'] = tokenized.input_ids
            return df, rawname2tokenid

        self.vocab_df, self.rawname2tokenid = tokenize_vocab(vocab_df)
    
    def read_data(self):
        # Override to ensure we look at the right vocab for all splits
        kgdata = self.kgdata
        train_set, valid_set, test_set = kgdata.split()

        def convert_to_df(subset, ent_vocab, rel_vocab):
            ev = pd.Series(ent_vocab)
            rv = pd.Series(rel_vocab)
            
            indices = subset.indices
            triplets = subset.dataset.triplets[indices]
            data_np = triplets.cpu().numpy()
            
            df = pd.DataFrame(data_np, columns=['h_id', 't_id', 'r_id'])
            df['h_raw'] = ev[df['h_id'].values].values
            df['t_raw'] = ev[df['t_id'].values].values
            df['r_raw'] = rv[df['r_id'].values].values

            df['h_tokenid'] = self.rawname2tokenid[df['h_raw'].values].values
            df['t_tokenid'] = self.rawname2tokenid[df['t_raw'].values].values
            df['r_tokenid'] = self.rawname2tokenid[df['r_raw'].values].values
            df['inv_r_tokenid'] = self.rawname2tokenid[self.inv_prefix + df['r_raw'].values].values

            df['h_fine'] = self.vocab_df.loc[df['h_tokenid'].values, 'fine_name'].values
            df['t_fine'] = self.vocab_df.loc[df['t_tokenid'].values, 'fine_name'].values
            df['r_fine'] = self.vocab_df.loc[df['r_tokenid'].values, 'fine_name'].values
            df['inv_r_fine'] = self.vocab_df.loc[df['inv_r_tokenid'].values, 'fine_name'].values

            return df

        # Use transductive vocab for all splits in standard KGC
        vocab = getattr(kgdata, 'transductive_vocab', [])
        train_df = convert_to_df(train_set, vocab, kgdata.relation_vocab)
        valid_df = convert_to_df(valid_set, vocab, kgdata.relation_vocab)
        test_df = convert_to_df(test_set, vocab, kgdata.relation_vocab)

        train_df['split'] = 'train'
        valid_df['split'] = 'valid'
        test_df['split'] = 'test'
        self.train_df, self.valid_df, self.test_df = train_df, valid_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument("--config", "-c", type=str, default='config/fb15k237.yaml')
    parser.add_argument("--version", "-v", type=str, default='')
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()
    
    # Load Config
    with open(args.config, "r") as f:
        cfg = easydict.EasyDict(yaml.safe_load(f))
        if 'ind' in args.config:
            if args.version:
                cfg.dataset.version = args.version
            else:
                print("Warning: Inductive config used but no version specified. Defaulting to 'v1'.")
                cfg.dataset.version = 'v1'

    # Set Config Name
    config_name = args.config.split('/')[-1].split('.')[0]
    if hasattr(cfg.dataset, 'version'):
        config_name += '_' + cfg.dataset.version
    args.config_name = config_name

    print('***************Read dataset from PyG (Migrated)***************')
    print("Config file: %s" % args.config)
    print("Config name: %s" % args.config_name)
    import pprint
    pprint.pprint(cfg)
    
    # Instantiate Dataset (Replaces TorchDrug core.Configurable)
    dataset_class_str = cfg.dataset.get('class', '')
    dataset_version = cfg.dataset.get('version', 'v1')
    
    kgdata = None
    # Inductive Check
    if 'FB15k237Inductive' in dataset_class_str:
        kgdata = FB15k237Inductive(version=dataset_version)
    elif 'WN18RRInductive' in dataset_class_str:
        kgdata = WN18RRInductive(version=dataset_version)
    # Standard Check (New!)
    elif 'FB15k237' in dataset_class_str:
        kgdata = FB15k237(version=dataset_version)
    elif 'WN18RR' in dataset_class_str:
        kgdata = WN18RR(version=dataset_version)
    else:
        print(f"Warning: Unknown dataset class {dataset_class_str} in config.")
        if 'ind' in args.config:
             raise ValueError("Please ensure dataset.py contains the inductive class requested.")

    print('***************Load tokenizer***************')
    tokenizer = AutoTokenizer.from_pretrained(**cfg.tokenizer)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'
    
    if kgdata:
        if 'ind' in args.config:
            dataset = InductiveKGCDataset(args, kgdata, tokenizer)
        else:
            dataset = KGCDataset(args, kgdata, tokenizer)