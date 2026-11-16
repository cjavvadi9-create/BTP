import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import logging
from preprocessing.compound import *
from preprocessing.protein_pretrain.protein_structure import residue_features

logger = logging.getLogger(__name__)


class CPIDataset(Dataset):


    def __init__(self, file_path, compound_data_path, protein_data_path, strict=False):

        if isinstance(file_path, pd.DataFrame):
            self.raw_data = file_path.reset_index(drop=True)
        else:
            self.raw_data = pd.read_csv(file_path)
        self.smiles_values = self.raw_data['compound_iso_smiles'].values
        self.sequence_values = self.raw_data['target_sequence'].values
        self.compound_data_path = compound_data_path
        self.protein_data_path = protein_data_path
        self.strict = strict

        self.label_values = self.raw_data['label'].values

        self.prot_id = self.raw_data['target_id'].values

        mol_dict_file = f'{self.compound_data_path}/mol_dict.pkl'
        if os.path.exists(mol_dict_file):
            with open(mol_dict_file, 'rb') as file:
                self.loaded_dict = pickle.load(file)
        else:
            self.loaded_dict = None

   

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        smiles = self.smiles_values[idx]
        label = self.label_values[idx]
        target_id = str(self.prot_id[idx])
        sequence = self.sequence_values[idx]

        compound_node_features, compound_adj_matrix = get_mol_features(smiles, strict=self.strict)

        if self.loaded_dict is not None and smiles in self.loaded_dict:
            smiles_embedding = self.loaded_dict[smiles]
        else:
            if self.strict and self.loaded_dict is not None:
                raise KeyError(f"SMILES '{smiles}' missing from {self.compound_data_path}/mol_dict.pkl under --strict")
            smiles_embedding = np.zeros((1, 384), dtype=np.float32)

        # Protein sequence embedding
        seq_emb_path = os.path.join(self.protein_data_path, f'{target_id}.npy')
        if not os.path.isfile(seq_emb_path):
            parent_seq_emb = os.path.join(os.path.dirname(self.protein_data_path), f'{target_id}.npy')
            if os.path.isfile(parent_seq_emb):
                seq_emb_path = parent_seq_emb

        if os.path.isfile(seq_emb_path):
            protein_seq_embedding = np.load(seq_emb_path).squeeze()
        else:
            if self.strict:
                raise FileNotFoundError(f"Missing protein sequence embedding for {target_id} at {seq_emb_path} under --strict")
            L_seq = min(len(sequence), 1022)
            protein_seq_embedding = np.zeros((L_seq, 480), dtype=np.float32)

        # Protein structure files
        node_file = os.path.join(self.protein_data_path, 'structure', f'{target_id}_node.npy')
        adj_file = os.path.join(self.protein_data_path, 'structure', f'{target_id}_adj.npy')

        if not (os.path.isfile(node_file) and os.path.isfile(adj_file)):
            parent_node = os.path.join(os.path.dirname(self.protein_data_path), 'structure', f'{target_id}_node.npy')
            parent_adj = os.path.join(os.path.dirname(self.protein_data_path), 'structure', f'{target_id}_adj.npy')
            if os.path.isfile(parent_node) and os.path.isfile(parent_adj):
                node_file = parent_node
                adj_file = parent_adj

        if os.path.isfile(node_file) and os.path.isfile(adj_file):
            protein_node_features = np.load(node_file)
            protein_adj_matrix = np.load(adj_file)
        else:
            if self.strict:
                raise FileNotFoundError(f"Missing protein structure files for {target_id} under --strict")
            L = len(sequence)
            protein_node_features = residue_features(sequence, np.zeros(L, dtype=np.float32))
            protein_adj_matrix = np.eye(L, dtype=np.float32)

        return {
            'COMPOUND_NODE_FEAT': compound_node_features,
            'COMPOUND_ADJ': compound_adj_matrix,
            'COMPOUND_EMBEDDING': smiles_embedding,
            'PROTEIN_EMBEDDING': protein_seq_embedding,
            'PROTEIN_NODE_FEAT': protein_node_features,
            'PROTEIN_ADJ': protein_adj_matrix,
            # 'PROTEIN_CONTACT_MAP':contact_map,
            'LABEL': label,
        }

    def collate_fn(self, batch):

        batch_size = len(batch)

        compound_node_nums = [item['COMPOUND_NODE_FEAT'].shape[0] for item in batch]
        compound_smiles_nums = [item['COMPOUND_EMBEDDING'].shape[0] for item in batch]
        protein_node_nums = [item['PROTEIN_EMBEDDING'].shape[0] for item in batch]
        protein_struct_node_nums = [item['PROTEIN_NODE_FEAT'].shape[0] for item in batch]

        max_compound_len = max(compound_node_nums)
        max_protein_len = max(protein_node_nums)
        max_smiles_len = max(compound_smiles_nums)
        max_protein_struct_len = max(protein_struct_node_nums)

        compound_node_features = torch.zeros((batch_size, max_compound_len, batch[0]['COMPOUND_NODE_FEAT'].shape[1]))
        compound_adj_matrix = torch.zeros((batch_size, max_compound_len, max_compound_len))
        compound_embedding = torch.zeros((batch_size, max_smiles_len, batch[0]['COMPOUND_EMBEDDING'].shape[1]))
        protein_seq_embedding = torch.zeros((batch_size, max_protein_len, batch[0]['PROTEIN_EMBEDDING'].shape[1]))
        protein_node_features = torch.zeros((batch_size, max_protein_struct_len, batch[0]['PROTEIN_NODE_FEAT'].shape[1]))
        protein_adj_matrix = torch.zeros((batch_size, max_protein_struct_len, max_protein_struct_len))
        # protein_contact_map = torch.zeros((batch_size, max_protein_len, max_protein_len))

        labels, seqs = list(), list()
        for i, item in enumerate(batch):
            v = item['COMPOUND_NODE_FEAT']
            compound_node_features[i, :v.shape[0], :] = torch.FloatTensor(v)
            v = item['COMPOUND_ADJ']
            compound_adj_matrix[i, :v.shape[0], :v.shape[0]] = torch.FloatTensor(v)

            v = item['COMPOUND_EMBEDDING']
            compound_embedding[i, :v.shape[0], :] = torch.FloatTensor(v)

            # v = item['PROTEIN_CONTACT_MAP']
            # protein_contact_map[i, :v.shape[0], :v.shape[0]] = torch.FloatTensor(v)

            v = item['PROTEIN_EMBEDDING']
            protein_seq_embedding[i, :v.shape[0], :] = torch.FloatTensor(v)

            v = item['PROTEIN_NODE_FEAT']
            protein_node_features[i, :v.shape[0], :] = torch.FloatTensor(v)
            v = item['PROTEIN_ADJ']
            protein_adj_matrix[i, :v.shape[0], :v.shape[0]] = torch.FloatTensor(v)

            labels.append(item['LABEL'])

        compound_node_nums = torch.LongTensor(compound_node_nums)
        compound_smiles_length = torch.LongTensor(compound_smiles_nums)
        protein_node_nums = torch.LongTensor(protein_node_nums)
        protein_struct_node_nums = torch.LongTensor(protein_struct_node_nums)
        labels = torch.tensor(labels).type(torch.float32)

        return {
            'COMPOUND_NODE_FEAT': compound_node_features,
            'COMPOUND_ADJ': compound_adj_matrix,
            'COMPOUND_NODE_NUM': compound_node_nums,
            'COMPOUND_SMILES_LENGTH': compound_smiles_length,
            'COMPOUND_EMBEDDING': compound_embedding,
            # 'PROTEIN_CONTACT_MAP':protein_contact_map,
            'PROTEIN_EMBEDDING': protein_seq_embedding,
            'PROTEIN_NODE_NUM': protein_node_nums,
            'PROTEIN_NODE_FEAT': protein_node_features,
            'PROTEIN_ADJ': protein_adj_matrix,
            'PROTEIN_STRUCT_NODE_NUM': protein_struct_node_nums,
            'LABEL': labels,
        }



if __name__ == "__main__":
    galaxydb_data_path = 'data/'

    train_set = CPIDataset(galaxydb_data_path + 'davis.csv')

    #test_set = CPIDataset(galaxydb_data_path + 'davis_test.csv')

    item = train_set[3]
    print('Test Item:')
    print('Compound Node Feature Shape:', item['COMPOUND_NODE_FEAT'].shape)
    print('Compound Adjacency Matrix Shape:', item['COMPOUND_ADJ'].shape)
    print('Compound EMBEDDING Shape:', item['COMPOUND_EMBEDDING'].shape)
    print('Protein Embedding Shape:', item['PROTEIN_EMBEDDING'].shape)
    print('Protein Node Feature Shape:', item['PROTEIN_NODE_FEAT'].shape)
    print('Protein Adjacency Matrix Shape:', item['PROTEIN_ADJ'].shape)
    print('Label:', item['LABEL'])


    train_loader = DataLoader(
        train_set,
        batch_size=8,
        collate_fn=train_set.collate_fn,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    print('Test Batch:')
    for batch in train_loader:
        print('Test Batch Item:')
        print('Compound Node Feature Shape:', batch['COMPOUND_NODE_FEAT'].shape)
        print('Compound Adjacency Matrix Shape:', batch['COMPOUND_ADJ'].shape)
        print('Compound Node Numbers Shape:', batch['COMPOUND_NODE_NUM'].shape)
        print('Protein Embedding Shape:', batch['PROTEIN_EMBEDDING'].shape)
        print('Protein Node Numbers Shape:', batch['PROTEIN_NODE_NUM'].shape)
        print('Protein Node Feature Shape:', batch['PROTEIN_NODE_FEAT'].shape)
        print('Protein Adjacency Matrix Shape:', batch['PROTEIN_ADJ'].shape)
        print('Protein Struct Node Numbers Shape:', batch['PROTEIN_STRUCT_NODE_NUM'].shape)
        print('Label Shape:', batch['LABEL'].shape)
        break