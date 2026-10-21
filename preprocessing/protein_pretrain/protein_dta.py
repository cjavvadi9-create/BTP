import os.path
import numpy as np
import pandas as pd
import torch
import esm
import argparse
import os


# ---------------------------------------------------------------------------
# Load ESM-2 model ONCE (not per-sequence) for efficiency
# ---------------------------------------------------------------------------
_esm_model = None
_esm_alphabet = None
_esm_converter = None

def _load_esm():
    """Lazy-load ESM-2 model and alphabet (singleton)."""
    global _esm_model, _esm_alphabet, _esm_converter
    if _esm_model is None:
        print("Loading ESM-2 model (esm2_t12_35M_UR50D)...")
        _esm_model, _esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        _esm_converter = _esm_alphabet.get_batch_converter()
        _esm_model.eval()
        print("ESM-2 model loaded.")
    return _esm_model, _esm_alphabet, _esm_converter


def get_pretrained_embedding(s):
    if len(s) > 1022:
        s = s[:1022]
    model, alphabet, batch_converter = _load_esm()

    batch_labels, batch_strs, batch_tokens = batch_converter([("protein", s)])
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12], return_contacts=True)
    token_representations = results["representations"][12]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1])

    return sequence_representations[0]

def generate_feature(args):

    data_path = args.root_data_path
    dataset = args.dataset
    output_data_path = data_path + '/' + dataset + '/protein/'

    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
        print(f"{output_data_path} created")
    else:
        print(f"{output_data_path} exists")

    # Pre-load the model before the loop
    _load_esm()

    opt = ['train', 'test']

    id_list = []
    count = 0


    for t in opt:
        if dataset == 'davis':
            raw_data = pd.read_csv(f'{data_path}/{dataset}.csv')
        else:
            raw_data = pd.read_csv(f'{data_path}/{dataset}_{t}.csv')
        sequence_values = raw_data['target_sequence'].values
        for i, s in enumerate(sequence_values):
            id = raw_data['target_id'][i]
            if id in id_list:
                continue
            if os.path.isfile(f'{output_data_path}' + id + '.npy'):
                continue
            seq_emb = get_pretrained_embedding(s)
            print(f"  [{count+1}] {id}: shape={seq_emb.shape}")
            np.save(f'{output_data_path}' + id, seq_emb)
            id_list.append(id)
            count += 1
    print(f"\nDone — {count} protein embeddings generated.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_data_path', type=str, default='../../data', help='Raw Data Path')
    parser.add_argument('--dataset', type=str, default='davis', help='Datasets')
    return parser.parse_args()

if __name__ == '__main__':
    params = parse_args()
    print(params)
    generate_feature(params)
