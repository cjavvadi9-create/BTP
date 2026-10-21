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

    opts = ['train', 'test']

    # Pre-load the model before the loop
    _load_esm()

    for o in opts:

        if dataset == 'bindingdb':
            out_dir = output_data_path + o + '/'
        else:
            out_dir = output_data_path

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print(f"{out_dir} created")
        else:
            print(f"{out_dir} exists")

        count = 0
        id_list = []
        raw_data = pd.read_csv(f'{data_path}/{dataset}_{o}.csv')
        sequence_values = raw_data['target_sequence'].values
        for i, s in enumerate(sequence_values):
            id = str(raw_data['target_id'][i])
            if id in id_list:
                continue
            if os.path.isfile(out_dir + id + '.npy'):
                id_list.append(id)
                continue
            seq_emb = get_pretrained_embedding(s.upper())
            print(f"  [{o}][{count+1}] {id}: shape={seq_emb.shape}")
            np.save(out_dir + id, seq_emb)
            id_list.append(id)
            count += 1
        print(f"  [{o}] Done — {count} protein embeddings generated.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_data_path', type=str, default='../../data', help='Raw Data Path')
    parser.add_argument('--dataset', type=str, default='bindingdb', help='Datasets')
    return parser.parse_args()

if __name__ == '__main__':
    params = parse_args()
    print(params)
    generate_feature(params)
