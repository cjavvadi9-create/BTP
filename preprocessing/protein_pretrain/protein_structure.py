# Requirements: biopython>=1.79, requests>=2.28
# Install via: pip install biopython requests

import os
import os.path
import logging
import argparse

import numpy as np
import pandas as pd
import requests
from Bio import PDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_AMINO_ACIDS = [
    'A', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y',
    'Unknown',
]  # 20 standard + 1 unknown bucket  ->  length 21
# Node feature width: 21 (one-hot AA) + 1 (normalised confidence) = 22

ALPHAFOLD_URL_TEMPLATE = (
    "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
)

RCSB_URL_TEMPLATE = (
    "https://files.rcsb.org/download/{pdb_id}.pdb"
)

CONTACT_THRESHOLD_ANGSTROM = 8.0

# Fixed confidence score for experimental structures (no pLDDT available).
# 90/100 = 0.9 after normalisation — high but not maximum, reflecting that
# experimental coordinates are reliable but may have local disorder.
EXPERIMENTAL_CONFIDENCE = 90.0


# ---------------------------------------------------------------------------
# Helper - mirrors one_of_k_encoding_unk from compound.py
# ---------------------------------------------------------------------------

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# ---------------------------------------------------------------------------
# 1a. Fetch from RCSB PDB (for PDB-code target IDs, e.g. PDBbind)
# ---------------------------------------------------------------------------

def fetch_rcsb_structure(pdb_id, cache_dir):
    """Download the experimental PDB file from RCSB and cache it.

    Returns the path to the cached PDB file, or None on failure.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    pdb_path = os.path.join(cache_dir, f"{pdb_id}_rcsb.pdb")

    if os.path.isfile(pdb_path):
        return pdb_path

    url = RCSB_URL_TEMPLATE.format(pdb_id=pdb_id.upper())
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 404:
            logger.warning(
                "RCSB structure not found for %s (HTTP 404).", pdb_id
            )
            return None
        response.raise_for_status()
        with open(pdb_path, "wb") as f:
            f.write(response.content)
        return pdb_path
    except requests.RequestException as e:
        logger.warning(
            "Failed to download RCSB structure for %s: %s.", pdb_id, e
        )
        return None


# ---------------------------------------------------------------------------
# 1b. Fetch AlphaFold PDB (for UniProt-ID target IDs, e.g. Davis)
# ---------------------------------------------------------------------------

def fetch_alphafold_structure(uniprot_id, cache_dir):
    """Download the AlphaFold v4 model PDB for *uniprot_id* and cache it.

    Returns the path to the cached PDB file, or None if the protein is not
    present in the AlphaFold DB (HTTP 404) or if any other network error
    occurs.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"{cache_dir} created")

    pdb_path = os.path.join(cache_dir, f"{uniprot_id}.pdb")

    if os.path.isfile(pdb_path):
        return pdb_path

    url = ALPHAFOLD_URL_TEMPLATE.format(uniprot_id=uniprot_id)
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 404:
            logger.warning(
                "AlphaFold structure not found for %s (HTTP 404). "
                "Falling back to sequence-only features.",
                uniprot_id,
            )
            return None
        response.raise_for_status()
        with open(pdb_path, "wb") as f:
            f.write(response.content)
        return pdb_path
    except requests.RequestException as e:
        logger.warning(
            "Failed to download AlphaFold structure for %s: %s. "
            "Falling back to sequence-only features.",
            uniprot_id,
            e,
        )
        return None


# ---------------------------------------------------------------------------
# 2. Parse Ca coordinates + B-factors / pLDDT from PDB
# ---------------------------------------------------------------------------

def _parse_chain_ca(chain):
    """Extract Ca coordinates and B-factors from a single chain."""
    ca_coords = []
    bfactors = []
    for residue in chain.get_residues():
        if residue.get_id()[0] != ' ':
            continue
        if 'CA' not in residue:
            continue
        ca_atom = residue['CA']
        ca_coords.append(ca_atom.get_vector().get_array())
        bfactors.append(ca_atom.get_bfactor())
    return (
        np.array(ca_coords, dtype=np.float32),
        np.array(bfactors, dtype=np.float32),
    )


def parse_ca_coordinates(pdb_path, target_seq_len=None):
    """Extract Ca coordinates and B-factors, picking the best-matching chain.

    If *target_seq_len* is provided, selects the chain whose residue count
    is closest to target_seq_len.  Otherwise uses the first chain.

    Returns
    -------
    ca_coords : np.ndarray, shape (L, 3)
    bfactors  : np.ndarray, shape (L,)
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = next(structure.get_models())

    if target_seq_len is not None:
        best_coords, best_bf = None, None
        best_diff = float('inf')
        for chain in model.get_chains():
            coords, bf = _parse_chain_ca(chain)
            if len(coords) == 0:
                continue
            diff = abs(len(coords) - target_seq_len)
            if diff < best_diff:
                best_diff = diff
                best_coords, best_bf = coords, bf
        if best_coords is not None:
            return best_coords, best_bf
        # Fallback to first chain if nothing matched
        chain = next(model.get_chains())
        return _parse_chain_ca(chain)
    else:
        chain = next(model.get_chains())
        return _parse_chain_ca(chain)


# ---------------------------------------------------------------------------
# 3. Residue node features
# ---------------------------------------------------------------------------

def residue_features(sequence, confidence_scores):
    """One-hot encode residues (20 AA + Unknown) and append normalised confidence.

    Parameters
    ----------
    sequence          : str, protein sequence of length L
    confidence_scores : np.ndarray, shape (L,), values in [0, 100]

    Returns
    -------
    np.ndarray, shape (L, 22)
    """
    features = []
    for i, aa in enumerate(sequence):
        one_hot = one_of_k_encoding_unk(aa, STANDARD_AMINO_ACIDS)
        norm_conf = float(confidence_scores[i]) / 100.0
        features.append(np.array(one_hot + [norm_conf], dtype=np.float32))
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# 4. Contact graph (Ca distance matrix binarised at threshold)
# ---------------------------------------------------------------------------

def build_contact_graph(ca_coords, threshold=CONTACT_THRESHOLD_ANGSTROM):
    """Pairwise Ca distance matrix binarised at *threshold* Angstroms with self-loops.

    Mirrors adjacent_matrix() from compound.py:
        return np.array(adjacency) + np.eye(adjacency.shape[0])

    Parameters
    ----------
    ca_coords : np.ndarray, shape (L, 3)
    threshold : float, distance cutoff in Angstroms

    Returns
    -------
    np.ndarray, shape (L, L)  - binary adjacency with self-loops
    """
    L = ca_coords.shape[0]
    # Squared distances via broadcasting: (L,1,3) - (1,L,3)
    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))        # (L, L)
    adjacency = (dist_matrix < threshold).astype(np.float32)
    # Remove self-loops created by the distance threshold (dist=0 < threshold)
    # then re-add them uniformly via np.eye - exactly as adjacent_matrix() does
    adjacency = adjacency - np.eye(L, dtype=np.float32)
    return adjacency + np.eye(L, dtype=np.float32)


# ---------------------------------------------------------------------------
# 5. Orchestrator
# ---------------------------------------------------------------------------

def _looks_like_pdb_code(target_id):
    """Heuristic: 4-char alphanumeric strings are likely PDB codes."""
    return len(target_id) == 4 and target_id.isalnum()


def get_protein_structure_features(target_id, sequence, cache_dir):
    """Fetch structure (RCSB or AlphaFold) and return (node_features, adjacency).

    Strategy:
    1. If target_id looks like a PDB code (4 chars), try RCSB PDB first.
       For experimental structures, confidence is set to a fixed high value.
    2. Otherwise try AlphaFold (expects UniProt accession IDs).
       pLDDT is read from the B-factor column.
    3. Falls back gracefully when no structure is available:
       - adjacency  -> identity matrix (self-loops only)
       - confidence -> 0 for every residue

    Parameters
    ----------
    target_id : str, PDB code or UniProt accession
    sequence  : str, amino-acid sequence of length L
    cache_dir : str, directory for caching downloaded PDB files

    Returns
    -------
    node_features : np.ndarray, shape (L, 22)
    adjacency     : np.ndarray, shape (L, L)
    """
    L = len(sequence)
    pdb_path = None
    is_experimental = False

    # --- Try RCSB PDB first (for PDB codes) ---
    if _looks_like_pdb_code(target_id):
        pdb_path = fetch_rcsb_structure(target_id, cache_dir)
        if pdb_path is not None:
            is_experimental = True

    # --- Try AlphaFold (for UniProt IDs) ---
    if pdb_path is None:
        pdb_path = fetch_alphafold_structure(target_id, cache_dir)

    # --- Fallback: no structure available ---
    if pdb_path is None:
        print(
            f"[WARNING] No structure found for target_id={target_id}. "
            "Using identity adjacency and zero confidence."
        )
        conf_scores = np.zeros(L, dtype=np.float32)
        node_features = residue_features(sequence, conf_scores)
        adjacency = np.eye(L, dtype=np.float32)
        return node_features, adjacency

    # --- Parse structure ---
    try:
        ca_coords, raw_scores = parse_ca_coordinates(pdb_path, target_seq_len=L)
    except Exception as e:
        print(
            f"[WARNING] Failed to parse PDB for target_id={target_id}: {e}. "
            "Using identity adjacency and zero confidence."
        )
        conf_scores = np.zeros(L, dtype=np.float32)
        node_features = residue_features(sequence, conf_scores)
        adjacency = np.eye(L, dtype=np.float32)
        return node_features, adjacency

    # For experimental structures, B-factors ≠ pLDDT; use fixed confidence.
    if is_experimental:
        conf_scores = np.full(len(raw_scores), EXPERIMENTAL_CONFIDENCE, dtype=np.float32)
    else:
        conf_scores = raw_scores  # AlphaFold pLDDT

    # Align length with target sequence (guard against edge-case truncation)
    if len(conf_scores) != L:
        logger.warning(
            "Sequence length (%d) != parsed residue count (%d) for %s. "
            "Padding / truncating to match sequence.",
            L, len(conf_scores), target_id,
        )
        if len(conf_scores) < L:
            pad_len = L - len(conf_scores)
            conf_scores = np.concatenate(
                [conf_scores, np.zeros(pad_len, dtype=np.float32)]
            )
            ca_coords_pad = np.zeros((L, 3), dtype=np.float32)
            ca_coords_pad[:len(ca_coords)] = ca_coords
            ca_coords = ca_coords_pad
        else:
            conf_scores = conf_scores[:L]
            ca_coords = ca_coords[:L]

    node_features = residue_features(sequence, conf_scores)
    adjacency = build_contact_graph(ca_coords)

    return node_features, adjacency


# ---------------------------------------------------------------------------
# 6. CLI entrypoint - mirrors generate_feature() in protein_dta.py / protein_dti.py
# ---------------------------------------------------------------------------

def generate_feature(args):

    data_path = args.root_data_path
    dataset = args.dataset
    # BUG FIX: output path must be protein/structure/ to match data.py which
    # loads from {protein_data_path}/structure/{id}_node.npy
    output_data_path = os.path.join(data_path, dataset, 'protein', 'structure') + os.sep
    cache_dir = os.path.join(data_path, dataset, 'pdb_cache')

    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
        print(f"{output_data_path} created")
    else:
        print(f"{output_data_path} exists")

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
            target_id = str(raw_data['target_id'][i])
            if target_id in id_list:
                continue

            node_out = f'{output_data_path}{target_id}_node.npy'
            adj_out = f'{output_data_path}{target_id}_adj.npy'

            if os.path.isfile(node_out) and os.path.isfile(adj_out):
                id_list.append(target_id)
                continue

            node_features, adjacency = get_protein_structure_features(
                target_id, s, cache_dir
            )

            np.save(node_out, node_features)
            np.save(adj_out, adjacency)
            print(f"[{count+1}] {target_id}  node={node_features.shape}  adj={adjacency.shape}")

            id_list.append(target_id)
            count += 1

    print(f"\nDone — generated structure features for {count} proteins.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data_path', type=str, default='../../data', help='Raw Data Path')
    parser.add_argument('--dataset', type=str, default='davis', help='Datasets')
    return parser.parse_args()


if __name__ == '__main__':
    params = parse_args()
    print(params)
    generate_feature(params)
