"""
leakage_report.py
-----------------
Audits data leakage across all benchmark datasets (BindingDB, TDC-DG, PDBbind, Davis splits).
Counts exact duplicate (canonical SMILES / compound_iso_smiles, sequence) pairs shared
between train and test (and val), plus within-split duplicate pairs carrying conflicting labels.
Outputs leakage_report.json.
"""

import sys
import os
import json
import pandas as pd

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from preprocessing.cold_start_split import get_davis_split


def analyze_dataset_leakage(name, train_df, test_df, val_df=None):
    """Analyze pair sharing and internal conflicting duplicates across splits."""
    def get_pairs_dict(df):
        pdict = {}
        for _, row in df.iterrows():
            pair = (str(row['compound_iso_smiles']).strip(), str(row['target_sequence']).strip())
            lbl = float(row['label'])
            pdict.setdefault(pair, []).append(lbl)
        return pdict

    tr_pairs = get_pairs_dict(train_df)
    te_pairs = get_pairs_dict(test_df)
    va_pairs = get_pairs_dict(val_df) if val_df is not None else {}

    # Shared pairs
    tr_te_shared = set(tr_pairs.keys()).intersection(set(te_pairs.keys()))
    tr_va_shared = set(tr_pairs.keys()).intersection(set(va_pairs.keys())) if val_df is not None else set()
    va_te_shared = set(va_pairs.keys()).intersection(set(te_pairs.keys())) if val_df is not None else set()

    # Conflicting labels within splits
    def find_conflicts(pdict):
        conflicts = 0
        duplicate_rows = 0
        for pair, lbls in pdict.items():
            if len(lbls) > 1:
                duplicate_rows += len(lbls) - 1
                if len(set(lbls)) > 1:
                    conflicts += 1
        return duplicate_rows, conflicts

    tr_dups, tr_conflicts = find_conflicts(tr_pairs)
    te_dups, te_conflicts = find_conflicts(te_pairs)
    va_dups, va_conflicts = (find_conflicts(va_pairs) if val_df is not None else (0, 0))

    report = {
        'dataset': name,
        'train_pairs_count': len(train_df),
        'test_pairs_count': len(test_df),
        'train_unique_pairs': len(tr_pairs),
        'test_unique_pairs': len(te_pairs),
        'train_test_shared_pairs': len(tr_te_shared),
        'test_leakage_rate_pct': round(len(tr_te_shared) / len(test_df) * 100, 2),
        'train_internal_duplicates': tr_dups,
        'train_conflicting_label_pairs': tr_conflicts,
        'test_internal_duplicates': te_dups,
        'test_conflicting_label_pairs': te_conflicts,
    }
    if val_df is not None:
        report['valid_pairs_count'] = len(val_df)
        report['valid_unique_pairs'] = len(va_pairs)
        report['train_val_shared_pairs'] = len(tr_va_shared)
        report['val_test_shared_pairs'] = len(va_te_shared)
        report['val_internal_duplicates'] = va_dups
        report['val_conflicting_label_pairs'] = va_conflicts

    return report


def generate_leakage_report(data_dir='data', output_file='leakage_report.json'):
    results = {}

    # 1. BindingDB
    bdb_tr_path = os.path.join(data_dir, 'bindingdb_train.csv')
    bdb_te_path = os.path.join(data_dir, 'bindingdb_test.csv')
    if os.path.exists(bdb_tr_path) and os.path.exists(bdb_te_path):
        results['bindingdb'] = analyze_dataset_leakage('bindingdb', pd.read_csv(bdb_tr_path), pd.read_csv(bdb_te_path))

    # 2. TDC-DG
    tdc_tr_path = os.path.join(data_dir, 'tdc_dg_train.csv')
    tdc_te_path = os.path.join(data_dir, 'tdc_dg_test.csv')
    if os.path.exists(tdc_tr_path) and os.path.exists(tdc_te_path):
        results['tdc_dg'] = analyze_dataset_leakage('tdc_dg', pd.read_csv(tdc_tr_path), pd.read_csv(tdc_te_path))

    # 3. PDBbind
    pdb_tr_path = os.path.join(data_dir, 'pdb_train.csv')
    pdb_te_path = os.path.join(data_dir, 'pdb_test.csv')
    if os.path.exists(pdb_tr_path) and os.path.exists(pdb_te_path):
        results['pdbbind'] = analyze_dataset_leakage('pdbbind', pd.read_csv(pdb_tr_path), pd.read_csv(pdb_te_path))

    # 4. Davis splits
    davis_path = os.path.join(data_dir, 'davis.csv')
    if os.path.exists(davis_path):
        davis_df = pd.read_csv(davis_path)
        for s in ['random', 'cold_drug', 'cold_target', 'cold_pair']:
            folds = get_davis_split(davis_df, split=s, seed=0)
            results[f'davis_{s}'] = analyze_dataset_leakage(f'davis_{s}', folds['train'], folds['test'], folds['valid'])

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print("DATA LEAKAGE & DUPLICATE AUDIT REPORT")
    print("=" * 80)
    col1 = 18
    col2 = 12
    col3 = 12
    col4 = 16
    col5 = 14
    header = f"| {'Dataset / Split':<{col1}} | {'Train Pairs':<{col2}} | {'Test Pairs':<{col3}} | {'Shared Pairs':<{col4}} | {'Leakage %':<{col5}} |"
    sep = f"|{'-'*(col1+2)}|{'-'*(col2+2)}|{'-'*(col3+2)}|{'-'*(col4+2)}|{'-'*(col5+2)}|"
    print(header)
    print(sep)
    for k, v in results.items():
        print(f"| {k:<{col1}} | {v['train_pairs_count']:<{col2},} | {v['test_pairs_count']:<{col3},} | {v['train_test_shared_pairs']:<{col4},} | {v['test_leakage_rate_pct']:<{col5}.2f}% |")
    print(sep)
    print(f"\nSaved detailed leakage report to: {output_file}\n")
    return results


def deduplicate_train_against_eval(train_df, eval_dfs, logger=None):
    """
    Remove from train_df all pairs (compound_iso_smiles, target_sequence)
    that appear in any eval_df (val or test). Never modifies eval_df.
    """
    eval_pairs = set()
    for edf in eval_dfs:
        if edf is not None and len(edf) > 0:
            for _, row in edf.iterrows():
                eval_pairs.add((str(row['compound_iso_smiles']).strip(), str(row['target_sequence']).strip()))

    orig_len = len(train_df)
    mask = train_df.apply(
        lambda r: (str(r['compound_iso_smiles']).strip(), str(r['target_sequence']).strip()) not in eval_pairs,
        axis=1
    )
    deduped_train_df = train_df[mask].reset_index(drop=True)
    removed_count = orig_len - len(deduped_train_df)
    msg = f"[DEDUP] Removed {removed_count} leaked pairs from training set ({orig_len} -> {len(deduped_train_df)}). Test/Val untouched."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return deduped_train_df, removed_count


if __name__ == '__main__':
    generate_leakage_report()
