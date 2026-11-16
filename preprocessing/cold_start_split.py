import pandas as pd
import argparse

# referenced from https://tdc.readthedocs.io/en/main/index.html
def create_fold(df, fold_seed, frac):
    """create random split

    Args:
        df (pd.DataFrame): dataset dataframe
        fold_seed (int): the random seed
        frac (list): a list of train/valid/test fractions

    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(
        frac=val_frac / (1 - test_frac), replace=False, random_state=1
    )
    train = train_val[~train_val.index.isin(val.index)]

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def create_fold_setting_cold(df, fold_seed, frac, entities):
    """create cold-split where given one or multiple columns, it first splits based on
    entities in the columns and then maps all associated data points to the partition

    Args:
            df (pd.DataFrame): dataset dataframe
            fold_seed (int): the random seed
            frac (list): a list of train/valid/test fractions
            entities (Union[str, List[str]]): either a single "cold" entity or a list of
                    "cold" entities on which the split is done

    Returns:
            dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    if isinstance(entities, str):
        entities = [entities]

    train_frac, val_frac, test_frac = frac

    # For each entity, sample the instances belonging to the test datasets
    test_entity_instances = [
        df[e]
        .drop_duplicates()
        .sample(frac=test_frac, replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]

    # Select samples where all entities are in the test set
    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]

    if len(test) == 0:
        raise ValueError(
            "No test samples found. Try another seed, increasing the test frac or a "
            "less stringent splitting strategy."
        )

    # Proceed with validation data
    train_val = df.copy()
    for i, e in enumerate(entities):
        train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

    val_entity_instances = [
        train_val[e]
        .drop_duplicates()
        .sample(frac=val_frac / (1 - test_frac), replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]
    val = train_val.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]

    if len(val) == 0:
        raise ValueError(
            "No validation samples found. Try another seed, increasing the test frac "
            "or a less stringent splitting strategy."
        )

    train = train_val.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def get_davis_split(df_or_path, split='random', seed=0, frac=[0.8, 0.1, 0.1]):
    """Return train, valid, test dataframes for Davis according to split protocol."""
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    if split == 'random':
        return create_fold(df, fold_seed=seed, frac=frac)
    elif split == 'cold_target':
        return create_fold_setting_cold(df, fold_seed=seed, frac=frac, entities=['target_sequence'])
    elif split == 'cold_drug':
        return create_fold_setting_cold(df, fold_seed=seed, frac=frac, entities=['compound_iso_smiles'])
    elif split == 'cold_pair':
        return create_fold_setting_cold(df, fold_seed=seed, frac=frac, entities=['target_sequence', 'compound_iso_smiles'])
    else:
        raise ValueError(f"Unknown split '{split}'. Expected one of ['random', 'cold_drug', 'cold_target', 'cold_pair']")


def compute_all_davis_split_stats(csv_path='data/davis.csv', seed=0):
    """Compute exact pair counts and entity overlap across all 4 Davis splits."""
    df = pd.read_csv(csv_path) if isinstance(csv_path, str) else csv_path
    splits = {
        'random': get_davis_split(df, split='random', seed=seed),
        'cold_drug': get_davis_split(df, split='cold_drug', seed=seed),
        'cold_target': get_davis_split(df, split='cold_target', seed=seed),
        'cold_pair': get_davis_split(df, split='cold_pair', seed=seed),
    }

    stats = {
        'dataset': 'davis',
        'seed': seed,
        'pkd_formula': '-log10(Kd / 1e9)',
        'pkd_range': [float(df['label'].min()), float(df['label'].max())],
        'total_dataset_pairs': len(df),
        'total_unique_drugs': int(df['compound_iso_smiles'].nunique()),
        'total_unique_targets': int(df['target_sequence'].nunique()),
        'total_unique_target_ids': int(df['target_id'].nunique()),
        'splits': {}
    }

    for name, fold in splits.items():
        tr, va, te = fold['train'], fold['valid'], fold['test']
        tot = len(tr) + len(va) + len(te)
        disc = len(df) - tot

        tr_drugs, va_drugs, te_drugs = set(tr['compound_iso_smiles']), set(va['compound_iso_smiles']), set(te['compound_iso_smiles'])
        tr_targets, va_targets, te_targets = set(tr['target_sequence']), set(va['target_sequence']), set(te['target_sequence'])

        stats['splits'][name] = {
            'pairs': {
                'train': len(tr),
                'valid': len(va),
                'test': len(te),
                'total_retained': tot,
                'discarded': disc,
            },
            'drugs': {
                'train': int(tr['compound_iso_smiles'].nunique()),
                'valid': int(va['compound_iso_smiles'].nunique()),
                'test': int(te['compound_iso_smiles'].nunique()),
            },
            'targets': {
                'train': int(tr['target_sequence'].nunique()),
                'valid': int(va['target_sequence'].nunique()),
                'test': int(te['target_sequence'].nunique()),
            },
            'target_ids': {
                'train': int(tr['target_id'].nunique()),
                'valid': int(va['target_id'].nunique()),
                'test': int(te['target_id'].nunique()),
            },
            'overlap': {
                'drug_train_test': len(tr_drugs.intersection(te_drugs)),
                'drug_train_val': len(tr_drugs.intersection(va_drugs)),
                'drug_val_test': len(va_drugs.intersection(te_drugs)),
                'target_train_test': len(tr_targets.intersection(te_targets)),
                'target_train_val': len(tr_targets.intersection(va_targets)),
                'target_val_test': len(va_targets.intersection(te_targets)),
            }
        }
    return stats


dataset_name = 'davis'



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data_path', type=str, default='../data', help='Raw Data Path')
    parser.add_argument("--dataset_name", type=str, default=dataset_name)
    parser.add_argument("--SEED", type=int, default=0)

    args = parser.parse_args()
    # read data 
    df = pd.read_csv(f'{args.root_data_path}/{args.dataset_name}.csv')
    # create folds
    cold_target_fold = create_fold_setting_cold(df, args.SEED, [0.8, 0.1, 0.1], ['target_sequence'])
    cold_target_fold["train"].to_csv(args.root_data_path + '/' +args.dataset_name + "_cold_target_train_"+str(args.SEED)+".csv", index=False)
    cold_target_fold["valid"].to_csv(args.root_data_path + '/' +args.dataset_name + "_cold_target_valid_"+str(args.SEED)+".csv", index=False)
    cold_target_fold["test"].to_csv(args.root_data_path + '/' +args.dataset_name + "_cold_target_test_"+str(args.SEED)+".csv", index=False)
    print(args.dataset_name+ " cold_target_fold done!  the shape of train, valid, test are: ", cold_target_fold["train"].shape, cold_target_fold["valid"].shape, cold_target_fold["test"].shape)

    cold_drug_fold = create_fold_setting_cold(df, args.SEED, [0.8, 0.1, 0.1], ['compound_iso_smiles'])
    cold_drug_fold["train"].to_csv(args.root_data_path + '/' +args.dataset_name + "_cold_drug_train_"+str(args.SEED)+".csv", index=False)
    cold_drug_fold["valid"].to_csv(args.root_data_path + '/' +args.dataset_name + "_cold_drug_valid_"+str(args.SEED)+".csv", index=False)
    cold_drug_fold["test"].to_csv(args.root_data_path + '/' +args.dataset_name + "_cold_drug_test_"+str(args.SEED)+".csv", index=False)
    print(args.dataset_name+ " cold_drug_fold done!  the shape of train, valid, test are: ", cold_drug_fold["train"].shape, cold_drug_fold["valid"].shape, cold_drug_fold["test"].shape)

    cold_target_drug_fold = create_fold_setting_cold(df, args.SEED, [0.8, 0.1, 0.1], ['target_sequence', 'compound_iso_smiles'])
    cold_target_drug_fold["train"].to_csv(args.root_data_path + '/' +args.dataset_name + "_cold_target_drug_train_"+str(args.SEED)+".csv", index=False)
    cold_target_drug_fold["valid"].to_csv(args.root_data_path + '/' +args.dataset_name + "_cold_target_drug_valid_"+str(args.SEED)+".csv", index=False)
    cold_target_drug_fold["test"].to_csv(args.root_data_path + '/' +args.dataset_name + "_cold_target_drug_test_"+str(args.SEED)+".csv", index=False)
    print(args.dataset_name+ " cold_target_drug_fold done!  the shape of train, valid, test are: ", cold_target_drug_fold["train"].shape, cold_target_drug_fold["valid"].shape, cold_target_drug_fold["test"].shape)
