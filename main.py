import sys, os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import argparse
from data import CPIDataset
from preprocessing.cold_start_split import get_davis_split, compute_all_davis_split_stats
from leakage_report import deduplicate_train_against_eval
from models.core import *
from utils import *
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, accuracy_score
import random
import torch.backends.cudnn as cudnn
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_device_and_model(model, gpus_arg='all', logger=None):
    """
    Configure compute device and DataParallel wrapping.
    If multiple GPUs are available and requested, wrap model with torch.nn.DataParallel.
    """
    if gpus_arg == 'cpu' or not torch.cuda.is_available():
        dev = torch.device('cpu')
        model = model.to(dev)
        if logger:
            logger.info("Using device: CPU")
        return model, dev

    num_available = torch.cuda.device_count()
    if gpus_arg == 'all':
        device_ids = list(range(num_available))
    elif isinstance(gpus_arg, str) and ',' in gpus_arg:
        device_ids = [int(x.strip()) for x in gpus_arg.split(',') if x.strip().isdigit()]
    elif isinstance(gpus_arg, (int, str)) and str(gpus_arg).strip().isdigit():
        device_ids = [int(str(gpus_arg).strip())]
    else:
        device_ids = [0]

    device_ids = [d for d in device_ids if d < num_available]
    if not device_ids:
        device_ids = [0]

    primary_device = torch.device(f'cuda:{device_ids[0]}')
    model = model.to(primary_device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        if logger:
            logger.info(f"Using DataParallel across {len(device_ids)} GPUs: {device_ids}")
    else:
        if logger:
            logger.info(f"Using single GPU: cuda:{device_ids[0]}")

    return model, primary_device


def save_model_checkpoint(model, path):
    """Save model state dict, unwrapping DataParallel or any module-wrapped model."""
    raw_model = getattr(model, 'module', model)
    torch.save(raw_model.state_dict(), path)


def load_model_checkpoint(model, path, device):
    """Load model state dict, unwrapping DataParallel if necessary and handling 'module.' prefix."""
    raw_model = getattr(model, 'module', model)
    state_dict = torch.load(path, map_location=device)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v
    raw_model.load_state_dict(cleaned_state_dict)


# training function at each epoch
def train_dta(model, loss_fn, train_loader, optimizer, epoch, device=None):
    if device is None:
        device = next(model.parameters()).device
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    model.train()

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        # print(data.y.view(-1, 1).float().to(device))

        loss = loss_fn(output, data['LABEL'].view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data['LABEL']),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting_dta(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data['LABEL'].view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


def train_dti(model, loss_fn, train_loader, optimizer, epoch, device=None):
    if device is None:
        device = next(model.parameters()).device
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    model.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data['LABEL'].to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data['LABEL']),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting_dti(model, loader):
    model.eval()
    total_preds = []
    total_labels = []

    print('Making predictions for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            output = model(data)
            total_preds.append(output.cpu().numpy())
            total_labels.append(data['LABEL'].cpu().numpy())

    return np.concatenate(total_labels), np.concatenate(total_preds)


DATASET_ALIASES = {
    'davis': 'davis',
    'pdb': 'pdbbind',
    'pdbbind': 'pdbbind',
    'tdc': 'tdc_dg',
    'tdc_dg': 'tdc_dg',
    'tdcdg': 'tdc_dg',
    'bindingdb': 'bindingdb',
    'bdb': 'bindingdb',
}


def resolve_dataset_file_prefix(dataset, data_path='./data'):
    """Resolve file prefix and directory name for a dataset, handling aliases."""
    dataset_norm = DATASET_ALIASES.get(dataset.lower(), dataset.lower())

    if dataset_norm == 'davis':
        csv_file = os.path.join(data_path, 'davis.csv')
        dir_prefix = os.path.join(data_path, 'davis')
        return csv_file, dir_prefix

    # Check CSV prefix
    if os.path.isfile(os.path.join(data_path, f'{dataset_norm}_train.csv')):
        csv_prefix = os.path.join(data_path, dataset_norm)
    elif dataset_norm == 'pdbbind' and os.path.isfile(os.path.join(data_path, 'pdb_train.csv')):
        csv_prefix = os.path.join(data_path, 'pdb')
    else:
        csv_prefix = os.path.join(data_path, dataset_norm)

    # Check directory prefix
    if os.path.isdir(os.path.join(data_path, dataset_norm)):
        dir_prefix = os.path.join(data_path, dataset_norm)
    elif dataset_norm == 'pdbbind' and os.path.isdir(os.path.join(data_path, 'pdb')):
        dir_prefix = os.path.join(data_path, 'pdb')
    else:
        dir_prefix = os.path.join(data_path, dataset_norm)

    return csv_prefix, dir_prefix


def davis_dataloader(seed=0, batch_size=128, workers=4, dataset='davis', data_path='./data', split='random', dedup='none', strict=False):
    print(f'\nrunning on {dataset} with split: {split}, dedup: {dedup}')

    path = data_path + '/' + dataset
    csv_file = f'{path}.csv'

    folds = get_davis_split(csv_file, split=split, seed=seed)
    if dedup == 'train':
        folds['train'], _ = deduplicate_train_against_eval(folds['train'], [folds['valid'], folds['test']])

    train_dataset = CPIDataset(folds['train'], f'{path}/compound', f'{path}/protein', strict=strict)
    val_dataset = CPIDataset(folds['valid'], f'{path}/compound', f'{path}/protein', strict=strict)
    test_dataset = CPIDataset(folds['test'], f'{path}/compound', f'{path}/protein', strict=strict)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )

    return train_loader, val_loader, test_loader

def others_dataloader(batch_size, workers, dataset='pdbbind', data_path='./data', seed=0, val_frac=0.1, dedup='none', strict=False):
    print(f'\nrunning on {dataset} (val_frac={val_frac}, dedup={dedup})')
    dataset_norm = DATASET_ALIASES.get(dataset.lower(), dataset.lower())
    csv_prefix, dir_prefix = resolve_dataset_file_prefix(dataset_norm, data_path)

    if dataset_norm == 'bindingdb':
        train_df = pd.read_csv(f'{csv_prefix}_train.csv')
        test_df = pd.read_csv(f'{csv_prefix}_test.csv')

        if dedup == 'train':
            train_df, _ = deduplicate_train_against_eval(train_df, [test_df])

        if val_frac > 0 and len(train_df) > 1:
            val_df = train_df.sample(frac=val_frac, replace=False, random_state=seed)
            train_df = train_df[~train_df.index.isin(val_df.index)].reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
        else:
            val_df = train_df.copy()

        train_set = CPIDataset(train_df, f'{dir_prefix}/compound', f'{dir_prefix}/protein/train', strict=strict)
        val_set = CPIDataset(val_df, f'{dir_prefix}/compound', f'{dir_prefix}/protein/train', strict=strict)
        test_set = CPIDataset(test_df, f'{dir_prefix}/compound', f'{dir_prefix}/protein/test', strict=strict)

    else:
        train_df = pd.read_csv(f'{csv_prefix}_train.csv')
        test_df = pd.read_csv(f'{csv_prefix}_test.csv')

        if dedup == 'train':
            train_df, _ = deduplicate_train_against_eval(train_df, [test_df])

        if val_frac > 0 and len(train_df) > 1:
            val_df = train_df.sample(frac=val_frac, replace=False, random_state=seed)
            train_df = train_df[~train_df.index.isin(val_df.index)].reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
        else:
            val_df = train_df.copy()

        train_set = CPIDataset(train_df, f'{dir_prefix}/compound', f'{dir_prefix}/protein', strict=strict)
        val_set = CPIDataset(val_df, f'{dir_prefix}/compound', f'{dir_prefix}/protein', strict=strict)
        test_set = CPIDataset(test_df, f'{dir_prefix}/compound', f'{dir_prefix}/protein', strict=strict)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=train_set.collate_fn,
        shuffle=True,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=val_set.collate_fn,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        collate_fn=test_set.collate_fn,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )

    return train_loader, val_loader, test_loader



def validate_dataset_prerequisites(dataset, data_path='./data', strict=False):
    """
    Validate that required raw data and preprocessed files exist for the dataset.
    If missing, print a descriptive error banner with the exact preprocessing commands
    required to generate them, and exit cleanly (or raise under --strict).
    """
    missing = []
    dataset_norm = DATASET_ALIASES.get(dataset.lower(), dataset.lower())
    csv_prefix, dir_prefix = resolve_dataset_file_prefix(dataset_norm, data_path)

    # 1. Raw CSV check
    if dataset_norm == 'davis':
        raw_csv = os.path.join(data_path, 'davis.csv')
        if not os.path.isfile(raw_csv):
            missing.append(f"Raw dataset CSV: {raw_csv}")
    else:
        train_csv = f'{csv_prefix}_train.csv'
        test_csv = f'{csv_prefix}_test.csv'
        if not os.path.isfile(train_csv):
            missing.append(f"Raw train CSV: {train_csv}")
        if not os.path.isfile(test_csv):
            missing.append(f"Raw test CSV: {test_csv}")

    # 2. Compound features
    mol_dict = os.path.join(dir_prefix, 'compound', 'mol_dict.pkl')
    if not os.path.isfile(mol_dict):
        missing.append(f"Pretrained ChemBERTa embeddings: {mol_dict}")

    # 3. Protein sequence embeddings
    prot_dir = os.path.join(dir_prefix, 'protein')
    if dataset_norm == 'bindingdb':
        prot_dir = os.path.join(dir_prefix, 'protein', 'train')
    if not os.path.isdir(prot_dir) or not any(f.endswith('.npy') for f in os.listdir(prot_dir) if os.path.isfile(os.path.join(prot_dir, f))):
        missing.append(f"Pretrained ESM-2 sequence embeddings directory: {prot_dir}")

    # 4. Protein structure contact graphs
    struct_dir = os.path.join(dir_prefix, 'protein', 'structure')
    if dataset_norm == 'bindingdb':
        struct_dir = os.path.join(dir_prefix, 'protein', 'train', 'structure')
    if not os.path.isdir(struct_dir) or not any(f.endswith('_node.npy') for f in os.listdir(struct_dir) if os.path.isfile(os.path.join(struct_dir, f))):
        missing.append(f"Protein structure contact graphs directory: {struct_dir}")

    if missing:
        task = 'dti' if dataset_norm == 'bindingdb' else 'dta'
        banner = [
            "=" * 80,
            f"PREREQUISITE CHECK FAILED: Missing Preprocessed Files for Dataset '{dataset}'",
            "=" * 80,
            "The following required data files were not found:",
        ]
        for m in missing:
            banner.append(f"  - {m}")
        banner.extend([
            "",
            "To generate the missing preprocessed files, run the preprocessing pipeline:",
            f"  1. ChemBERTa compound embeddings:",
            f"     python preprocessing/compound_pretrain.py --root_data_path {data_path} --dataset {dataset}",
            f"  2. ESM-2 protein sequence embeddings:",
            f"     python preprocessing/protein_pretrain/protein_{task}.py --root_data_path {data_path} --dataset {dataset}",
            f"  3. AlphaFold / RCSB protein contact graphs:",
            f"     python preprocessing/protein_pretrain/protein_structure.py --root_data_path {data_path} --dataset {dataset}",
            "=" * 80,
        ])
        msg = "\n".join(banner)
        print(msg)
        if strict:
            raise FileNotFoundError(msg)
        return False, missing
    return True, []



def run(args: argparse.Namespace):
    data_path = args.root_data_path
    dataset = args.dataset
    batch_size = args.batch_size
    LR = args.learning_rate
    NUM_EPOCHS = args.max_epochs
    seed = args.seed
    split_name = getattr(args, 'split', 'random')
    variant_name = getattr(args, 'variant', 'full')
    strict = getattr(args, 'strict', False)

    set_seed(seed)

    # Prerequisite validation
    if not getattr(args, 'skip_pre_check', False):
        ok, missing = validate_dataset_prerequisites(dataset, data_path, strict=strict)
        if not ok:
            sys.exit(1)

    run_dir = setup_run_dir(args.output_dir, dataset, split_name, variant_name)
    save_run_config(run_dir, args)
    logger = setup_logger(run_dir)

    logger.info(f"Initialized BiSSA run directory: {run_dir}")
    logger.info(f"Args: dataset={dataset}, split={split_name}, variant={variant_name}, objective={args.objective}, seed={seed}, lr={LR}, epochs={NUM_EPOCHS}")

    # Graph coverage check & reporting
    coverage_file = os.path.join(data_path, dataset, 'graph_coverage.json')
    if os.path.isfile(coverage_file):
        with open(coverage_file, 'r', encoding='utf-8') as f:
            cov = json.load(f)
        logger.info(f"[Graph Coverage] {dataset}: {cov.get('resolved', 0)}/{cov.get('total_proteins', 0)} resolved "
                    f"({(1.0 - cov.get('fallback_rate', 0.0)):.1%}), {cov.get('fell_back', 0)} fell back ({cov.get('fallback_rate', 0.0):.1%})")
        if strict and cov.get('fell_back', 0) > 0:
            raise RuntimeError(f"Dataset '{dataset}' has {cov['fell_back']} proteins requiring fallback under --strict.")

    # Build model and setup device / DataParallel
    model = BiSSANet(args)
    model, device = setup_device_and_model(model, getattr(args, 'gpus', 'all'), logger)

    # Load data
    if dataset == 'davis':
        train_loader, val_loader, raw_test_loader = davis_dataloader(
            seed, batch_size, args.num_workers, dataset, data_path,
            split=split_name,
            dedup=getattr(args, 'dedup', 'none'),
            strict=strict,
        )
    else:
        train_loader, val_loader, raw_test_loader = others_dataloader(
            batch_size, args.num_workers, dataset, data_path,
            seed=seed,
            val_frac=getattr(args, 'val_frac', 0.1),
            dedup=getattr(args, 'dedup', 'none'),
            strict=strict,
        )

    # Check for invalid SMILES fallbacks during loading
    invalid_smiles = get_invalid_smiles_log()
    if invalid_smiles:
        logger.warning(f"[Compound Fallback] {len(invalid_smiles)} invalid SMILES encountered during loading.")

    # STRICT TEST-SET ISOLATION: wrap test loader in an IsolatedLoader
    test_loader = IsolatedLoader(raw_test_loader, name=f"test_loader({dataset})")

    # Record split stats for this run
    run_split_stats = {
        'dataset': dataset,
        'split': split_name,
        'seed': seed,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'test_samples': len(test_loader),
    }
    with open(os.path.join(run_dir, 'split_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(run_split_stats, f, indent=2)

    logger.info(f"Dataset partitions: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader)}")

    best_model_path = os.path.join(run_dir, 'best_model.pth')

    if args.objective == 'regression':
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=30, eps=1e-08
        )
        best_val_mae = float('inf')
        best_epoch = -1

        for epoch in range(NUM_EPOCHS):
            train_dta(model, loss_fn, train_loader, optimizer, epoch + 1)

            # Strict validation evaluation during training loop (test set is locked)
            G_val, P_val = predicting_dta(model, val_loader)
            val_rmse = rmse(G_val, P_val)
            val_mae = mae(P_val, G_val)
            val_mse = mse(G_val, P_val)
            val_r = pearson(G_val, P_val)
            val_s = spearman(G_val, P_val)
            val_c = ci(G_val, P_val)
            val_r2m = rm2(G_val, P_val)

            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val] - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MSE: {val_mse:.4f}, "
                f"Pearson: {val_r:.4f}, Spearman: {val_s:.4f}, CI: {val_c:.4f}, r2m: {val_r2m:.4f}, LR: {current_lr:.6f}"
            )

            # Checkpoint selection strictly on lowest validation MAE
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch + 1
                save_model_checkpoint(model, best_model_path)
                logger.info(f"  --> Best Val MAE improved to {best_val_mae:.4f} at epoch {best_epoch}. Saved {best_model_path}")

            scheduler.step(val_mae)

        # Final test evaluation once from best-val checkpoint
        logger.info(f"Training completed. Loading best checkpoint from epoch {best_epoch} (Val MAE: {best_val_mae:.4f}) for final test evaluation...")
        test_loader.unlock()
        if os.path.exists(best_model_path):
            load_model_checkpoint(model, best_model_path, device)
        model.eval()

        G_test, P_test = predicting_dta(model, test_loader)
        final_test_metrics = {
            'dataset': dataset,
            'split': split_name,
            'variant': variant_name,
            'objective': args.objective,
            'best_epoch': int(best_epoch),
            'best_val_mae': float(best_val_mae),
            'test_rmse': float(rmse(G_test, P_test)),
            'test_mae': float(mae(P_test, G_test)),
            'test_mse': float(mse(G_test, P_test)),
            'test_pearson': float(pearson(G_test, P_test)),
            'test_spearman': float(spearman(G_test, P_test)),
            'test_ci': float(ci(G_test, P_test)),
            'test_rm2': float(rm2(G_test, P_test)),
        }
        test_metrics_path = os.path.join(run_dir, 'final_test_metrics.json')
        with open(test_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(final_test_metrics, f, indent=2)
        logger.info(f"Final Test Metrics: {json.dumps(final_test_metrics, indent=2)}")

    else:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=30, eps=1e-08
        )
        best_val_aupr = -1.0
        best_epoch = -1

        for epoch in range(NUM_EPOCHS):
            train_dti(model, loss_fn, train_loader, optimizer, epoch + 1)

            # Strict validation evaluation during training loop (test set is locked)
            y_true_v, y_scores_v = predicting_dti(model, val_loader)
            y_prob_v = torch.softmax(torch.tensor(y_scores_v), dim=1)[:, 1].numpy()
            y_pred_v = np.argmax(y_scores_v, axis=1)

            precision_v, recall_v, _ = precision_recall_curve(y_true_v, y_prob_v)
            val_aupr = float(auc(recall_v, precision_v))
            fpr_v, tpr_v, _ = roc_curve(y_true_v, y_prob_v)
            val_auroc = float(auc(fpr_v, tpr_v))
            val_f1 = float(f1_score(y_true_v, y_pred_v))
            val_acc = float(accuracy_score(y_true_v, y_pred_v))

            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val] - AUPR: {val_aupr:.4f}, AUROC: {val_auroc:.4f}, "
                f"Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, LR: {current_lr:.6f}"
            )

            # Checkpoint selection strictly on highest validation AUPR
            if val_aupr > best_val_aupr:
                best_val_aupr = val_aupr
                best_epoch = epoch + 1
                save_model_checkpoint(model, best_model_path)
                logger.info(f"  --> Best Val AUPR improved to {best_val_aupr:.4f} at epoch {best_epoch}. Saved {best_model_path}")

            scheduler.step(val_aupr)

        # Final test evaluation once from best-val checkpoint
        logger.info(f"Training completed. Loading best checkpoint from epoch {best_epoch} (Val AUPR: {best_val_aupr:.4f}) for final test evaluation...")
        test_loader.unlock()
        if os.path.exists(best_model_path):
            load_model_checkpoint(model, best_model_path, device)
        model.eval()

        y_true_t, y_scores_t = predicting_dti(model, test_loader)
        y_prob_t = torch.softmax(torch.tensor(y_scores_t), dim=1)[:, 1].numpy()
        y_pred_t = np.argmax(y_scores_t, axis=1)

        precision_t, recall_t, _ = precision_recall_curve(y_true_t, y_prob_t)
        test_aupr = float(auc(recall_t, precision_t))
        fpr_t, tpr_t, _ = roc_curve(y_true_t, y_prob_t)
        test_auroc = float(auc(fpr_t, tpr_t))
        test_f1 = float(f1_score(y_true_t, y_pred_t))
        test_acc = float(accuracy_score(y_true_t, y_pred_t))

        final_test_metrics = {
            'dataset': dataset,
            'split': split_name,
            'variant': variant_name,
            'objective': args.objective,
            'best_epoch': int(best_epoch),
            'best_val_aupr': float(best_val_aupr),
            'test_accuracy': float(test_acc),
            'test_aupr': float(test_aupr),
            'test_auroc': float(test_auroc),
            'test_f1': float(test_f1),
        }
        test_metrics_path = os.path.join(run_dir, 'final_test_metrics.json')
        with open(test_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(final_test_metrics, f, indent=2)
        logger.info(f"Final Test Metrics: {json.dumps(final_test_metrics, indent=2)}")



def parse_args(args=None):
    if args is not None:
        args = [str(a) for a in args]
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_data_path', type=str, default='./data', help='Raw Data Path')
    parser.add_argument('--dataset', type=str, default='davis',
                        help='Datasets: davis, pdbbind, tdc_dg, bindingdb (aliases: pdb, tdc, bdb)')
    parser.add_argument('--task', type=str, default=None, choices=['dta', 'dti'],
                        help='Task type: dta (regression) or dti (classification). Inferred from dataset if not provided.')
    parser.add_argument('--objective',
                        type=str,
                        default=None,
                        help='Objective (classification / regression). Inferred from task/dataset if omitted.')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch Size for Train(Validation/Test)')
    parser.add_argument('--epochs', '--max_epochs', dest='max_epochs', type=int, default=200, help='Max Training Epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of Subprocesses for Data Loading')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning Rate for Trainning')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of stacked DecoderLayer blocks in the drug and protein fusion decoders')
    parser.add_argument('--linear_heads', type=int, default=10, help='Number of linear attention heads')
    parser.add_argument('--linear_hidden_dim', type=int, default=32, help='Dimension of linear attention heads')
    parser.add_argument('--decoder_heads', type=int, default=4, help='Number of headers in the decoder')
    parser.add_argument('--encoder_heads', type=int, default=4, help='Number of Transformer heads')
    parser.add_argument('--gnn_layers', type=int, default=3, help='Layers of GNN')
    parser.add_argument('--encoder_layers', type=int, default=1, help='Layers of Transformer')
    parser.add_argument('--decoder_nums', type=int, default=1, help='Layers of Decoder')
    parser.add_argument('--decoder_dim', type=int, default=128, help='Dimension of Decoder')
    parser.add_argument('--compound_gnn_dim', type=int, default=78, help='Hidden Dimension for Attention')
    parser.add_argument('--pf_dim', type=int, default=1024, help='Hidden Dimension for Positional Feed Forward')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout Rate')
    parser.add_argument('--protein_dim', type=int, default=128, help='Dimension for Protein')
    parser.add_argument('--compound_structure_dim', type=int, default=78, help='Dimension for Compound Structure')
    parser.add_argument('--protein_structure_dim', type=int, default=22, help='Dimension for protein AlphaFold-graph node features (20 aa one-hot + unknown + pLDDT)')
    parser.add_argument('--compound_text_dim', type=int, default=128, help='Dimension for Compound Text')
    parser.add_argument('--compound_pretrained_dim', type=int, default=384, help='Dimension of pretrained for '
                                                                                 'compound language model')
    parser.add_argument('--protein_pretrained_dim', type=int, default=480, help='Dimension of pretrained for protein '
                                                                                'language model')
    parser.add_argument('--variant', type=str, default='full',
                        choices=['full', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
                        help='Ablation variant: full (BiSSA), A1 (graph-only), A2 (sequence-only), A3 (w/o protein graph), A4 (concat w/o cross-attn), A5 (S<-G only), A6 (G<-S only)')
    parser.add_argument('--split', type=str, default='random',
                        choices=['random', 'cold_drug', 'cold_target', 'cold_pair'],
                        help='Split setting for Davis dataset (random, cold_drug, cold_target, cold_pair)')
    parser.add_argument('--dedup', type=str, default='none',
                        choices=['none', 'train'],
                        help='Deduplication strategy: none (preserve published partition), train (remove pairs from training set that appear in test/val)')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Validation fraction carved from training partition for non-Davis datasets (default: 0.1)')
    parser.add_argument('--output_dir', type=str, default='runs',
                        help='Base directory for run outputs (config, logs, checkpoints, final metrics)')
    parser.add_argument('--gpus', type=str, default='all',
                        help='GPUs to use: "all", comma-separated IDs like "0,1,2,3", single ID like "0", or "cpu"')
    parser.add_argument('--strict', action='store_true', default=False,
                        help='Strict mode: raise error on invalid SMILES, missing structure files, or fallback data')
    parser.add_argument('--skip_pre_check', action='store_true', default=False,
                        help='Skip startup prerequisite file check (useful for synthetic smoke tests)')

    parsed = parser.parse_args(args)

    # Normalize dataset name
    dataset_raw = parsed.dataset.lower()
    parsed.dataset = DATASET_ALIASES.get(dataset_raw, dataset_raw)

    # Infer task and objective
    if parsed.task is None:
        if parsed.dataset == 'bindingdb':
            parsed.task = 'dti'
        else:
            parsed.task = 'dta'

    if parsed.objective is None:
        if parsed.task == 'dti':
            parsed.objective = 'classification'
        else:
            parsed.objective = 'regression'
    else:
        if parsed.task == 'dti' and parsed.objective != 'classification':
            parsed.objective = 'classification'
        elif parsed.task == 'dta' and parsed.objective != 'regression':
            parsed.objective = 'regression'

    return parsed

if __name__ == '__main__':
    params = parse_args()
    print(params)
    set_seed(params.seed)
    run(params)

