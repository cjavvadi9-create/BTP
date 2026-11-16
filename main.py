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

# training function at each epoch
def train_dta(model, loss_fn, train_loader, optimizer, epoch):
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


def train_dti(model, loss_fn, train_loader, optimizer, epoch):
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


def davis_dataloader(seed=0, batch_size=128, workers=4, dataset='davis', data_path='./data', split='random', dedup='none'):
    print(f'\nrunning on {dataset} with split: {split}, dedup: {dedup}')

    path = data_path + '/' + dataset
    csv_file = f'{path}.csv'

    folds = get_davis_split(csv_file, split=split, seed=seed)
    if dedup == 'train':
        folds['train'], _ = deduplicate_train_against_eval(folds['train'], [folds['valid'], folds['test']])

    train_dataset = CPIDataset(folds['train'], f'{path}/compound', f'{path}/protein')
    val_dataset = CPIDataset(folds['valid'], f'{path}/compound', f'{path}/protein')
    test_dataset = CPIDataset(folds['test'], f'{path}/compound', f'{path}/protein')

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

def others_dataloader(batch_size, workers, dataset='pdb', data_path='./data', seed=0, val_frac=0.1, dedup='none'):
    print(f'\nrunning on {dataset} (val_frac={val_frac}, dedup={dedup})')

    path = data_path + '/' + dataset

    if dataset == 'bindingdb':
        train_df = pd.read_csv(f'{path}_train.csv')
        test_df = pd.read_csv(f'{path}_test.csv')

        if dedup == 'train':
            train_df, _ = deduplicate_train_against_eval(train_df, [test_df])

        if val_frac > 0 and len(train_df) > 1:
            val_df = train_df.sample(frac=val_frac, replace=False, random_state=seed)
            train_df = train_df[~train_df.index.isin(val_df.index)].reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
        else:
            val_df = train_df.copy()

        train_set = CPIDataset(train_df, f'{path}/compound', f'{path}/protein/train')
        val_set = CPIDataset(val_df, f'{path}/compound', f'{path}/protein/train')
        test_set = CPIDataset(test_df, f'{path}/compound', f'{path}/protein/test')

    else:
        train_df = pd.read_csv(f'{path}_train.csv')
        test_df = pd.read_csv(f'{path}_test.csv')

        if dedup == 'train':
            train_df, _ = deduplicate_train_against_eval(train_df, [test_df])

        if val_frac > 0 and len(train_df) > 1:
            val_df = train_df.sample(frac=val_frac, replace=False, random_state=seed)
            train_df = train_df[~train_df.index.isin(val_df.index)].reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
        else:
            val_df = train_df.copy()

        train_set = CPIDataset(train_df, f'{path}/compound', f'{path}/protein')
        val_set = CPIDataset(val_df, f'{path}/compound', f'{path}/protein')
        test_set = CPIDataset(test_df, f'{path}/compound', f'{path}/protein')

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



def run(args: argparse.Namespace):
    data_path = args.root_data_path
    dataset = args.dataset
    batch_size = args.batch_size
    LR = args.learning_rate
    NUM_EPOCHS = args.max_epochs
    seed = args.seed
    split_name = getattr(args, 'split', 'random')
    variant_name = getattr(args, 'variant', 'full')

    set_seed(seed)

    run_dir = setup_run_dir(args.output_dir, dataset, split_name, variant_name)
    save_run_config(run_dir, args)
    logger = setup_logger(run_dir)

    logger.info(f"Initialized BiSSA run directory: {run_dir}")
    logger.info(f"Args: dataset={dataset}, split={split_name}, variant={variant_name}, objective={args.objective}, seed={seed}, lr={LR}, epochs={NUM_EPOCHS}")

    # Build model
    model = BiSSANet(args).to(device)

    # Load data
    if dataset == 'davis':
        train_loader, val_loader, raw_test_loader = davis_dataloader(
            seed, batch_size, args.num_workers, dataset, data_path,
            split=split_name,
            dedup=getattr(args, 'dedup', 'none'),
        )
    else:
        train_loader, val_loader, raw_test_loader = others_dataloader(
            batch_size, args.num_workers, dataset, data_path,
            seed=seed,
            val_frac=getattr(args, 'val_frac', 0.1),
            dedup=getattr(args, 'dedup', 'none'),
        )

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
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"  --> Best Val MAE improved to {best_val_mae:.4f} at epoch {best_epoch}. Saved {best_model_path}")

            scheduler.step(val_mae)

        # Final test evaluation once from best-val checkpoint
        logger.info(f"Training completed. Loading best checkpoint from epoch {best_epoch} (Val MAE: {best_val_mae:.4f}) for final test evaluation...")
        test_loader.unlock()
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
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
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"  --> Best Val AUPR improved to {best_val_aupr:.4f} at epoch {best_epoch}. Saved {best_model_path}")

            scheduler.step(val_aupr)

        # Final test evaluation once from best-val checkpoint
        logger.info(f"Training completed. Loading best checkpoint from epoch {best_epoch} (Val AUPR: {best_val_aupr:.4f}) for final test evaluation...")
        test_loader.unlock()
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_data_path', type=str, default='./data', help='Raw Data Path')
    parser.add_argument('--dataset', type=str, default='pdb', help='Datasets')
    parser.add_argument('--objective',
                        type=str,
                        default='regression',
                        help='Objective (classification / regression)')
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

    return parser.parse_args(args)

if __name__ == '__main__':
    params = parse_args()
    print(params)
    set_seed(params.seed)
    run(params)

