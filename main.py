import argparse
import os
import json
import random
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, roc_auc_score

from model import TransMIL
from dataset import CustomDataset

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_subset(dataset, ids):
    idxs = [dataset.file_names.index(fid) for fid in ids if fid in dataset.file_names]
    return Subset(dataset, idxs)

def kl_divergence(p, q, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return np.mean(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))

def evaluate_model(model, loader, device, criterion, normal_dim, func_dim, output_dim):
    model.eval()
    running_loss = 0.0
    all_labels, all_outputs = [], []
    with torch.no_grad():
        for features, normal_lab, func_lab in loader:
            features = features.to(device)
            if output_dim == normal_dim:
                labels = normal_lab.to(device).float()
            elif output_dim == func_dim:
                labels = func_lab.to(device).float()
            else:
                raise ValueError(f"output_dim {output_dim} unsupported")
            outputs, _, _ = model(features)
            running_loss += criterion(outputs, labels).item() * features.size(0)
            all_labels .append(labels.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

    all_labels  = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)
    preds       = (all_outputs > 0.5).astype(int)

    f1_per    = f1_score(all_labels, preds, average=None)
    auc_per   = [roc_auc_score(all_labels[:,i], all_outputs[:,i])
                 if len(np.unique(all_labels[:,i]))>1 else np.nan
                 for i in range(all_labels.shape[1])]
    f1_macro  = np.nanmean(f1_per)
    auc_macro = np.nanmean(auc_per)
    acc_macro = np.nanmean((preds == all_labels).mean(axis=0))
    L         = all_labels.shape[1]
    kl_score  = kl_divergence(all_labels.sum(1)/L, preds.sum(1)/L)
    avg_loss  = running_loss / len(loader.dataset)

    return avg_loss, f1_macro, auc_macro, acc_macro, kl_score

def main(args):
    set_seed(args.seed)

    dims = {
        "resnet50": 2048,
        "vit_b_16": 768,
        "min": 2048,
        "mean": 2048,
        "max": 2048,
        "hadamard": 2048,
        "conc": 4096,
        "gated": 2048,
        "cross": 2048,
        "moco": 256,
        "geotwin": 256
    }
    feature_dir = f"./{args.feature_type}_{args.model_name}_feature_1024"
    input_dim   = dims[args.model_name]

    # Load splits
    train_ids = pd.read_csv(args.train_csv)['id'].astype(str).tolist()
    val_ids   = pd.read_csv(args.valid_csv)['id'].astype(str).tolist()
    test_ids  = pd.read_csv(args.test_csv)['id'].astype(str).tolist()

    ds = CustomDataset(feature_dir, args.label_dir)

    # determine normal vs function label dims
    _, normal0, func0 = ds[0]
    normal_dim = normal0.shape[0]
    func_dim   = func0.shape[0]

    # build function-label matrix for train split
    active = []
    for fid in train_ids:
        if fid in ds.file_names:
            with open(os.path.join(args.label_dir, f"{fid}.json"), 'r') as f:
                jd = json.load(f)
            func = np.array(list(jd['function_label'].values()), dtype=float)
            active.append((fid, func))
    if active:
        active_ids, mat_list = zip(*active)
        mat = np.array(mat_list)
    else:
        active_ids = []
        mat = np.empty((0, func_dim))

    # select train IDs by mode on function_label
    if args.mode == "zero":
        selected_ids = []
    elif args.mode == "few":
        sel = set()
        for cls_idx in range(mat.shape[1]):
            pos = [active_ids[i] for i, v in enumerate(mat[:, cls_idx]) if v == 1]
            k   = min(args.shot, len(pos))
            if k > 0:
                sel.update(random.sample(pos, k))
        selected_ids = list(sel)
    else:  # full
        selected_ids = list(active_ids)

    # DataLoaders
    train_loader = DataLoader(create_subset(ds, selected_ids),
                              batch_size=args.batch_size,
                              shuffle=True) if selected_ids else None
    val_loader   = DataLoader(create_subset(ds, val_ids),
                              batch_size=args.batch_size,
                              shuffle=False)
    test_loader  = DataLoader(create_subset(ds, test_ids),
                              batch_size=args.batch_size,
                              shuffle=False)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = TransMIL(args.output_dim, input_dim).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    best_state    = model.state_dict().copy()
    best_val_loss = float('inf')
    patience      = 0

    # training
    if train_loader:
        for epoch in range(1, args.num_epochs + 1):
            model.train()
            running = 0.0
            for feats, normal_lab, func_lab in train_loader:
                feats = feats.to(device)
                labels = (normal_lab if args.output_dim==normal_dim else func_lab).to(device).float()
                optimizer.zero_grad()
                outputs, _, _ = model(feats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running += loss.item() * feats.size(0)

            train_loss = running / len(train_loader.dataset)
            v_loss, v_f1, v_auc, v_acc, v_kl = evaluate_model(
                model, val_loader, device, criterion,
                normal_dim, func_dim, args.output_dim
            )

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state    = model.state_dict().copy()
                # checkpoint
                torch.save(best_state, args.checkpoint_path)
                patience      = 0
            else:
                patience += 1

            if patience >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break
    else:
        print("zero-shot training pass")

    # test evaluation
    model.load_state_dict(best_state)
    t_loss, t_f1, t_auc, t_acc, t_kl = evaluate_model(
        model, test_loader, device, criterion,
        normal_dim, func_dim, args.output_dim
    )
    print(f"[Test] F1:{t_f1:.4f} | AUC:{t_auc:.4f} | KL:{t_kl:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero","few","full"], default="full", help="training mode")
    parser.add_argument("--shot", type=int, default=4, help="few-shot sample k")
    parser.add_argument("--feature_type", type=str, default="image", help="map or image")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--label_dir", type=str, default="./seoul_image/label")
    parser.add_argument("--train_csv", type=str, default="./train.csv")
    parser.add_argument("--valid_csv", type=str, default="./valid.csv")
    parser.add_argument("--test_csv", type=str, default="./test.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dim", type=int, default=6, help="6→function, 22→institution")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="seed=1, 17, 42")
    parser.add_argument("--checkpoint_path", type=str, default="best_model"
    ".pth")
    args = parser.parse_args()
    main(args)
