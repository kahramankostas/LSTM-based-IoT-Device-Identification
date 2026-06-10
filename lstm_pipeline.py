import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

import optuna
from optuna.pruners import MedianPruner

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. ÖZELLİK TANIMLARI & YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────

FEATURE_COLS = [
    'ARP', 'LLC', 'EAPOL', 'IP', 'ICMP', 'ICMP6', 'TCP', 'UDP',
    'TCP_w_size', 'HTTP', 'HTTPS', 'DHCP', 'BOOTP', 'SSDP', 'DNS',
    'MDNS', 'NTP', 'IP_padding', 'IP_add_count', 'IP_ralert',
    'Portcl_src', 'Portcl_dst', 'Pck_size', 'Pck_rawdata', 'Entropy'
]
LABEL_COL = 'Label'

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len - 1])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)

def prepare_dataset(file_path, seq_len, scaler=None, label_encoder=None, fit=False):
    df = pd.read_csv(file_path)
    for col in FEATURE_COLS:
        if col not in df.columns: df[col] = 0
            
    X_raw = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df[LABEL_COL].values

    if fit:
        label_encoder = LabelEncoder()
        y_enc = label_encoder.fit_transform(y_raw)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
    else:
        y_enc = label_encoder.transform(y_raw)
        X_scaled = scaler.transform(X_raw)

    X_seq, y_seq = create_sequences(X_scaled, y_enc, seq_len)
    return X_seq, y_seq, label_encoder, scaler

# ─────────────────────────────────────────────
# 2. MODEL MİMARİSİ
# ─────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_classes, dropout=0.3, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        direction = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size * direction)
        self.fc = nn.Linear(hidden_size * direction, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)

# ─────────────────────────────────────────────
# 3. GÖRSELLEŞTİRME VE RAPORLAMA
# ─────────────────────────────────────────────

def plot_learning_curve(history, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Eğrisi"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["val_f1"], color="green", label="Val F1")
    axes[1].set_title("Validation F1 Eğrisi"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curve.pdf"))
    plt.close()

def plot_cm(y_true, y_pred, class_names, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    temp=pd.DataFrame(cm)
    temp.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))#, index=False)

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    temp=pd.DataFrame(cm_norm)
    temp.to_csv(os.path.join(out_dir, "confusion_matrix_normalized.csv"))#, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(28, 10))
    for ax, data, fmt, title in zip(axes, [cm, cm_norm], ["d", ".2f"], ["Confusion Matrix", "Normalized CM"]):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.pdf"))
    plt.close()

# ─────────────────────────────────────────────
# 4. EĞİTİM & ANA AKIŞ
# ─────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def run_eval(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), f1, np.array(all_preds), np.array(all_labels)

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    model_file = os.path.join(args.out_dir, "lstm_model.pt")
    
    # Veri Hazırlığı
    X_train, y_train, le, scaler = prepare_dataset(args.train_csv, args.seq_len, fit=True)
    X_val, y_val, _, _ = prepare_dataset(args.val_csv, args.seq_len, scaler, le)
    X_test, y_test, _, _ = prepare_dataset(args.test_csv, args.seq_len, scaler, le)
    
    n_classes, input_size = len(le.classes_), X_train.shape[2]
    loader_tr = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32, shuffle=True)
    loader_va = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=32)
    loader_te = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=32)

    # Hiperparametre Kontrolü
    best_params = None
    if os.path.exists(model_file):
        print(f"✓ Mevcut model bulundu: {model_file}. HP Optimizasyonu atlanıyor.")
        checkpoint = torch.load(model_file, map_location="cpu")
        best_params = checkpoint.get("best_params")

    if best_params is None:
        print("── Optuna HP Arama ──")
        def objective(trial):
            hp = {
                "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "bidirectional": trial.suggest_categorical("bidirectional", [True, False])
            }
            m = LSTMClassifier(input_size, hp["hidden_size"], hp["num_layers"], n_classes, bidirectional=hp["bidirectional"]).to(device)
            opt = torch.optim.Adam(m.parameters(), lr=hp["lr"])
            crit = nn.CrossEntropyLoss()
            
            best_f = 0
            for e in range(args.opt_epochs):
                m.train()
                for xb, yb in loader_tr:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
                _, f1, _, _ = run_eval(m, loader_va, crit, device)
                best_f = max(best_f, f1)
            return best_f

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.trials)
        best_params = study.best_params

    # Final Eğitim
    print("\n── Final Eğitim Başlıyor ──")
    model = LSTMClassifier(input_size, best_params["hidden_size"], best_params["num_layers"], n_classes, bidirectional=best_params["bidirectional"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params.get("lr", 1e-3))
    
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1, best_state = 0, None
    
    for epoch in range(args.epochs):
        model.train()
        t_loss = 0
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            l = criterion(model(xb), yb)
            l.backward(); optimizer.step()
            t_loss += l.item()
        
        v_loss, v_f1, _, _ = run_eval(model, loader_va, criterion, device)
        history["train_loss"].append(t_loss/len(loader_tr)); history["val_loss"].append(v_loss); history["val_f1"].append(v_f1)
        
        if v_f1 > best_f1:
            best_f1 = v_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 10 == 0: print(f"Epoch {epoch+1} | Val F1: {v_f1:.4f}")

    # Raporlama & Kayıt
    plot_learning_curve(history, args.out_dir)
    model.load_state_dict(best_state)
    _, _, t_preds, t_labels = run_eval(model, loader_te, criterion, device)
    plot_cm(t_labels, t_preds, le.classes_, args.out_dir)
    
    with open(os.path.join(args.out_dir, "report.txt"), "w") as f:
        f.write(classification_report(t_labels, t_preds, target_names=le.classes_))
    
    torch.save({"model_state": best_state, "best_params": best_params, "label_encoder": le, "scaler": scaler}, model_file)
    print(f"✅ Tamamlandı. Çıktılar '{args.out_dir}' klasöründe.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--out_dir", default="./output")
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--opt_epochs", type=int, default=15)
    parser.add_argument("--trials", type=int, default=20)
    main(parser.parse_args())