import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, cohen_kappa_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. ORİJİNAL FONKSİYONLAR & MİMARİ
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
# 2. PDF GÖRSELLEŞTİRME FONKSİYONU
# ─────────────────────────────────────────────

def save_confusion_matrices_to_pdf(avg_cm, avg_cm_norm, class_names, output_path):
    """
    Ham ve Normalize edilmiş Confusion Matrix'leri yan yana çizip PDF olarak kaydeder.
    """
    # Sınıf sayısına göre dinamik figür boyutu ayarlama
    fig_size_width = max(32, len(class_names) * 1.5)
    fig_size_height = max(14, len(class_names) * 0.7)
    
    fig, axes = plt.subplots(1, 2, figsize=(fig_size_width, fig_size_height))
    
    # 1. Ham Confusion Matrix (Adetler)
    sns.heatmap(
        avg_cm, annot=True, fmt=".1f", cmap="Blues", 
        xticklabels=class_names, yticklabels=class_names, ax=axes[0],
        cbar=True, square=True
    )
    axes[0].set_title("Average Confusion Matrix (Counts)", fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel("Predicted Label", fontsize=12)
    axes[0].set_ylabel("True Label", fontsize=12)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

    # 2. Normalize Confusion Matrix (Yüzdeler)
    sns.heatmap(
        avg_cm_norm, annot=True, fmt=".2f", cmap="Oranges", 
        xticklabels=class_names, yticklabels=class_names, ax=axes[1],
        cbar=True, square=True
    )
    axes[1].set_title("Average Confusion Matrix (Normalized)", fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel("Predicted Label", fontsize=12)
    axes[1].set_ylabel("True Label", fontsize=12)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

    plt.tight_layout()
    
    # PDF olarak kaydetme
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────
# 3. GELİŞTİRİLMİŞ DEĞERLENDİRME DÖNGÜSÜ
# ─────────────────────────────────────────────

def run_statistical_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Cihaz: {device} üzerinde değerlendirme başlıyor...")
    
    # Model ve Checkpoint Yükleme
    if not os.path.exists(args.model_path):
        print(f"❌ Hata: Model dosyası bulunamadı: {args.model_path}")
        return

    checkpoint = torch.load(args.model_path, map_location=device)
    le = checkpoint['label_encoder']
    scaler = checkpoint['scaler']
    best_params = checkpoint['best_params']
    
    class_names = [str(c) for c in le.classes_]
    n_classes = len(class_names)

    # Test Verisini Hazırlama
    X_test, y_test, _, _ = prepare_dataset(
        args.test_csv, args.seq_len, scaler=scaler, label_encoder=le, fit=False
    )
    
    loader_te = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), 
        batch_size=args.batch_size, 
        shuffle=False 
    )

    # Modeli Kurma
    input_size = X_test.shape[2]
    
    model = LSTMClassifier(
        input_size, 
        best_params["hidden_size"], 
        best_params["num_layers"], 
        n_classes, 
        bidirectional=best_params["bidirectional"]
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    results = []
    
    # Kümülatif matris takibi
    total_cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    total_cm_norm = np.zeros((n_classes, n_classes), dtype=np.float64)
    
    class_metrics_history = {c: {"precision": [], "recall": [], "f1-score": []} for c in class_names}
    support_dict = {}

    for i in range(args.iterations):
        start_time = time.time()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for xb, yb in loader_te:
                xb = xb.to(device)
                logits = model(xb)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(yb.numpy())
        
        end_time = time.time()
        iter_time = end_time - start_time
        
        # Genel Metrikler
        metrics = {
            "Iteration": i + 1,
            "Acc": accuracy_score(all_labels, all_preds),
            "b_Acc": balanced_accuracy_score(all_labels, all_preds),
            "Prec": precision_score(all_labels, all_preds, average='macro', zero_division=0),
            "Rec": recall_score(all_labels, all_preds, average='macro', zero_division=0),
            "F1": f1_score(all_labels, all_preds, average='macro', zero_division=0),
            "kap": cohen_kappa_score(all_labels, all_preds),
            "time": iter_time
        }
        results.append(metrics)
        
        # Confusion Matrix (Bu iterasyon için)
        cm = confusion_matrix(all_labels, all_preds, labels=range(n_classes))
        total_cm += cm
        
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
        total_cm_norm += cm_norm
        
        # Sınıf Bazlı Detaylı Rapor
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
        for c in class_names:
            class_metrics_history[c]["precision"].append(report[c]["precision"])
            class_metrics_history[c]["recall"].append(report[c]["recall"])
            class_metrics_history[c]["f1-score"].append(report[c]["f1-score"])
            support_dict[c] = report[c]["support"]

        print(f"[{i+1}/{args.iterations}] F1: {metrics['F1']:.4f} | Süre: {metrics['time']:.2f}s")

    # ─────────────────────────────────────────────
    # SONUÇLARI ORTALAMA & KAYDETME
    # ─────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Genel İstatistikler
    df_results = pd.DataFrame(results)
    stats = df_results.drop(columns=['Iteration']).agg(['mean', 'std']).T
    stats.columns = ['Mean', 'Std']
    df_results.to_csv(os.path.join(args.out_dir, f"{args.seq_len}-detailed_iterations.csv"), index=False)
    stats.to_csv(os.path.join(args.out_dir, f"{args.seq_len}-final_summary_stats.csv"))
    
    # 2. Ortalama Confusion Matrix Hesaplama
    avg_cm = total_cm / args.iterations
    avg_cm_norm = total_cm_norm / args.iterations
    
    df_cm = pd.DataFrame(avg_cm, index=class_names, columns=class_names)
    df_cm_norm = pd.DataFrame(avg_cm_norm, index=class_names, columns=class_names)
    
    df_cm.to_csv(os.path.join(args.out_dir, f"{args.seq_len}-avg_confusion_matrix.csv"))
    df_cm_norm.to_csv(os.path.join(args.out_dir, f"{args.seq_len}-avg_confusion_matrix_normalized.csv"))
    
    # PDF Görselleştirme Fonksiyonunu Çağırma
    pdf_output_path = os.path.join(args.out_dir, f"{args.seq_len}-confusion_matrices.pdf")
    save_confusion_matrices_to_pdf(avg_cm, avg_cm_norm, class_names, pdf_output_path)
    
    # 3. Sınıf Bazlı Başarı Raporu
    class_report_rows = []
    for c in class_names:
        class_report_rows.append({
            "Class": c,
            "Precision_Mean": np.mean(class_metrics_history[c]["precision"]),
            "Precision_Std": np.std(class_metrics_history[c]["precision"]),
            "Recall_Mean": np.mean(class_metrics_history[c]["recall"]),
            "Recall_Std": np.std(class_metrics_history[c]["recall"]),
            "F1-Score_Mean": np.mean(class_metrics_history[c]["f1-score"]),
            "F1-Score_Std": np.std(class_metrics_history[c]["f1-score"]),
            "Support": support_dict[c]
        })
    df_class_report = pd.DataFrame(class_report_rows).set_index("Class")
    df_class_report.to_csv(os.path.join(args.out_dir, f"{args.seq_len}-class_wise_performance.csv"))
    
    # Konsol Çıktıları
    print("\n" + "="*50)
    print("📊 GENEL ÖZET İSTATİSTİKLER (MEAN & STD)")
    print(stats)
    print("="*50)
    
    print("\n🎯 SINIF BAZLI PERFORMANS RAPORU (ORTALAMA)")
    print(df_class_report[['Precision_Mean', 'Recall_Mean', 'F1-Score_Mean', 'Support']])
    print("="*50)
    
    print(f"✅ Tüm sonuçlar '{args.out_dir}' klasörüne kaydedildi:")
    print(f"  - {args.seq_len}-avg_confusion_matrix.csv")
    print(f"  - {args.seq_len}-avg_confusion_matrix_normalized.csv")
    print(f"  - {args.seq_len}-class_wise_performance.csv")
    print(f"  - 📄 {args.seq_len}-confusion_matrices.pdf ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="lstm_model.pt dosyasının yolu")
    parser.add_argument("--test_csv", required=True, help="Test verisi (CSV)")
    parser.add_argument("--out_dir", default="./evaluation_results")
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=10, help="Kaç kez test edilsin?")
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    run_statistical_evaluation(args)