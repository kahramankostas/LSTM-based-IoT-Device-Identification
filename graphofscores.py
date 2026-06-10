import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 2-final_summary_stats.csv -> 20-final_summary_stats.csv
all_data = []
sns.set_style("whitegrid")
for n in range(2, 21):
    file_name = f"./evaluation_results/{n}-final_summary_stats.csv"

    if os.path.exists(file_name):
        df = pd.read_csv(file_name, index_col=0)

        # Mean değerlerini al
        means = df["Mean"].to_dict()
        means["n"] = n

        all_data.append(means)

# Tek dataframe haline getir
results_df = pd.DataFrame(all_data)

# n değerine göre sırala
results_df = results_df.sort_values("n")

print(results_df)










# ---------------------------------------------------
# Grafik 1: Accuracy metrikleri
# ---------------------------------------------------

plt.figure(figsize=(12, 6))

metrics = ["Acc",  "Prec", "Rec", "F1", "kap"]

for metric in metrics:
    plt.plot(results_df["n"], results_df[metric], marker='o', label=metric)

plt.xlabel("n")
plt.ylabel("Score")
plt.title("Performance Metrics vs n")
plt.legend()
plt.grid(True)
plt.savefig("performance_metrics.pdf", bbox_inches='tight', format="pdf")#, dpi=400)
plt.show()



import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(6, 3))

metrics = ["Acc", "Prec", "Rec", "F1", "kap"]

for metric in metrics:
    plt.plot(results_df["n"], results_df[metric], marker='o', label=metric)

plt.xlabel("sequence length")
plt.ylabel("Score")
plt.title("Performance Metrics vs sequence length")

# X eksenini ardışık tam sayılar yap
plt.xticks(np.arange(
    int(results_df["n"].min()),
    int(results_df["n"].max()) + 1,
    1
))

plt.legend()
plt.grid(True)

plt.savefig("performance_metrics2.pdf", bbox_inches='tight', format="pdf")
plt.show()











# ---------------------------------------------------
# Grafik 2: Çalışma Süresi
# ---------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(results_df["n"], results_df["time"], marker='o')

plt.xlabel("sequence length")
plt.ylabel("Time")
plt.title("Execution Time vs sequence length")
plt.grid(True)
plt.savefig("execution_time.pdf", bbox_inches='tight', format="pdf")#, dpi=400)
plt.show()

# ---------------------------------------------------
# Grafik 3: Heatmap benzeri görünüm
# ---------------------------------------------------

plt.figure(figsize=(12, 6))

heatmap_data = results_df.set_index("n")[metrics]

plt.imshow(heatmap_data.T, aspect='auto')

plt.yticks(range(len(metrics)), metrics)
plt.xticks(range(len(results_df["n"])), results_df["n"])

plt.colorbar(label="Score")
plt.title("Metric Distribution Heatmap")

plt.savefig("metric_distribution.pdf", bbox_inches='tight', format="pdf")#, dpi=400)
plt.show()


import matplotlib.pyplot as plt
import pandas as pd
# ... (results_df ve metrics'in zaten tanımlı olduğunu varsayıyoruz)

plt.figure(figsize=(12, 6))

# 1. Veriyi hazırlama
heatmap_data = results_df.set_index("n")[metrics]

# 2. Normalizasyon Adımı (Her bir 'n' sütunundaki değerleri 0-1 arasına ölçekleme)
# axis=0, satırlar (yani her bir 'n' değeri) boyunca min/max hesaplanmasını sağlar.
# Bu, her bir 'n' için skorların göreceli dağılımını gösterir.
min_vals = heatmap_data.min(axis=0)
max_vals = heatmap_data.max(axis=0)

# Sadece max_vals > min_vals olan sütunlar için bölme işlemi yapılır (Sıfıra bölmeyi önlemek için)
range_vals = max_vals - min_vals
# Sıfır bölmesini engellemek için, range_vals'taki sıfır değerlerini maskeleyebiliriz.
# Ancak basitlik adına, tüm değerleri normalize edelim:
normalized_data = (heatmap_data - min_vals) / range_vals

# 3. Heatmap'i çizme
plt.imshow(normalized_data.T, aspect='auto') # .T (Transpose) kullanmaya devam ediyoruz

# Etiketler ve Başlıklar
plt.yticks(range(len(metrics)), metrics)
plt.xticks(range(len(results_df["n"])), results_df["n"])

plt.colorbar(label="Normalized Score") # Renk çubuğunu güncelledik
plt.title("Metric Distribution Heatmap (Normalized)")

plt.savefig("metric_distribution_normalized.pdf", bbox_inches='tight', format="pdf")
plt.show()
