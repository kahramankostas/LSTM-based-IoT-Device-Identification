import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def csv_to_heatmap(file_path):
    # 1. CSV dosyasını oku
    # index_col=0: İlk sütunu satır etiketleri (indeks) olarak ayarlar.
    # header=0: İlk satırı sütun etiketleri (başlık) olarak ayarlar (zaten varsayılandır).
    df = pd.read_csv(file_path, index_col=0, header=0)

    # 2. Grafik boyutunu ve çözünürlüğünü ayarla
    plt.figure(figsize=(15, 12), dpi=100)

    # 3. Isı haritasını çizdir
    # annot=True: Hücrelerin içine gerçek sayısal değerleri yazar.
    # cmap='YlGnBu': Renk paleti (Sarı-Yeşil-Mavi tonları). İsteğe göre 'coolwarm' veya 'magma' yapılabilir.
    # fmt='.2f': Sayıların virgülden sonra 2 basamaklı gösterilmesini sağlar (tam sayıysa 'd' yapabilirsin).
    sns.heatmap(df, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)

    # 4. Grafik başlığını ve estetiğini düzenle
    #plt.title("CSV Verisi Isı Haritası (Heatmap)", fontsize=16, pad=15)
    plt.xlabel("true label", fontsize=12)
    plt.ylabel("predicted label", fontsize=12)

    # Eksen etiketlerinin düzgün görünmesi için sıkıştır ve göster
    plt.tight_layout()
    plt.savefig("cm.pdf", bbox_inches='tight', format="pdf")#, dpi=400)
    plt.show()


# Kodu çalıştırmak için dosya yolunu yazman yeterli:
csv_to_heatmap('veri.csv')