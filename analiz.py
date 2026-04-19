import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Türkiye enflasyon verisi (2019-2024)
veri = {
    "yil": [2019, 2020, 2021, 2022, 2023, 2024],
    "enflasyon": [11.8, 14.6, 19.6, 72.3, 64.8, 58.5],
    "issizlik": [13.7, 13.2, 12.0, 10.4, 9.4, 8.8],
    "dolar_kuru": [5.67, 7.44, 8.85, 16.55, 23.08, 32.50]
}

df = pd.DataFrame(veri)

print("=== Türkiye Ekonomi Verileri ===")
print(df)
print()
print(f"En yüksek enflasyon: {df['enflasyon'].max()}% ({df.loc[df['enflasyon'].idxmax(), 'yil']})")
print(f"En düşük işsizlik: {df['issizlik'].min()}% ({df.loc[df['issizlik'].idxmin(), 'yil']})")

# Grafik ayarları
sns.set_theme(style="darkgrid")
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle("Türkiye Ekonomi Analizi (2019-2024)", fontsize=16, fontweight="bold")

# Enflasyon grafiği
axes[0].plot(df["yil"], df["enflasyon"], color="red", marker="o", linewidth=2)
axes[0].fill_between(df["yil"], df["enflasyon"], alpha=0.3, color="red")
axes[0].set_title("Enflasyon (%)")
axes[0].set_ylabel("Yüzde")

# İşsizlik grafiği
axes[1].plot(df["yil"], df["issizlik"], color="blue", marker="o", linewidth=2)
axes[1].fill_between(df["yil"], df["issizlik"], alpha=0.3, color="blue")
axes[1].set_title("İşsizlik (%)")
axes[1].set_ylabel("Yüzde")

# Dolar kuru grafiği
axes[2].plot(df["yil"], df["dolar_kuru"], color="green", marker="o", linewidth=2)
axes[2].fill_between(df["yil"], df["dolar_kuru"], alpha=0.3, color="green")
axes[2].set_title("Dolar Kuru (TL)")
axes[2].set_ylabel("TL")

plt.tight_layout()
plt.savefig("ekonomi_grafik.png", dpi=150)
plt.show()
print("Grafik kaydedildi!")

# Korelasyon analizi
fig2, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["yil"], df["enflasyon"], color="red", marker="o", linewidth=2, label="Enflasyon (%)")
ax.plot(df["yil"], df["issizlik"], color="blue", marker="o", linewidth=2, label="İşsizlik (%)")

ax2 = ax.twinx()
ax2.plot(df["yil"], df["dolar_kuru"], color="green", marker="s", linewidth=2, linestyle="--", label="Dolar (TL)")
ax2.set_ylabel("Dolar Kuru (TL)", color="green")

ax.set_title("Enflasyon, İşsizlik ve Dolar Kuru İlişkisi")
ax.set_xlabel("Yıl")
ax.set_ylabel("Yüzde (%)")
ax.legend(loc="upper left")
ax2.legend(loc="upper center")

plt.tight_layout()
plt.savefig("korelasyon_grafik.png", dpi=150)
plt.show()
print("Korelasyon grafiği kaydedildi!")



from sklearn.linear_model import LinearRegression
import numpy as np

# Gelecek tahmini
X = np.array(df["yil"]).reshape(-1, 1)
y_enflasyon = np.array(df["enflasyon"])
y_dolar = np.array(df["dolar_kuru"])

# Model eğit
model_enflasyon = LinearRegression()
model_enflasyon.fit(X, y_enflasyon)

model_dolar = LinearRegression()
model_dolar.fit(X, y_dolar)

# 2025-2026 tahmini
gelecek = np.array([[2025], [2026]])
tahmin_enflasyon = model_enflasyon.predict(gelecek)
tahmin_dolar = model_dolar.predict(gelecek)

print("=== Gelecek Tahmini ===")
print(f"2025 Enflasyon Tahmini: %{tahmin_enflasyon[0]:.1f}")
print(f"2026 Enflasyon Tahmini: %{tahmin_enflasyon[1]:.1f}")
print(f"2025 Dolar Tahmini: {tahmin_dolar[0]:.2f} TL")
print(f"2026 Dolar Tahmini: {tahmin_dolar[1]:.2f} TL")

# Tahmin grafiği
fig3, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["yil"], df["enflasyon"], color="red", marker="o", linewidth=2, label="Gerçek Enflasyon")
ax.plot([2024, 2025, 2026],
        [df["enflasyon"].iloc[-1], tahmin_enflasyon[0], tahmin_enflasyon[1]],
        color="red", marker="o", linewidth=2, linestyle="--", label="Tahmin")

ax2 = ax.twinx()
ax2.plot(df["yil"], df["dolar_kuru"], color="green", marker="s", linewidth=2, label="Gerçek Dolar")
ax2.plot([2024, 2025, 2026],
         [df["dolar_kuru"].iloc[-1], tahmin_dolar[0], tahmin_dolar[1]],
         color="green", marker="s", linewidth=2, linestyle="--", label="Tahmin")
ax2.set_ylabel("Dolar Kuru (TL)", color="green")

ax.set_title("Türkiye Ekonomi Tahmini (2025-2026)")
ax.set_xlabel("Yıl")
ax.set_ylabel("Enflasyon (%)")
ax.legend(loc="upper left")
ax2.legend(loc="upper center")

plt.tight_layout()
plt.savefig("tahmin_grafik.png", dpi=150)
plt.show()
print("Tahmin grafiği kaydedildi!")