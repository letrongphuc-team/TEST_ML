import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
import mysql.connector
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from connector.connector import Connector

# In được tiếng Việt ra console Windows (tránh UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Đặt SHOW_PLOTS=True nếu muốn hiện cửa sổ biểu đồ; mặc định chỉ lưu ảnh vào results/
SHOW_PLOTS = False
if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

RESULT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Higher-contrast colors that stay soft while keeping clusters easy to distinguish.
CLUSTER_PALETTE = ["#5B84B1", "#E39A52", "#C96D73", "#5EAAA0", "#79A65A",
                   "#C6A63A", "#8E72AA", "#CF829C", "#A77D5D", "#7D8791"]
HIST_COLOR = "#6F91B8"
ELBOW_COLOR = "#426B99"
SILHOUETTE_COLOR = "#D7833F"
OPTIMAL_LINE_COLOR = "#B94F5A"
HEATMAP_CMAP = sns.color_palette(["#78A6C8", "#F8F6F0", "#CC7777"], as_cmap=True)
PROFILE_CMAP = sns.color_palette(["#68A99D", "#FFF8E8", "#C8758B"], as_cmap=True)


def savePlot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_FOLDER, filename), dpi=120)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

# ---------------------------------------------------------------- 1. Lấy dữ liệu
def connect(server="localhost", port=3306, database="sakila",
            username="root", password="@MySQL@123456"):
    try:
        return mysql.connector.connect(host=server, port=port, database=database,
                                       user=username, password=password,
                                       use_pure=True)
    except Exception as e:
        print(str(e))
        return None


def queryDataset(conn, sql):
    cursor = conn.cursor()
    cursor.execute(sql)
    df = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)
    cursor.close()
    return df


SQL_FEATURES = """
SELECT c.customer_id,
       CONCAT(c.first_name, ' ', c.last_name)                        AS customer_name,
       c.store_id                                                    AS home_store,
       c.active,
       COUNT(r.rental_id)                                            AS total_rentals,
       COUNT(DISTINCT f.film_id)                                     AS distinct_films,
       COUNT(DISTINCT i.inventory_id)                                AS distinct_inventory,
       COUNT(r.rental_id) /
           GREATEST(TIMESTAMPDIFF(DAY, MIN(r.rental_date), MAX(r.rental_date)) / 30, 1)
                                                                     AS rentals_per_month,
       AVG(f.rental_rate)                                            AS avg_rental_rate,
       AVG(f.replacement_cost)                                       AS avg_replacement_cost,
       AVG(f.length)                                                 AS avg_film_length,
       AVG(DATEDIFF(r.return_date, r.rental_date))                   AS avg_days_kept,
       AVG(CASE WHEN r.return_date IS NULL THEN NULL
                WHEN DATEDIFF(r.return_date, r.rental_date) > f.rental_duration THEN 1
                ELSE 0 END)                                          AS late_ratio,
       AVG(i.store_id <> c.store_id)                                 AS other_store_ratio
FROM customer c
JOIN rental    r ON r.customer_id  = c.customer_id
JOIN inventory i ON i.inventory_id = r.inventory_id
JOIN film      f ON f.film_id      = i.film_id
GROUP BY c.customer_id, customer_name, c.store_id, c.active
ORDER BY c.customer_id
"""

connector = Connector(database="sakila")
try:
    connector.connect()
    df = connector.queryDataset(SQL_FEATURES)
finally:
    if connector.conn is not None and connector.conn.is_connected():
        connector.conn.close()

if df is None:
    raise RuntimeError("Không lấy được dữ liệu từ Sakila; hãy kiểm tra lỗi truy vấn phía trên.")
if df.empty:
    raise RuntimeError("Truy vấn Sakila thành công nhưng không có dữ liệu khách hàng.")

featureColumns = ['total_rentals', 'distinct_films', 'distinct_inventory',
                  'rentals_per_month', 'avg_rental_rate', 'avg_replacement_cost',
                  'avg_film_length', 'avg_days_kept', 'late_ratio',
                  'other_store_ratio']
df[featureColumns] = df[featureColumns].astype(float)
print(df.head())
print(df[featureColumns].describe().T)


# ---------------------------------------------------------------- 2. Khám phá dữ liệu
def showHistogram(df, columns):
    plt.figure(figsize=(14, 10))
    for n, col in enumerate(columns, start=1):
        plt.subplot(4, 3, n)
        sns.histplot(df[col], bins=30, color=HIST_COLOR, edgecolor="white", alpha=0.95)
        plt.title(f"Histogram of {col}", fontweight="bold")
    savePlot("01_histograms.png")


def showCorrelation(df, columns):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[columns].corr(), annot=True, fmt=".2f", cmap=HEATMAP_CMAP,
                center=0, annot_kws={"weight": "bold", "size": 9})
    plt.title("Correlation of features", fontweight="bold")
    savePlot("02_correlation.png")


showHistogram(df, featureColumns)
showCorrelation(df, featureColumns)

# distinct_inventory gần như trùng total_rentals (corr ~ 1) -> bỏ để tránh đếm 2 lần
columnsForCluster = [c for c in featureColumns if c != 'distinct_inventory']

# Chuẩn hóa vì các thuộc tính khác đơn vị (lượt, $, phút, tỷ lệ)
scaler = StandardScaler()
X = scaler.fit_transform(df[columnsForCluster].values)


# ---------------------------------------------------------------- 3. Chọn số cụm k
def elbowAndSilhouette(X, ks=range(2, 11), threshold=0.05):
    ks = list(ks)
    inertia, silhouettes = [], []
    for k in ks:
        model = KMeans(n_clusters=k, init='k-means++', max_iter=500,
                       n_init=10, random_state=42)
        labels = model.fit_predict(X)
        inertia.append(model.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    # Điểm khuỷu: k nhỏ nhất mà khi tăng lên k+1 cụm, inertia chỉ giảm thêm
    # ít hơn threshold (5%) so với inertia của k nhỏ nhất -> thêm cụm không còn đáng kể
    drops = -np.diff(inertia) / inertia[0]
    belowThreshold = np.where(drops < threshold)[0]
    optimalK = ks[belowThreshold[0]] if len(belowThreshold) > 0 else ks[-1]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(ks, inertia, 'o-', color=ELBOW_COLOR, linewidth=2.8,
             markersize=8, label='Inertia (Elbow)')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia', color=ELBOW_COLOR)
    ax1.tick_params(axis='y', labelcolor=ELBOW_COLOR)
    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, 's--', color=SILHOUETTE_COLOR, linewidth=2.8,
             markersize=8, label='Silhouette')
    best_silhouette_index = int(np.argmax(silhouettes))
    ax2.scatter(ks[best_silhouette_index], silhouettes[best_silhouette_index],
                s=150, color=SILHOUETTE_COLOR, edgecolor="white", linewidth=1.5,
                zorder=6, label=f"Best silhouette: k={ks[best_silhouette_index]}")
    ax2.set_ylabel('Silhouette score', color=SILHOUETTE_COLOR)
    ax2.tick_params(axis='y', labelcolor=SILHOUETTE_COLOR)
    ax1.axvline(x=optimalK, color=OPTIMAL_LINE_COLOR, linestyle='--',
                linewidth=3, label=f'Optimal k = {optimalK}')
    plt.title(f'Elbow & Silhouette (chọn k = {optimalK})', fontweight="bold")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    savePlot("03_elbow_silhouette.png")

    for k, ine, s in zip(ks, inertia, silhouettes):
        print(f"k={k:<2}  inertia={ine:10.2f}  silhouette={s:.4f}")
    return optimalK


optimalK = elbowAndSilhouette(X)
print("Số cụm được chọn:", optimalK)

# ---------------------------------------------------------------- 4. KMeans
model = KMeans(n_clusters=optimalK, init='k-means++', max_iter=500,
               n_init=10, random_state=42)
df['cluster'] = model.fit_predict(X)

# ---------------------------------------------------------------- 5. Mô tả cụm
profile = df.groupby('cluster')[featureColumns].mean()
profile.insert(0, 'num_customers', df['cluster'].value_counts().sort_index())
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 30)
print("\nĐặc trưng trung bình mỗi cụm:")
print(profile.round(2))


LABELS = {  # thuộc tính: (nhãn khi cao, nhãn khi thấp)
    'total_rentals': ("thuê nhiều", "thuê ít"),
    'rentals_per_month': ("thuê thường xuyên", "thuê thưa"),
    'avg_rental_rate': ("thích phim giá thuê cao", "thích phim giá thuê rẻ"),
    'avg_replacement_cost': ("chọn phim giá trị cao", "chọn phim giá trị thấp"),
    'avg_film_length': ("thích phim dài", "thích phim ngắn"),
    'avg_days_kept': ("giữ đĩa lâu", "trả đĩa nhanh"),
    'late_ratio': ("hay trả trễ", "trả đúng hạn"),
    'other_store_ratio': ("hay thuê ở cửa hàng khác", "trung thành cửa hàng nhà"),
}


def nameCluster(z, minZ=0.4):
    """Đặt tên cụm theo các thuộc tính lệch nhiều nhất (|z| >= minZ) so với trung bình"""
    tags = [LABELS[c][0] if z[c] > 0 else LABELS[c][1]
            for c in z.abs().sort_values(ascending=False).index
            if c in LABELS and abs(z[c]) >= minZ]
    return ", ".join(tags[:4]) if tags else "khách hàng trung bình"


zProfile = (profile[featureColumns] - df[featureColumns].mean()) / df[featureColumns].std()
profile['description'] = zProfile.apply(nameCluster, axis=1)
print()
for cid, desc in profile['description'].items():
    print(f"Cluster {cid} ({profile.loc[cid, 'num_customers']} KH): {desc}")


# ---------------------------------------------------------------- 6. Trực quan hóa
def visualizePCA(X, labels, k):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    centers2 = pca.transform(model.cluster_centers_)
    plt.figure(figsize=(10, 8))
    palette = CLUSTER_PALETTE[:k]
    for i in range(k):
        plt.scatter(X2[labels == i, 0], X2[labels == i, 1], s=45,
                    color=palette[i], alpha=0.9, edgecolors="white", linewidths=0.6,
                    label=f'Cluster {i}')
    plt.scatter(centers2[:, 0], centers2[:, 1], s=380, c='#343B42', marker='X',
                edgecolors='white', linewidths=1.8, label='Centroids', zorder=5)
    ev = pca.explained_variance_ratio_
    plt.xlabel(f'PC1 ({ev[0]:.0%} variance)')
    plt.ylabel(f'PC2 ({ev[1]:.0%} variance)')
    plt.title('Clusters of Customers (PCA 2D) - Film & Inventory interest')
    plt.legend()
    savePlot("04_clusters_pca.png")


def visualizeScatter(df, x, y, k):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x=x, y=y, hue='cluster',
                    palette=CLUSTER_PALETTE[:k], s=75, edgecolor="white", linewidth=0.5)
    plt.title(f'Clusters of Customers - {x} X {y}', fontweight="bold")
    savePlot(f"05_scatter_{x}_{y}.png")


def visualizeProfileHeatmap(df, columns):
    # z-score của trung bình mỗi cụm -> thấy cụm nào cao/thấp ở thuộc tính nào
    z = df.groupby('cluster')[columns].mean()
    z = (z - df[columns].mean()) / df[columns].std()
    plt.figure(figsize=(12, 4 + 0.4 * len(z)))
    sns.heatmap(z, annot=True, fmt=".2f", cmap=PROFILE_CMAP, center=0,
                annot_kws={"weight": "bold"})
    plt.title('Cluster profile (z-score so với trung bình toàn bộ)', fontweight="bold")
    savePlot("06_cluster_profile.png")


visualizePCA(X, df['cluster'].values, optimalK)
visualizeScatter(df, 'total_rentals', 'avg_rental_rate', optimalK)
visualizeScatter(df, 'avg_film_length', 'late_ratio', optimalK)
visualizeProfileHeatmap(df, columnsForCluster)


# ---------------------------------------------------------------- 7. Xuất kết quả
def export_df2excel(df, profile, filename):
    filepath = os.path.join(RESULT_FOLDER, filename)
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        profile.round(3).to_excel(writer, sheet_name="Cluster_Profile")
        for cluster_id in sorted(df["cluster"].unique()):
            df[df["cluster"] == cluster_id].to_excel(
                writer, sheet_name=f"Cluster_{cluster_id}", index=False)
    print(f"\nĐã xuất {df['cluster'].nunique()} cluster ra file: {filepath}")


export_df2excel(df, profile, "sakila_customer_clusters.xlsx")
