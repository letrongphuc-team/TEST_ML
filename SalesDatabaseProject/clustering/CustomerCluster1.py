import os
import sys
import html
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from SalesDatabaseProject.dal.customer_dao import CustomerDao
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
from openpyxl.chart import PieChart, Reference
from openpyxl.chart.label import DataLabelList
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs


def showHistogram(df, columns):
    if not columns:
        raise ValueError("At least one column is required to plot histograms.")

    figure, axes = plt.subplots(len(columns), 1, figsize=(7, 3 * len(columns)))
    axes = np.atleast_1d(axes)
    for axis, column in zip(axes, columns):
        sns.histplot(data=df, x=column, bins=32, ax=axis)
        axis.set_title(f"Histogram of {column}")
    figure.tight_layout()
    plt.show()


def elbowMethod(df, columnsForElbow):
    if not columnsForElbow:
        raise ValueError("At least one column is required for clustering.")

    missing_columns = set(columnsForElbow) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns required for clustering: {sorted(missing_columns)}")

    X = df.loc[:, columnsForElbow].values
    if len(X) < 3:
        raise ValueError("At least three customer rows are required to determine an elbow.")

    inertia = []
    cluster_counts = range(1, min(10, len(X)) + 1)

    for n in cluster_counts:
        model = KMeans(n_clusters=n,
                       init='k-means++',
                       max_iter=500,
                       n_init=10,
                       random_state=42)
        model.fit(X)
        inertia.append(model.inertia_)

    # Find the elbow as the point furthest from the line joining the curve's endpoints.
    x_values = np.asarray(list(cluster_counts), dtype=float)
    x_normalized = (x_values - x_values[0]) / (x_values[-1] - x_values[0])
    inertia_values = np.asarray(inertia, dtype=float)
    inertia_range = np.ptp(inertia_values)
    if inertia_range == 0:
        raise ValueError("The selected columns do not show a measurable clustering elbow.")

    y_normalized = (inertia_values - inertia_values.min()) / inertia_range
    start = np.array([x_normalized[0], y_normalized[0]])
    end = np.array([x_normalized[-1], y_normalized[-1]])
    curve = np.column_stack((x_normalized, y_normalized))
    line = end - start
    offsets = curve - start
    distances = np.abs(line[0] * offsets[:, 1] - line[1] * offsets[:, 0])
    distances /= np.linalg.norm(line)

    max_distance = distances.max()
    if max_distance == 0:
        raise ValueError("The selected columns do not show a measurable clustering elbow.")

    # Treat elbows within 5% of the strongest candidate as equivalent; prefer larger k.
    elbow_candidates = np.flatnonzero(distances >= max_distance * 0.95)
    optimal_k = int(x_values[elbow_candidates[-1]])

    figure, axis = plt.subplots(figsize=(15, 6))
    axis.plot(x_values, inertia_values, 'o-')
    axis.axvline(optimal_k, color='red', linestyle='--',
                 label=f'Optimal clusters: k={optimal_k}')
    axis.scatter([optimal_k], [inertia_values[optimal_k - 1]], color='red', zorder=3)
    axis.set_xlabel('Number of Clusters')
    axis.set_ylabel('Cluster sum of squared distances')
    axis.set_title('Elbow Method')
    axis.legend()
    figure.tight_layout()
    plt.show()
    return optimal_k


def runKMeans(X, cluster):
    model = KMeans(n_clusters=cluster,
                   init='k-means++',
                   max_iter=500,
                   n_init=10,
                   random_state=42)
    y_kmeans = model.fit_predict(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    return y_kmeans, centroids, labels


def visualizeKMeans(X, y_kmeans, cluster, title, xlabel, ylabel, colors):
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("KMeans visualization requires exactly two input columns.")

    figure, axis = plt.subplots(figsize=(10, 10))
    color_map = plt.get_cmap("tab10", cluster)
    for i in range(cluster):
        color = colors[i] if i < len(colors) else color_map(i)
        axis.scatter(X[y_kmeans == i, 0],
                     X[y_kmeans == i, 1],
                     s=100,
                     c=[color],
                     label=f'Cluster {i + 1}')
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend()
    figure.tight_layout()
    plt.show()

def export_df_excel(df, filename):
    if "cluster" not in df.columns:
        raise ValueError("DataFrame must contain a 'cluster' column.")
    if df.empty:
        raise ValueError("Cannot export an empty DataFrame.")
    if df["cluster"].isna().any():
        raise ValueError("The 'cluster' column cannot contain missing values.")

    output_directory = os.path.dirname(os.path.abspath(filename))
    os.makedirs(output_directory, exist_ok=True)

    cluster_values = sorted(df["cluster"].unique())
    cluster_counts = df["cluster"].value_counts().reindex(cluster_values)
    summary = cluster_counts.rename_axis("Cluster").reset_index(name="Customer count")
    summary["Percentage"] = summary["Customer count"] / len(df)

    used_sheet_names = {"Cluster distribution"}
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for cluster_value in cluster_values:
            sheet_name = f"Cluster_{cluster_value}"
            for invalid_character in '[]:*?/\\':
                sheet_name = sheet_name.replace(invalid_character, "_")
            sheet_name = sheet_name[:31] or "Cluster"

            base_name = sheet_name
            suffix = 1
            while sheet_name in used_sheet_names:
                suffix_text = f"_{suffix}"
                sheet_name = f"{base_name[:31 - len(suffix_text)]}{suffix_text}"
                suffix += 1
            used_sheet_names.add(sheet_name)

            df.loc[df["cluster"] == cluster_value].to_excel(
                writer, sheet_name=sheet_name, index=False
            )

        summary_sheet_name = "Cluster distribution"
        summary.to_excel(writer, sheet_name=summary_sheet_name, index=False)
        worksheet = writer.sheets[summary_sheet_name]

        chart = PieChart()
        chart.title = "Customer distribution by cluster"
        chart_data = Reference(
            worksheet, min_col=2, min_row=1, max_row=len(summary) + 1
        )
        chart_categories = Reference(
            worksheet, min_col=1, min_row=2, max_row=len(summary) + 1
        )
        chart.add_data(chart_data, titles_from_data=True)
        chart.set_categories(chart_categories)
        chart.dataLabels = DataLabelList()
        chart.dataLabels.showPercent = True
        chart.height = 9
        chart.width = 14
        worksheet.add_chart(chart, "E2")

    print(f"Exported cluster data to: {os.path.abspath(filename)}")


def export_df_to_html(df, filename):
    """Write a standalone interactive customer segmentation dashboard."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Cannot create a dashboard from an empty DataFrame.")

    # Normalize column names only for matching; retain the original table columns.
    normalized_columns = {
        str(column).strip().lower().replace("_", "").replace(" ", ""): column
        for column in df.columns
    }

    def find_column(*candidates):
        for candidate in candidates:
            key = candidate.lower().replace("_", "").replace(" ", "")
            if key in normalized_columns:
                return normalized_columns[key]
        return None

    cluster_column = find_column("cluster")
    age_column = find_column("age")
    income_column = find_column("annual_income", "annual income")
    spending_column = find_column("spending_score", "spending score")
    missing = [
        name for name, column in (
            ("cluster", cluster_column),
            ("Age", age_column),
            ("Annual Income", income_column),
            ("Spending Score", spending_column),
        ) if column is None
    ]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Work on a copy: discard rows without a cluster label, and coerce chart metrics.
    data = df.copy()
    data = data.loc[data[cluster_column].notna()].copy()
    if data.empty:
        raise ValueError("No rows have a valid cluster label.")

    for column in (age_column, income_column, spending_column):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    try:
        clusters = sorted(data[cluster_column].unique())
    except TypeError:
        clusters = sorted(data[cluster_column].unique(), key=lambda value: str(value))

    cluster_counts = data[cluster_column].value_counts().reindex(clusters, fill_value=0)
    cluster_palette = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    ]
    cluster_colors = {
        cluster: cluster_palette[index % len(cluster_palette)]
        for index, cluster in enumerate(clusters)
    }
    age_groups = ["Thanh niên (≤35)", "Trung niên (35–45)", "Người cao tuổi (>45)"]
    age_colors = {
        age_groups[0]: "#4E79A7",
        age_groups[1]: "#F28E2B",
        age_groups[2]: "#B07AA1",
    }
    data["_age_group"] = pd.cut(
        data[age_column],
        bins=[-np.inf, 35, 45, np.inf],
        labels=age_groups,
        right=True,
    )

    def display_value(value):
        return html.escape(str(value), quote=True)

    def make_scatter(cluster_data, y_column, title, y_label, chart_id):
        chart_data = cluster_data.dropna(subset=[age_column, y_column, "_age_group"])
        if chart_data.empty:
            figure = go.Figure()
            figure.add_annotation(
                text="Không có dữ liệu hợp lệ để vẽ biểu đồ",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            )
        else:
            hover_columns = [
                column for column in data.columns
                if column not in (age_column, y_column, "_age_group")
            ][:5]
            figure = px.scatter(
                chart_data,
                x=age_column,
                y=y_column,
                color="_age_group",
                category_orders={"_age_group": age_groups},
                color_discrete_map=age_colors,
                hover_data=hover_columns,
                labels={
                    age_column: "Độ tuổi (Age)",
                    y_column: y_label,
                    "_age_group": "Nhóm tuổi",
                },
                title=title,
            )
            figure.update_traces(marker={"size": 10, "opacity": 0.82})
        figure.update_layout(
            template="plotly_white",
            height=390,
            margin={"l": 55, "r": 20, "t": 65, "b": 50},
            legend_title_text="Nhóm tuổi",
        )
        return pio.to_html(
            figure,
            full_html=False,
            include_plotlyjs=False,
            div_id=chart_id,
            config={"responsive": True, "displaylogo": False},
        )

    pie = go.Figure(data=[go.Pie(
        labels=[str(cluster) for cluster in clusters],
        values=[int(cluster_counts.loc[cluster]) for cluster in clusters],
        marker={"colors": [cluster_colors[cluster] for cluster in clusters]},
        hole=0.48,
        sort=False,
        texttemplate="%{label}<br>%{value} khách<br>%{percent}",
        hovertemplate="Cụm %{label}<br>%{value} khách hàng<br>%{percent}<extra></extra>",
    )])
    pie.update_layout(
        title="Tỷ lệ khách hàng theo cụm",
        template="plotly_white",
        height=390,
        margin={"l": 20, "r": 20, "t": 65, "b": 20},
        showlegend=True,
        legend_title_text="Cụm",
    )
    pie_html = pio.to_html(
        pie,
        full_html=False,
        include_plotlyjs=False,
        div_id="cluster-distribution",
        config={"responsive": True, "displaylogo": False},
    )

    age_mean = data[age_column].mean()
    income_mean = data[income_column].mean()
    cluster_options = "".join(
        f'<option value="{index}">Cụm {display_value(cluster)} '
        f'({int(cluster_counts.loc[cluster])} khách hàng)</option>'
        for index, cluster in enumerate(clusters)
    )

    sections = []
    for index, cluster in enumerate(clusters):
        cluster_data = data.loc[data[cluster_column] == cluster].copy()
        cluster_age_counts = (
            cluster_data["_age_group"].value_counts().reindex(age_groups, fill_value=0)
        )
        sections.append(f"""
        <section class="cluster-panel" id="cluster-panel-{index}"
                 style="display:{'block' if index == 0 else 'none'}">
          <h3>Cụm {display_value(cluster)}</h3>
          <div class="cluster-kpis">
            <div class="mini-kpi"><span>Số khách hàng</span><strong>{len(cluster_data):,}</strong></div>
            <div class="mini-kpi"><span>Tuổi trung bình</span><strong>{cluster_data[age_column].mean():.1f}</strong></div>
            <div class="mini-kpi"><span>Thu nhập trung bình</span><strong>{cluster_data[income_column].mean():,.2f}</strong></div>
            <div class="mini-kpi"><span>Điểm chi tiêu TB</span><strong>{cluster_data[spending_column].mean():.2f}</strong></div>
          </div>
          <p class="age-counts">Nhóm tuổi:
            Thanh niên {int(cluster_age_counts.iloc[0])} ·
            Trung niên {int(cluster_age_counts.iloc[1])} ·
            Người cao tuổi {int(cluster_age_counts.iloc[2])}
          </p>
          <div class="chart-grid">
            <div class="chart-card">{make_scatter(cluster_data, income_column, f"Cụm {cluster}: Độ tuổi và thu nhập", "Thu nhập hằng năm", f"income-chart-{index}")}</div>
            <div class="chart-card">{make_scatter(cluster_data, spending_column, f"Cụm {cluster}: Độ tuổi và điểm chi tiêu", "Điểm chi tiêu", f"spending-chart-{index}")}</div>
          </div>
        </section>
        """)

    table_data = data.drop(columns=["_age_group"]).sort_values(
        by=cluster_column, key=lambda column: column.map(
            {cluster: index for index, cluster in enumerate(clusters)}
        )
    )
    table_html = table_data.to_html(
        index=False, escape=True, border=0, classes="data-table", na_rep="—"
    )
    table_cluster_options = "".join(
        f'<option value="{display_value(cluster)}">Cụm {display_value(cluster)}</option>'
        for cluster in clusters
    )
    plotly_script = get_plotlyjs()
    output_path = os.path.abspath(filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dashboard = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Customer Segmentation Dashboard</title>
  <style>
    :root {{ --navy:#18324b; --blue:#426b99; --bg:#f3f6fa; --card:#fff; --muted:#68798b; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:#263746; font-family:Arial,sans-serif; }}
    header {{ padding:30px max(22px,calc((100vw - 1320px)/2)); color:#fff;
      background:linear-gradient(120deg,#18324b,#426b99); }}
    header h1 {{ margin:0 0 8px; font-size:clamp(24px,4vw,36px); }}
    header p {{ margin:0; color:#d9e5f0; }}
    main {{ max-width:1320px; margin:24px auto; padding:0 20px 36px; }}
    .kpis,.chart-grid,.cluster-kpis {{ display:grid; gap:16px; }}
    .kpis {{ grid-template-columns:repeat(4,minmax(0,1fr)); margin-bottom:20px; }}
    .kpi,.card,.mini-kpi {{ background:var(--card); border-radius:12px; padding:19px;
      box-shadow:0 3px 14px #1a365014; }}
    .kpi span,.mini-kpi span {{ display:block; color:var(--muted); font-size:13px; }}
    .kpi strong {{ display:block; margin-top:8px; color:var(--navy); font-size:25px; }}
    .card {{ margin-bottom:20px; }}
    .card h2 {{ margin:0 0 16px; color:var(--navy); font-size:20px; }}
    .overview {{ display:grid; grid-template-columns:1fr 1.4fr; gap:18px; align-items:center; }}
    .cluster-kpis {{ grid-template-columns:repeat(4,minmax(0,1fr)); margin:14px 0; }}
    .mini-kpi {{ padding:14px; border:1px solid #edf0f4; box-shadow:none; }}
    .mini-kpi strong {{ display:block; margin-top:7px; color:var(--navy); font-size:19px; }}
    .age-counts {{ color:var(--muted); }}
    .chart-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .chart-card {{ min-width:0; border:1px solid #edf0f4; border-radius:10px; padding:8px; }}
    .controls {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px; }}
    input,select {{ min-height:40px; padding:8px 11px; border:1px solid #ccd6e0;
      border-radius:7px; background:#fff; font-size:14px; }}
    #search {{ flex:1; min-width:220px; }}
    .table-wrap {{ overflow:auto; max-height:620px; border:1px solid #e5eaf0; border-radius:8px; }}
    .data-table {{ width:100%; border-collapse:collapse; white-space:nowrap; font-size:13px; }}
    .data-table th {{ position:sticky; top:0; background:#eaf0f6; color:var(--navy);
      cursor:pointer; text-align:left; }}
    .data-table th,.data-table td {{ padding:10px 12px; border-bottom:1px solid #edf0f4; }}
    .data-table tbody tr:nth-child(even) {{ background:#f8fafc; }}
    .data-table tbody tr:hover {{ background:#eaf2fb; }}
    .note {{ color:var(--muted); font-size:13px; }}
    @media(max-width:850px) {{ .overview {{ grid-template-columns:1fr; }} .kpis {{ grid-template-columns:repeat(2,1fr); }} }}
    @media(max-width:620px) {{ main {{ padding:0 12px 24px; }} .chart-grid,.cluster-kpis {{ grid-template-columns:1fr; }} header {{ padding:24px 18px; }} }}
  </style>
  <script>{plotly_script}</script>
</head>
<body>
  <header><h1>Customer Segmentation Dashboard</h1>
    <p>Phân tích đặc điểm khách hàng theo kết quả phân cụm hiện có</p></header>
  <main>
    <section class="kpis">
      <div class="kpi"><span>Tổng khách hàng</span><strong>{len(data):,}</strong></div>
      <div class="kpi"><span>Tổng số cụm</span><strong>{len(clusters)}</strong></div>
      <div class="kpi"><span>Độ tuổi trung bình</span><strong>{age_mean:.1f}</strong></div>
      <div class="kpi"><span>Thu nhập trung bình</span><strong>{income_mean:,.2f}</strong></div>
    </section>
    <section class="card overview"><div>{pie_html}</div>
      <div><h2>Phân bố khách hàng theo cụm</h2>
        <p>Biểu đồ thể hiện số lượng và tỷ lệ khách hàng tính từ dữ liệu phân cụm.</p>
        <p class="note">Các cụm được sắp xếp tăng dần; màu sắc của từng cụm được giữ thống nhất.</p>
      </div>
    </section>
    <section class="card"><h2>Phân tích đặc điểm từng cụm</h2>
      <label for="cluster-select">Chọn cụm: </label>
      <select id="cluster-select">{cluster_options}</select>
      {''.join(sections)}
    </section>
    <section class="card"><h2>Dữ liệu khách hàng</h2>
      <div class="controls">
        <input id="search" type="search" placeholder="Tìm kiếm trong bảng...">
        <select id="table-cluster"><option value="">Tất cả cụm</option>{table_cluster_options}</select>
      </div>
      <div class="table-wrap">{table_html}</div>
      <p class="note">Nhấp vào tiêu đề cột để sắp xếp. Dữ liệu được hiển thị theo thứ tự cụm tăng dần.</p>
    </section>
  </main>
  <script>
    const clusterSelect = document.getElementById("cluster-select");
    clusterSelect.addEventListener("change", () => {{
      document.querySelectorAll(".cluster-panel").forEach((panel, index) => {{
        panel.style.display = index === Number(clusterSelect.value) ? "block" : "none";
      }});
      window.dispatchEvent(new Event("resize"));
    }});
    const table = document.querySelector(".data-table");
    const search = document.getElementById("search");
    const clusterFilter = document.getElementById("table-cluster");
    function filterRows() {{
      const query = search.value.toLocaleLowerCase();
      const selectedCluster = clusterFilter.value;
      const clusterIndex = Array.from(table.tHead.rows[0].cells)
        .findIndex(cell => cell.textContent.trim().toLowerCase() === {json.dumps(str(cluster_column))}.toLowerCase());
      Array.from(table.tBodies[0].rows).forEach(row => {{
        const matchesText = row.textContent.toLocaleLowerCase().includes(query);
        const matchesCluster = !selectedCluster || row.cells[clusterIndex].textContent.trim() === selectedCluster;
        row.style.display = matchesText && matchesCluster ? "" : "none";
      }});
    }}
    search.addEventListener("input", filterRows);
    clusterFilter.addEventListener("change", filterRows);
    Array.from(table.tHead.rows[0].cells).forEach((headerCell, columnIndex) => {{
      headerCell.title = "Nhấp để sắp xếp";
      headerCell.addEventListener("click", () => {{
        const body = table.tBodies[0];
        const ascending = headerCell.dataset.order !== "asc";
        Array.from(body.rows)
          .sort((rowA, rowB) => {{
            const a = rowA.cells[columnIndex].textContent.trim();
            const b = rowB.cells[columnIndex].textContent.trim();
            const numberA = Number(a.replace(/,/g, ""));
            const numberB = Number(b.replace(/,/g, ""));
            const comparison = a !== "" && b !== "" && Number.isFinite(numberA) && Number.isFinite(numberB)
              ? numberA - numberB : a.localeCompare(b, undefined, {{numeric:true, sensitivity:"base"}});
            return ascending ? comparison : -comparison;
          }})
          .forEach(row => body.appendChild(row));
        headerCell.dataset.order = ascending ? "asc" : "desc";
      }});
    }});
  </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(dashboard)
    print(f"Exported interactive dashboard to: {output_path}")


def visualize3DKMeans(df, columns, labels, filename):
    if len(columns) != 3:
        raise ValueError("3D KMeans visualization requires exactly three columns.")
    missing_columns = set(columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns required for 3D visualization: {sorted(missing_columns)}")
    if len(labels) != len(df):
        raise ValueError("The number of cluster labels must match the number of data rows.")

    chart_data = df.loc[:, columns].copy()
    chart_data["cluster_3d"] = labels
    cluster_values = sorted(chart_data["cluster_3d"].unique())
    fig = px.scatter_3d(
        chart_data,
        x=columns[0],
        y=columns[1],
        z=columns[2],
        color="cluster_3d",
        hover_data=columns,
        category_orders={"cluster_3d": cluster_values},
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={
            columns[0]: columns[0].replace("_", " "),
            columns[1]: columns[1].replace("_", " "),
            columns[2]: columns[2].replace("_", " "),
            "cluster_3d": "Cụm",
        },
        title="Trực quan hóa phân cụm khách hàng 3D",
    )
    fig.update_traces(marker={"size": 5, "opacity": 0.85})
    fig.update_layout(
        template="plotly_white",
        margin={"l": 0, "r": 0, "b": 0, "t": 55},
        legend_title_text="Cụm",
        scene={
            "xaxis_title": columns[0].replace("_", " "),
            "yaxis_title": columns[1].replace("_", " "),
            "zaxis_title": columns[2].replace("_", " "),
        },
    )

    output_path = os.path.abspath(filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path, full_html=True, include_plotlyjs=True)
    print(f"Exported 3D cluster visualization to: {output_path}")
    fig.show()


def main():
    df = CustomerDao.getCustomersForClustering()
    if df is None or df.empty:
        raise RuntimeError("Could not load customer data for clustering.")

    print(df)
    print(df.columns)
    showHistogram(df, list(df.columns[1:]))

    columns = ['Age', 'Spending_Score']
    optimal_k = elbowMethod(df, columns)
    print(f"Optimal number of clusters: {optimal_k}")

    X = df.loc[:, columns].to_numpy()
    y_kmeans, centroids, labels = runKMeans(X, optimal_k)
    print(y_kmeans)
    print(centroids)
    df["cluster"] = labels

    result_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")
    output_file = os.path.join(result_folder, "customer_clusters.xlsx")
    export_df_excel(df, output_file)
    html_output_file = os.path.join(result_folder, "index.html")
    export_df_to_html(df, html_output_file)

    colors = ["red", "green", "blue", "purple", "black", "pink", "orange"]
    visualizeKMeans(
        X,
        y_kmeans,
        optimal_k,
        "Clusters of Customers - Age X Spending Score",
        "Age",
        "Spending Score",
        colors,
    )

    columns_3d = ["Age", "Annual_Income", "Spending_Score"]
    optimal_3d_k = elbowMethod(df, columns_3d)
    X_3d = df.loc[:, columns_3d].to_numpy(dtype=float)
    labels_3d, centroids_3d, _ = runKMeans(X_3d, optimal_3d_k)
    print(f"Optimal number of clusters for 3D visualization: {optimal_3d_k}")
    print("3D cluster centers:")
    print(centroids_3d)

    html_3d_output_file = os.path.join(result_folder, "customer_clusters_3d.html")
    visualize3DKMeans(df, columns_3d, labels_3d, html_3d_output_file)


if __name__ == "__main__":
    main()