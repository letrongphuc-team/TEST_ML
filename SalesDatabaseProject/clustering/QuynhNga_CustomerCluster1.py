import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SalesDatabaseProject.dal.customer_dao import CustomerDao
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.express as px
import mysql.connector

CLUSTER_COLORS = [
    "#4361EE", "#F72585", "#4CC9F0", "#7209B7",
    "#FB8500", "#2A9D8F", "#E63946", "#8D99AE",
]


def showHistogram(df, columns):
    if not columns:
        raise ValueError("Cần ít nhất một cột để vẽ histogram.")

    figure, axes = plt.subplots(len(columns), 1, figsize=(8, 3 * len(columns)))
    axes = np.atleast_1d(axes)
    for axis, column in zip(axes, columns):
        sns.histplot(data=df, x=column, bins=32, kde=True, color="#4361EE", ax=axis)
        axis.set_title(f"Histogram for {column}")
    figure.tight_layout()
    plt.show()


def elbowMethod(df, columnsForElbow, maxK=10, tolerance=0.9):
    if not columnsForElbow:
        raise ValueError("Cần ít nhất một cột để phân cụm.")
    if not 0 <= tolerance <= 1:
        raise ValueError("tolerance phải nằm trong khoảng từ 0 đến 1.")
    missing_columns = set(columnsForElbow) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Thiếu cột dữ liệu: {sorted(missing_columns)}")

    X = df.loc[:, columnsForElbow].to_numpy(dtype=float)
    if len(X) == 0 or not np.isfinite(X).all():
        raise ValueError("Dữ liệu phân cụm rỗng hoặc chứa giá trị không hợp lệ.")
    if maxK < 1:
        raise ValueError("maxK phải lớn hơn hoặc bằng 1.")

    maxK = min(maxK, len(X), len(np.unique(X, axis=0)))
    kRange = np.arange(1, maxK + 1)
    inertia = []
    for n in kRange:
        model = KMeans(n_clusters=n,
                       init='k-means++',
                       max_iter=500,
                       n_init=10,
                       random_state=42)
        model.fit(X)
        inertia.append(model.inertia_)
    inertia = np.array(inertia)

    optimalK = 1
    elbowZone = np.array([1])
    inertiaRange = np.ptp(inertia)
    if len(kRange) > 1 and inertiaRange > 0:
        xNorm = (kRange - kRange.min()) / (kRange.max() - kRange.min())
        yNorm = (inertia - inertia.min()) / inertiaRange
        distance = (1 - xNorm) - yNorm
        maxDistance = distance.max()
        if maxDistance > 0:
            elbowZone = kRange[distance >= tolerance * maxDistance]
            optimalK = int(elbowZone.max())

    print("Inertia theo K:", dict(zip(kRange.tolist(), np.round(inertia, 2).tolist())))
    print("Khoảng elbow:", elbowZone.tolist(), "=> chọn K =", optimalK)

    figure, axis = plt.subplots(figsize=(15, 6))
    axis.plot(kRange, inertia, 'o-', color="#4361EE")
    if len(elbowZone) > 1:
        axis.axvspan(elbowZone.min() - 0.3, elbowZone.max() + 0.3,
                     color="#F72585", alpha=0.15,
                     label=f'Khoảng elbow: K = {elbowZone.min()} - {elbowZone.max()}')
    axis.axvline(x=optimalK, color="#E63946", linestyle='--', linewidth=1.5,
                 label=f'Số cụm tối ưu K = {optimalK}')
    axis.scatter([optimalK], [inertia[optimalK - 1]], color="#E63946", s=80, zorder=3)
    axis.set_xticks(kRange)
    axis.set_xlabel('Number of Clusters')
    axis.set_ylabel('Cluster sum of squared distances')
    axis.set_title('Elbow Method')
    axis.legend()
    figure.tight_layout()
    plt.show()

    return optimalK


def runKMeans(X, cluster):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Dữ liệu K-Means phải là mảng 2 chiều không rỗng.")
    if not np.isfinite(X).all():
        raise ValueError("Dữ liệu K-Means chứa giá trị không hợp lệ.")
    unique_rows = len(np.unique(X, axis=0))
    if cluster < 1 or cluster > min(len(X), unique_rows):
        raise ValueError("Số cụm phải nằm trong khoảng từ 1 đến số điểm dữ liệu khác nhau.")
    model = KMeans(n_clusters=cluster,
                   init='k-means++',
                   max_iter=500,
                   n_init=10,
                   random_state=42)
    labels = model.fit_predict(X)
    return labels, model.cluster_centers_, labels


def visualizeKMeans(X, y_kmeans, cluster, title, xlabel, ylabel, colors):
    X = np.asarray(X)
    y_kmeans = np.asarray(y_kmeans)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("Biểu đồ K-Means cần đúng hai cột dữ liệu.")
    if len(X) != len(y_kmeans):
        raise ValueError("Số nhãn cụm phải khớp với số dòng dữ liệu.")
    if not colors:
        raise ValueError("Cần ít nhất một màu để vẽ các cụm.")
    figure, axis = plt.subplots(figsize=(10, 10))
    for i in range(cluster):
        color = colors[i % len(colors)]
        axis.scatter(X[y_kmeans == i, 0],
                     X[y_kmeans == i, 1],
                     s=70,
                     c=color,
                     label=f'Cluster {i + 1}')
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend()
    figure.tight_layout()
    plt.show()


def export_df_to_excel(df, filename):
    if df.empty or "cluster" not in df.columns:
        raise ValueError("Dữ liệu xuất phải có ít nhất một dòng và cột 'cluster'.")
    from openpyxl.styles import Font, PatternFill, Alignment
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    # Sắp xếp các cụm theo thứ tự tăng dần để dễ quan sát
    clusters = sorted(df["cluster"].unique())
    total = len(df)

    # --- Sheet thống kê phân bố dữ liệu theo từng cụm ---
    # (bỏ qua cột cluster và các cột mã ID vì thống kê chúng không có ý nghĩa)
    numericCols = [c for c in df.select_dtypes(include="number").columns
                   if c != "cluster" and not c.lower().endswith("id")]
    rows = []
    for c in clusters:
        dfCluster = df[df["cluster"] == c]
        row = {
            "Cụm": f"Cluster {c + 1}",
            "Giá trị cột cluster": c,
            "Số lượng": len(dfCluster),
            "Tỷ lệ (%)": round(len(dfCluster) / total * 100, 2),
        }
        # Trung bình / nhỏ nhất / lớn nhất của các cột số trong từng cụm
        for col in numericCols:
            row[f"{col} (TB)"] = round(dfCluster[col].mean(), 2)
            row[f"{col} (Min)"] = dfCluster[col].min()
            row[f"{col} (Max)"] = dfCluster[col].max()
        rows.append(row)
    # Dòng tổng cộng
    totalRow = {"Cụm": "Tổng cộng", "Giá trị cột cluster": "",
                "Số lượng": total, "Tỷ lệ (%)": 100.0}
    for col in numericCols:
        totalRow[f"{col} (TB)"] = round(df[col].mean(), 2)
        totalRow[f"{col} (Min)"] = df[col].min()
        totalRow[f"{col} (Max)"] = df[col].max()
    rows.append(totalRow)
    dfStats = pd.DataFrame(rows)

    headerFont = Font(bold=True, color="FFFFFF")
    headerFill = PatternFill("solid", fgColor="4361EE")

    def formatSheet(ws):
        # Tô đậm tiêu đề, cố định dòng tiêu đề, tự giãn độ rộng cột
        for cell in ws[1]:
            cell.font = headerFont
            cell.fill = headerFill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.freeze_panes = "A2"
        for colCells in ws.columns:
            maxLen = max(len(str(cell.value)) if cell.value is not None else 0 for cell in colCells)
            ws.column_dimensions[colCells[0].column_letter].width = min(maxLen + 3, 40)

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # Mỗi cụm 1 sheet, xuất theo thứ tự cụm
        for c in clusters:
            sheetName = f"Cluster {c + 1}"
            dfCluster = df[df["cluster"] == c]
            dfCluster.to_excel(writer, sheet_name=sheetName, index=False)
            formatSheet(writer.sheets[sheetName])
            print(f"Đã xuất {sheetName}: {len(dfCluster)} dòng")

        # Sheet cuối cùng: thống kê
        dfStats.to_excel(writer, sheet_name="Thống kê", index=False)
        wsStats = writer.sheets["Thống kê"]
        formatSheet(wsStats)
        for cell in wsStats[wsStats.max_row]:   # tô đậm dòng tổng cộng
            cell.font = Font(bold=True)

    print(f"Đã xuất dữ liệu ra file: {filename}")
    print(dfStats)

def export_df_to_html(df, filename):
    if df.empty or "cluster" not in df.columns:
        raise ValueError("Dữ liệu xuất phải có ít nhất một dòng và cột 'cluster'.")
    required_columns = {"Age", "Annual_Income", "Spending_Score"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Thiếu cột cần thiết để xuất báo cáo HTML: {sorted(missing_columns)}")

    dfHtml = df.copy()
    dfHtml["cluster"] = dfHtml["cluster"].astype(int)
    dfHtml["Cụm"] = dfHtml["cluster"].apply(lambda c: f"Cluster {c + 1}")

    # --- Chia nhóm tuổi: <=35 Thanh niên, <=45 Trung niên, còn lại Cao tuổi ---
    def ageGroup(age):
        if age <= 35:
            return "Thanh niên (≤35)"
        elif age <= 45:
            return "Trung niên (36-45)"
        return "Cao tuổi (>45)"
    ageOrder = ["Thanh niên (≤35)", "Trung niên (36-45)", "Cao tuổi (>45)"]
    ageColors = {ageOrder[0]: "#2a9d8f", ageOrder[1]: "#e9c46a", ageOrder[2]: "#e76f51"}
    dfHtml["Nhóm tuổi"] = dfHtml["Age"].apply(ageGroup)

    clusters = sorted(dfHtml["cluster"].unique())
    clusterNames = [f"Cluster {c + 1}" for c in clusters]
    clusterColors = CLUSTER_COLORS
    total = len(dfHtml)
    incomeCol = "Annual_Income"
    scoreCol = "Spending_Score"

    # Biến đếm để chỉ nhúng thư viện plotly.js 1 lần (file html xem được cả khi offline)
    state = {"first": True}

    def figToHtml(fig):
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#151B2B",
            plot_bgcolor="#151B2B",
            font=dict(color="#E6EAF2", family="Segoe UI, Arial, sans-serif"),
            autosize=True,
            margin=dict(l=45, r=20, t=60, b=45),
        )
        out = fig.to_html(full_html=False, include_plotlyjs=state["first"],
                          default_width="100%", default_height="420px",
                          config={"displaylogo": False, "responsive": True})
        state["first"] = False
        return out

    def tableHtml(data):
        return data.to_html(index=False, classes="data-table", border=0,
                            float_format=lambda x: f"{x:,.2f}")

    # --- 1) Biểu đồ tỷ lệ khách hàng theo cụm ---
    dfCount = dfHtml.groupby("Cụm").size().reindex(clusterNames).reset_index(name="Số khách hàng")
    figPie = px.pie(dfCount, names="Cụm", values="Số khách hàng", hole=0.45,
                    title="Tỷ lệ khách hàng theo cụm",
                    color="Cụm", color_discrete_sequence=clusterColors,
                    category_orders={"Cụm": clusterNames})
    figPie.update_traces(textinfo="percent+value", sort=False,
                         marker=dict(line=dict(color="#151B2B", width=3)))

    # --- 2) Bảng tóm tắt từng cụm ---
    dfSummary = dfHtml.groupby("Cụm").agg(
        **{"Số khách hàng": ("cluster", "size"),
           "Tuổi TB": ("Age", "mean"),
           "Thu nhập TB": (incomeCol, "mean"),
           "Spending Score TB": (scoreCol, "mean")}).reindex(clusterNames).reset_index()
    dfSummary.insert(2, "Tỷ lệ (%)", dfSummary["Số khách hàng"] / total * 100)
    for g in ageOrder:
        dfSummary[g] = dfHtml[dfHtml["Nhóm tuổi"] == g].groupby("Cụm").size() \
            .reindex(clusterNames).fillna(0).astype(int).values

    overviewCharts = f'<div class="chart chart-overview">{figToHtml(figPie)}</div>'

    # --- 3) Mỗi cụm: Độ tuổi vs Income, Độ tuổi vs Spending Score ---
    clusterSections = ""
    for c, name, color in zip(
            clusters, clusterNames,
            [clusterColors[i % len(clusterColors)] for i in range(len(clusters))]):
        dfC = dfHtml[dfHtml["cluster"] == c]
        dfPlot = dfC.copy()
        agePositions = {group: position for position, group in enumerate(ageOrder)}
        dfPlot["_age_position"] = dfPlot["Nhóm tuổi"].map(agePositions).astype(float)
        for group, groupRows in dfPlot.groupby("Nhóm tuổi", sort=False):
            offsets = np.linspace(-0.18, 0.18, len(groupRows)) if len(groupRows) > 1 else [0]
            dfPlot.loc[groupRows.index, "_age_position"] += offsets

        charts = ""
        for col, label in [(incomeCol, "Thu nhập (Annual Income)"),
                           (scoreCol, "Spending Score")]:
            chartOptions = {
                "x": "_age_position",
                "y": col,
                "color": "Nhóm tuổi",
                "hover_data": ["CustomerId", "Age"] if "CustomerId" in dfC else ["Age"],
                "title": f"{name}: Nhóm tuổi vs {label}",
                "category_orders": {"Nhóm tuổi": ageOrder},
                "color_discrete_map": ageColors,
                "labels": {"_age_position": "Nhóm tuổi", col: label},
            }
            fig = px.scatter(dfPlot, **chartOptions)
            fig.update_traces(
                mode="markers",
                marker_size=10,
                marker_opacity=0.82,
                marker_line_width=1,
                marker_line_color="#151B2B",
            )
            fig.update_layout(
                showlegend=False,
                title=dict(x=0.04, xanchor="left", font=dict(size=16)),
                xaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(ageOrder))),
                    ticktext=ageOrder,
                    range=[-0.5, len(ageOrder) - 0.5],
                    showgrid=False,
                    zeroline=False,
                    tickfont=dict(color="#B8C3D9"),
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#29334A",
                    zeroline=False,
                    tickfont=dict(color="#B8C3D9"),
                    title_font=dict(color="#E6EAF2"),
                ),
            )
            charts += f'<div class="chart">{figToHtml(fig)}</div>'

        ageCounts = dfC["Nhóm tuổi"].value_counts().reindex(ageOrder).fillna(0).astype(int)
        ageText = " · ".join(f"{g}: <b>{n}</b>" for g, n in ageCounts.items())
        clusterSections += f"""
        <section class="cluster" style="border-top-color:{color}">
          <h2><span class="dot" style="background:{color}"></span>{name}
              <small>{len(dfC)} khách hàng ({len(dfC) / total * 100:.1f}%)</small></h2>
          <p class="meta">Tuổi TB <b>{dfC['Age'].mean():.1f}</b> ·
             Thu nhập TB <b>{dfC[incomeCol].mean():,.2f}</b> ·
             Spending Score TB <b>{dfC[scoreCol].mean():.1f}</b><br>{ageText}</p>
          <div class="grid">{charts}</div>
          <details><summary>Xem danh sách khách hàng của {name}</summary>
            {tableHtml(dfC.drop(columns=["cluster"]))}</details>
        </section>"""

    # --- 4) Ghép thành trang HTML ---
    page = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phân cụm khách hàng</title>
<style>
  :root {{ color-scheme: dark; --bg: #0B1020; --panel: #151B2B; --panel-soft: #1B2336;
           --text: #E6EAF2; --muted: #9AA6BD; --line: #29334A; --accent: #7C9CFF; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 0; min-height: 100vh;
          background: radial-gradient(ellipse at 15% 0%, #1A2850 0, var(--bg) 45%);
          color: var(--text); }}
  header {{ padding: 48px max(24px, calc((100% - 1280px) / 2)); border-bottom: 1px solid var(--line);
            background: linear-gradient(115deg, rgba(67,97,238,.22), rgba(21,27,43,.78)); }}
  header h1 {{ margin: 0 0 10px; font-size: clamp(28px, 4vw, 42px); letter-spacing: -.5px; }}
  header p {{ margin: 0; color: var(--muted); }}
  main {{ max-width: 1280px; margin: 0 auto; padding: 30px 24px 56px; }}
  .eyebrow {{ color: #91A9FF; text-transform: uppercase; letter-spacing: 2px; font-size: 12px;
              font-weight: 700; margin-bottom: 10px; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin-bottom: 22px; }}
  .card {{ background: linear-gradient(145deg, var(--panel-soft), var(--panel)); border: 1px solid var(--line);
          border-radius: 16px; padding: 20px 22px; color: var(--muted); box-shadow: 0 12px 30px #0003; }}
  .card b {{ display: block; margin-top: 8px; font-size: 30px; color: var(--text); }}
  section {{ background: rgba(21,27,43,.92); border: 1px solid var(--line); border-radius: 18px;
            padding: 24px; margin-bottom: 22px; box-shadow: 0 16px 40px #0003; }}
  section.cluster {{ border-top: 3px solid; }}
  h2 {{ margin: 0 0 16px; font-size: 22px; }}
  h2 small {{ display: inline-block; font-weight: normal; color: var(--muted); font-size: 14px; margin-left: 8px; }}
  .dot {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 10px; }}
  .meta {{ color: var(--muted); line-height: 1.9; margin: 0 0 14px; }}
  .meta b {{ color: var(--text); }}
  .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
  .chart {{ min-width: 0; overflow: hidden; border: 1px solid var(--line); border-radius: 14px;
            background: var(--panel); }}
  .chart-overview {{ max-width: 680px; margin: 0 auto; }}
  .data-table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 14px;
                 color: var(--text); white-space: nowrap; }}
  .data-table th {{ background: #253252; color: #DCE5FF; padding: 11px 12px; position: sticky; top: 0;
                    text-align: left; border-bottom: 1px solid #3A496B; }}
  .data-table td {{ padding: 9px 12px; border-bottom: 1px solid var(--line); text-align: left; }}
  .data-table tr:nth-child(even) td {{ background: #192135; }}
  details {{ margin-top: 16px; }}
  details > div, details table {{ max-height: 400px; overflow: auto; display: block; }}
  summary {{ cursor: pointer; color: #AFC1FF; font-weight: 600; padding: 12px 0; }}
  summary:hover {{ color: #fff; }}
  @media (max-width: 800px) {{ .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .grid {{ grid-template-columns: 1fr; }} main {{ padding: 20px 14px 40px; }}
    header {{ padding: 34px 18px; }} section {{ padding: 18px; }} }}
  @media (max-width: 460px) {{ .cards {{ grid-template-columns: 1fr 1fr; gap: 10px; }}
    .card {{ padding: 15px; }} .card b {{ font-size: 23px; }} h2 small {{ display: block; margin: 8px 0 0 22px; }} }}
</style>
</head>
<body>
<header>
  <div class="eyebrow">Customer analytics</div>
  <h1>Phân cụm khách hàng (K-Means)</h1>
  <p>Nhóm tuổi: Thanh niên ≤ 35 · Trung niên 36-45 · Cao tuổi &gt; 45</p>
</header>
<main>
  <div class="cards">
    <div class="card">Tổng khách hàng<b>{total:,}</b></div>
    <div class="card">Số cụm<b>{len(clusters)}</b></div>
    <div class="card">Tuổi trung bình<b>{dfHtml['Age'].mean():.1f}</b></div>
    <div class="card">Spending Score TB<b>{dfHtml[scoreCol].mean():.1f}</b></div>
  </div>
  <section>
    <div class="eyebrow">Overview</div>
    <h2>Tổng quan các cụm</h2>
    <div class="grid">{overviewCharts}</div>
    {tableHtml(dfSummary)}
  </section>
  {clusterSections}
  <section>
    <div class="eyebrow">Customer data</div>
    <h2>Toàn bộ dữ liệu phân cụm</h2>
    <details><summary>Xem tất cả {total} khách hàng</summary>
      {tableHtml(dfHtml.sort_values(["cluster"]).drop(columns=["cluster"]))}</details>
  </section>
</main>
<script>
  // Vẽ lại các biểu đồ theo đúng kích thước khung sau khi trang tải xong
  window.addEventListener("load", function () {{
    document.querySelectorAll(".js-plotly-plot").forEach(function (d) {{ Plotly.Plots.resize(d); }});
  }});
</script>
</body>
</html>"""

    folder = os.path.dirname(os.path.abspath(filename))
    os.makedirs(folder, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Đã xuất báo cáo HTML ra file: {os.path.abspath(filename)}")


def loadCustomersFromWorkbook(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Không tìm thấy workbook dữ liệu dự phòng: {filename}")

    required_columns = {"Age", "Annual_Income", "Spending_Score"}
    frames = []
    with pd.ExcelFile(filename) as workbook:
        for sheet_name in workbook.sheet_names:
            if not sheet_name.lower().startswith("cluster"):
                continue
            sheet = pd.read_excel(workbook, sheet_name=sheet_name)
            if required_columns.issubset(sheet.columns):
                frames.append(sheet.drop(columns=["cluster"], errors="ignore"))

    if not frames:
        raise ValueError(
            f"Workbook '{filename}' không có sheet cụm chứa đủ cột "
            f"{sorted(required_columns)}."
        )
    return pd.concat(frames, ignore_index=True)


def main():
    try:
        df = CustomerDao.getCustomersForClustering()
    except mysql.connector.Error as error:
        if error.errno in (2002, 2003):
            results_folder = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../results"
            )
            backup_file = os.path.join(results_folder, "customer_clusters.xlsx")
            print(
                "CẢNH BÁO: Không kết nối được MySQL tại localhost:3306. "
                "Đang dùng dữ liệu dự phòng từ workbook; dữ liệu có thể đã cũ.",
                file=sys.stderr,
            )
            try:
                df = loadCustomersFromWorkbook(backup_file)
            except (FileNotFoundError, ValueError) as backup_error:
                raise SystemExit(
                    f"{backup_error}\nKhởi động MySQL hoặc cung cấp workbook "
                    "customer_clusters.xlsx có các sheet cụm chứa dữ liệu khách hàng."
                ) from None
        else:
            raise

    if df is None or df.empty:
        raise RuntimeError("Không tải được dữ liệu khách hàng để phân cụm.")

    columns = ["Age", "Spending_Score"]
    missing_columns = set(columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Thiếu cột dữ liệu cần phân cụm: {sorted(missing_columns)}")

    print(df)
    print(df.columns)
    showHistogram(df, list(df.select_dtypes(include="number").columns))

    cluster = elbowMethod(df, columns)
    print("Bạn nên gom cụm thành =", cluster, "cụm")
    X = df.loc[:, columns].to_numpy(dtype=float)
    labels, centroids, _ = runKMeans(X, cluster)
    print(labels)
    print(centroids)
    df["cluster"] = labels

    visualizeKMeans(
        X, labels, cluster,
        "Clusters of Customers - Age X Spending Score",
        "Age", "Spending Score", CLUSTER_COLORS,
    )

    results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")
    export_df_to_excel(df, os.path.join(results_folder, "CustomerClusters.xlsx"))
    export_df_to_html(df, os.path.join(results_folder, "index.html"))


if __name__ == "__main__":
    main()
