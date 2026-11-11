import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "Chinook" / "data" / "Chinook_Sqlite.sqlite"
RESULTS_DIR = PROJECT_ROOT / "Chinook" / "results"
FORECAST_HORIZON = 12
BACKTEST_HORIZON = 12
LAG_PERIODS = (1, 2, 3, 6, 12)
NUMERIC_FEATURES = [
    "time_index",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "mean_3",
    "mean_12",
]
RANDOM_STATE = 42


def load_sales_data():
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy cơ sở dữ liệu: {DATABASE_PATH}")

    with sqlite3.connect(DATABASE_PATH) as connection:
        tracks = pd.read_sql_query(
            """
            SELECT t.TrackId AS track_id,
                   t.Name AS track_name,
                   COALESCE(t.AlbumId, 0) AS album_id,
                   COALESCE(a.Title, 'Không rõ album') AS album_name,
                   COALESCE(ar.Name, 'Không rõ nghệ sĩ') AS artist_name
            FROM Track t
            LEFT JOIN Album a ON a.AlbumId = t.AlbumId
            LEFT JOIN Artist ar ON ar.ArtistId = a.ArtistId
            ORDER BY t.TrackId
            """,
            connection,
        )
        sales = pd.read_sql_query(
            """
            SELECT strftime('%Y-%m', i.InvoiceDate) AS month,
                   il.TrackId AS track_id,
                   SUM(il.UnitPrice * il.Quantity) AS revenue
            FROM Invoice i
            JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
            GROUP BY strftime('%Y-%m', i.InvoiceDate), il.TrackId
            ORDER BY month, il.TrackId
            """,
            connection,
        )

    if tracks.empty or sales.empty:
        raise ValueError("Cơ sở dữ liệu không có dữ liệu track hoặc doanh thu.")
    if sales["month"].isna().any() or not np.isfinite(sales["revenue"]).all():
        raise ValueError("Dữ liệu hóa đơn có tháng hoặc doanh thu không hợp lệ.")

    return tracks, sales


def build_monthly_matrix(sales, entity_ids, periods):
    entity_positions = {entity_id: index for index, entity_id in enumerate(entity_ids)}
    period_positions = {str(period): index for index, period in enumerate(periods)}
    matrix = np.zeros((len(entity_ids), len(periods)), dtype=float)
    rows = sales["track_id"].map(entity_positions).to_numpy()
    columns = sales["month"].map(period_positions).to_numpy()
    if pd.isna(rows).any() or pd.isna(columns).any():
        raise ValueError("Không thể ánh xạ một số giao dịch vào track hoặc tháng.")
    np.add.at(matrix, (rows.astype(int), columns.astype(int)), sales["revenue"].to_numpy())
    return matrix


def make_features(matrix, entity_ids, start_index, stop_index):
    features = []
    targets = []
    entity_count = len(entity_ids)

    for time_index in range(start_index, stop_index):
        month_number = time_index % 12 + 1
        rows = {
            "entity_id": entity_ids,
            "time_index": np.full(entity_count, time_index, dtype=int),
            "month_sin": np.full(
                entity_count, np.sin(2 * np.pi * month_number / 12), dtype=float
            ),
            "month_cos": np.full(
                entity_count, np.cos(2 * np.pi * month_number / 12), dtype=float
            ),
        }
        for lag in LAG_PERIODS:
            rows[f"lag_{lag}"] = matrix[:, time_index - lag]
        rows["mean_3"] = matrix[:, time_index - 3:time_index].mean(axis=1)
        rows["mean_12"] = matrix[:, time_index - 12:time_index].mean(axis=1)
        features.append(pd.DataFrame(rows))
        targets.append(matrix[:, time_index])

    if not features:
        raise ValueError("Không đủ dữ liệu lịch sử để tạo đặc trưng huấn luyện.")
    return pd.concat(features, ignore_index=True), np.concatenate(targets)


def make_model():
    preprocessing = ColumnTransformer(
        transformers=[
            ("track", OneHotEncoder(handle_unknown="ignore"), ["entity_id"]),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("features", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=14,
                    min_samples_leaf=3,
                    max_features=0.1,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def predict_next_month(model, matrix, entity_ids, time_index):
    month_number = time_index % 12 + 1
    data = {
        "entity_id": entity_ids,
        "time_index": np.full(len(entity_ids), time_index, dtype=int),
        "month_sin": np.full(
            len(entity_ids), np.sin(2 * np.pi * month_number / 12), dtype=float
        ),
        "month_cos": np.full(
            len(entity_ids), np.cos(2 * np.pi * month_number / 12), dtype=float
        ),
    }
    for lag in LAG_PERIODS:
        data[f"lag_{lag}"] = matrix[:, time_index - lag]
    data["mean_3"] = matrix[:, time_index - 3:time_index].mean(axis=1)
    data["mean_12"] = matrix[:, time_index - 12:time_index].mean(axis=1)
    features = pd.DataFrame(data)
    return np.maximum(model.predict(features), 0)


def forecast_recursively(model, historical_matrix, entity_ids, start_index, end_index):
    extended = np.zeros(
        (len(entity_ids), end_index), dtype=float
    )
    extended[:, :start_index] = historical_matrix[:, :start_index]
    for time_index in range(start_index, end_index):
        extended[:, time_index] = predict_next_month(
            model, extended, entity_ids, time_index
        )
    return extended[:, start_index:end_index]


def reconcile_to_total(track_forecasts, total_forecasts):
    track_totals = track_forecasts.sum(axis=0)
    total_forecasts = np.asarray(total_forecasts, dtype=float)
    if np.any((track_totals == 0) & (total_forecasts > 0)):
        raise ValueError("Không thể phân bổ tổng dự báo vào các track có dự báo bằng 0.")
    scale = np.divide(
        total_forecasts,
        track_totals,
        out=np.zeros_like(total_forecasts),
        where=track_totals > 0,
    )
    return track_forecasts * scale


def roll_up_tracks(track_matrix, track_metadata, album_metadata):
    album_positions = {
        album_id: index for index, album_id in enumerate(album_metadata["album_id"])
    }
    track_album_positions = track_metadata["album_id"].map(album_positions).to_numpy()
    if pd.isna(track_album_positions).any():
        raise ValueError("Không thể ánh xạ một số track vào album.")
    album_matrix = np.zeros(
        (len(album_metadata), track_matrix.shape[1]), dtype=float
    )
    np.add.at(album_matrix, track_album_positions.astype(int), track_matrix)
    return album_matrix


def calculate_metrics(actual, predicted, level):
    actual_flat = np.asarray(actual, dtype=float).ravel()
    predicted_flat = np.asarray(predicted, dtype=float).ravel()
    absolute_errors = np.abs(actual_flat - predicted_flat)
    actual_total = actual_flat.sum()
    metrics = {
        "Cấp dự báo": level,
        "MAE": mean_absolute_error(actual_flat, predicted_flat),
        "RMSE": np.sqrt(mean_squared_error(actual_flat, predicted_flat)),
        "WAPE (%)": (
            absolute_errors.sum() / actual_total * 100 if actual_total > 0 else np.nan
        ),
    }
    metrics["R²"] = (
        r2_score(actual_flat, predicted_flat)
        if len(actual_flat) > 1 and np.ptp(actual_flat) > 0
        else np.nan
    )
    return metrics


def make_forecast_tables(
    tracks, albums, track_matrix, album_matrix, periods, history_length
):
    future_periods = periods[history_length:]
    track_forecasts = pd.DataFrame(
        {
            "TrackId": np.repeat(tracks["track_id"].to_numpy(), len(future_periods)),
            "Track": np.repeat(tracks["track_name"].to_numpy(), len(future_periods)),
            "AlbumId": np.repeat(tracks["album_id"].to_numpy(), len(future_periods)),
            "Album": np.repeat(tracks["album_name"].to_numpy(), len(future_periods)),
            "Artist": np.repeat(tracks["artist_name"].to_numpy(), len(future_periods)),
            "Tháng": np.tile([str(period) for period in future_periods], len(tracks)),
            "Doanh thu dự báo (USD)": track_matrix[:, history_length:].ravel(),
        }
    )
    album_forecasts = pd.DataFrame(
        {
            "AlbumId": np.repeat(albums["album_id"].to_numpy(), len(future_periods)),
            "Album": np.repeat(albums["album_name"].to_numpy(), len(future_periods)),
            "Artist": np.repeat(albums["artist_name"].to_numpy(), len(future_periods)),
            "Tháng": np.tile([str(period) for period in future_periods], len(albums)),
            "Doanh thu dự báo (USD)": album_matrix[:, history_length:].ravel(),
        }
    )
    return track_forecasts, album_forecasts


def make_summary_tables(
    tracks, albums, track_matrix, album_matrix, periods, history_length
):
    track_summary = tracks.copy()
    track_summary["Doanh thu lịch sử (USD)"] = track_matrix[:, :history_length].sum(axis=1)
    track_summary["Dự báo 12 tháng (USD)"] = track_matrix[:, history_length:].sum(axis=1)
    track_summary = track_summary.rename(
        columns={
            "track_id": "TrackId",
            "track_name": "Track",
            "album_name": "Album",
            "artist_name": "Artist",
        }
    )

    album_summary = albums.copy()
    album_summary["Doanh thu lịch sử (USD)"] = album_matrix[:, :history_length].sum(axis=1)
    album_summary["Dự báo 12 tháng (USD)"] = album_matrix[:, history_length:].sum(axis=1)
    album_summary = album_summary.rename(
        columns={"album_name": "Album", "artist_name": "Artist"}
    )
    return track_summary, album_summary


def make_plotly_html(fig, include_plotlyjs):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#151B2B",
        plot_bgcolor="#151B2B",
        font=dict(color="#E6EAF2", family="Segoe UI, Arial, sans-serif"),
        autosize=True,
        margin=dict(l=50, r=24, t=64, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        default_width="100%",
        default_height="410px",
        config={"displaylogo": False, "responsive": True},
    )


def table_html(data):
    return data.to_html(
        index=False,
        classes="data-table",
        border=0,
        escape=True,
        float_format=lambda value: f"{value:,.2f}",
    )


def export_html(
    filename,
    tracks,
    sold_track_count,
    albums,
    periods,
    total_history,
    total_backtest,
    total_future,
    metrics,
    track_summary,
    album_summary,
):
    colors = {
        "actual": "#4CC9F0",
        "backtest": "#F72585",
        "forecast": "#A78BFA",
    }
    history_length = len(total_history)
    backtest_start = history_length - len(total_backtest)
    chart_state = {"first": True}

    def chart(fig):
        result = make_plotly_html(fig, chart_state["first"])
        chart_state["first"] = False
        return result

    timeline = go.Figure()
    timeline.add_trace(
        go.Scatter(
            x=[str(period) for period in periods[:history_length]],
            y=total_history,
            mode="lines+markers",
            name="Doanh thu thực tế",
            line=dict(color=colors["actual"], width=2.5),
            marker=dict(size=5),
        )
    )
    timeline.add_trace(
        go.Scatter(
            x=[str(period) for period in periods[backtest_start:history_length]],
            y=total_backtest,
            mode="lines+markers",
            name="Dự báo kiểm định",
            line=dict(color=colors["backtest"], width=2, dash="dash"),
            marker=dict(size=6),
        )
    )
    timeline.add_trace(
        go.Scatter(
            x=[str(period) for period in periods[history_length:]],
            y=total_future,
            mode="lines+markers",
            name="Dự báo 12 tháng",
            line=dict(color=colors["forecast"], width=3),
            marker=dict(size=6),
        )
    )
    timeline.update_layout(
        title="Doanh thu theo tháng: lịch sử, kiểm định và dự báo",
        xaxis_title="Tháng",
        yaxis_title="Doanh thu (USD)",
        hovermode="x unified",
    )
    timeline.update_xaxes(showgrid=False)
    timeline.update_yaxes(gridcolor="#29334A", tickformat="$,.0f")

    track_top = track_summary.nlargest(10, "Dự báo 12 tháng (USD)").sort_values(
        "Dự báo 12 tháng (USD)"
    )
    album_top = album_summary.nlargest(10, "Dự báo 12 tháng (USD)").sort_values(
        "Dự báo 12 tháng (USD)"
    )

    def horizontal_bar(data, name_column, title):
        fig = go.Figure(
            go.Bar(
                x=data["Dự báo 12 tháng (USD)"],
                y=data[name_column],
                orientation="h",
                marker_color="#7C9CFF",
                hovertemplate="%{y}<br>$%{x:,.2f}<extra></extra>",
            )
        )
        fig.update_layout(title=title, xaxis_title="Dự báo 12 tháng (USD)")
        fig.update_xaxes(gridcolor="#29334A", tickformat="$,.0f")
        fig.update_yaxes(showgrid=False)
        return fig

    total_forecast = float(total_future.sum())
    total_historical = float(total_history.sum())
    total_wape = float(metrics.loc[metrics["Cấp dự báo"] == "Tổng doanh thu", "WAPE (%)"].iloc[0])
    evaluation_table = table_html(metrics)
    track_display = track_top.rename(
        columns={"track_name": "Track", "album_name": "Album", "artist_name": "Artist"}
    )[["Track", "Album", "Artist", "Dự báo 12 tháng (USD)"]]
    album_display = album_top.rename(
        columns={"album_name": "Album", "artist_name": "Artist"}
    )[["Album", "Artist", "Dự báo 12 tháng (USD)"]]

    page = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dự báo doanh thu Chinook</title>
<style>
  :root {{ color-scheme: dark; --bg:#0B1020; --panel:#151B2B; --soft:#1B2336;
    --text:#E6EAF2; --muted:#9AA6BD; --line:#29334A; --accent:#91A9FF; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; min-height:100vh; color:var(--text); font-family:"Segoe UI",Arial,sans-serif;
    background:radial-gradient(ellipse at 15% 0%,#1A2850 0,var(--bg) 46%); }}
  header {{ padding:46px max(24px,calc((100% - 1280px)/2)); border-bottom:1px solid var(--line);
    background:linear-gradient(115deg,rgba(67,97,238,.24),rgba(21,27,43,.82)); }}
  header h1 {{ margin:0 0 10px; font-size:clamp(28px,4vw,42px); letter-spacing:-.5px; }}
  header p {{ margin:0; color:var(--muted); line-height:1.7; }}
  main {{ max-width:1280px; margin:0 auto; padding:30px 24px 58px; }}
  .eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:2px;
    font-size:12px; font-weight:700; margin-bottom:10px; }}
  .cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin-bottom:22px; }}
  .card {{ background:linear-gradient(145deg,var(--soft),var(--panel)); border:1px solid var(--line);
    border-radius:16px; padding:20px 22px; color:var(--muted); box-shadow:0 12px 30px #0003; }}
  .card b {{ display:block; margin-top:8px; font-size:clamp(22px,3vw,30px); color:var(--text); }}
  section {{ background:rgba(21,27,43,.94); border:1px solid var(--line); border-radius:18px;
    padding:24px; margin-bottom:22px; box-shadow:0 16px 40px #0003; }}
  h2 {{ margin:0 0 8px; font-size:22px; }}
  .subtitle,.muted {{ color:var(--muted); line-height:1.75; }}
  .chart {{ min-width:0; overflow:hidden; border:1px solid var(--line); border-radius:14px;
    background:var(--panel); margin-top:16px; }}
  .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
  .data-table {{ border-collapse:collapse; width:100%; font-size:13px; margin-top:14px;
    color:var(--text); white-space:nowrap; }}
  .data-table th {{ background:#253252; color:#DCE5FF; padding:11px 12px; position:sticky;
    top:0; text-align:left; border-bottom:1px solid #3A496B; }}
  .data-table td {{ padding:9px 12px; border-bottom:1px solid var(--line); text-align:left; }}
  .data-table tr:nth-child(even) td {{ background:#192135; }}
  .table-wrap {{ max-height:440px; overflow:auto; margin-top:14px; border:1px solid var(--line);
    border-radius:12px; }}
  .table-wrap .data-table {{ margin:0; }}
  details {{ margin-top:14px; }}
  summary {{ cursor:pointer; color:#AFC1FF; font-weight:600; padding:12px 0; }}
  a {{ color:#AFC1FF; }} .downloads {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:12px; }}
  .step {{ border-left:3px solid #4361EE; padding:2px 0 2px 14px; margin:14px 0; color:var(--muted); }}
  .step strong {{ color:var(--text); }}
  code {{ color:#C7D2FE; background:#202A40; border-radius:5px; padding:2px 6px; }}
  @media(max-width:850px) {{ .cards {{grid-template-columns:repeat(2,minmax(0,1fr));}}
    .grid {{grid-template-columns:1fr;}} main {{padding:22px 14px 42px;}}
    header {{padding:34px 18px;}} section {{padding:18px;}} }}
</style>
</head>
<body>
<header>
  <div class="eyebrow">Chinook · Machine Learning</div>
  <h1>Dự báo doanh thu âm nhạc</h1>
  <p>Dữ liệu hóa đơn từ {periods[0]} đến {periods[history_length - 1]} ·
    Dự báo {FORECAST_HORIZON} tháng tiếp theo · Đơn vị: USD</p>
</header>
<main>
  <div class="cards">
    <div class="card">Doanh thu lịch sử<b>${total_historical:,.2f}</b></div>
    <div class="card">Dự báo 12 tháng<b>${total_forecast:,.2f}</b></div>
    <div class="card">Track trong danh mục<b>{len(tracks):,}</b></div>
    <div class="card">WAPE tổng (kiểm định)<b>{total_wape:.1f}%</b></div>
  </div>
  <section>
    <div class="eyebrow">01 · Tổng quan dự báo</div>
    <h2>Doanh thu theo tháng</h2>
    <p class="subtitle">Dải màu hồng là dự báo trên 12 tháng cuối để kiểm định mô hình;
      màu tím là dự báo tương lai. Giá trị tương lai không phải doanh thu thực tế.</p>
    <div class="chart">{chart(timeline)}</div>
  </section>
  <section>
    <div class="eyebrow">02 · Phân rã doanh thu</div>
    <h2>Track và album dự kiến đóng góp cao nhất</h2>
    <p class="subtitle">Dự báo album và tổng doanh thu được cộng từ dự báo từng track,
      vì vậy các cấp kết quả luôn nhất quán.</p>
    <div class="grid">
      <div class="chart">{chart(horizontal_bar(track_top, "Track", "Top 10 track"))}</div>
      <div class="chart">{chart(horizontal_bar(album_top, "Album", "Top 10 album"))}</div>
    </div>
  </section>
  <section>
    <div class="eyebrow">03 · Đánh giá mô hình</div>
    <h2>Backtest theo thời gian</h2>
    <p class="subtitle">Huấn luyện trên các tháng trước 12 tháng cuối, sau đó dự báo cuốn chiếu
      qua từng tháng kiểm định. WAPE = tổng sai số tuyệt đối / tổng doanh thu thực tế.
      WAPE phù hợp hơn MAPE khi nhiều track có tháng không phát sinh doanh thu.</p>
    {evaluation_table}
    <p class="muted">Dữ liệu bán theo track khá thưa: chỉ {sold_track_count:,}
      track từng có giao dịch và nhiều tháng có doanh thu bằng 0. Vì vậy, sai số cấp track/album
      cao hơn cấp tổng; xem các dự báo chi tiết như ước lượng tham khảo, không phải cam kết bán hàng.</p>
  </section>
  <div class="grid">
    <section>
      <div class="eyebrow">04 · Track</div>
      <h2>Top 10 track theo dự báo</h2>
      <div class="table-wrap">{table_html(track_display)}</div>
    </section>
    <section>
      <div class="eyebrow">05 · Album</div>
      <h2>Top 10 album theo dự báo</h2>
      <div class="table-wrap">{table_html(album_display)}</div>
    </section>
  </div>
  <section>
    <div class="eyebrow">06 · Cách đọc và thiết kế báo cáo</div>
    <h2>Quy trình phân tích</h2>
    <div class="step"><strong>1. Nguồn dữ liệu:</strong> nối <code>Invoice</code> và
      <code>InvoiceLine</code>; doanh thu = <code>UnitPrice × Quantity</code>.</div>
    <div class="step"><strong>2. Chuẩn hóa thời gian:</strong> cộng doanh thu theo tháng và track,
      bổ sung tháng không bán thành 0.</div>
    <div class="step"><strong>3. Huấn luyện:</strong> hai Random Forest hồi quy cho track và tổng
      doanh thu; đặc trưng gồm mã track dạng one-hot, tháng trong năm, xu hướng, doanh thu trễ
      1/2/3/6/12 tháng cùng trung bình trượt.</div>
    <div class="step"><strong>4. Kiểm định:</strong> giữ riêng 12 tháng cuối, dự báo cuốn chiếu
      để không dùng doanh thu tương lai làm đầu vào.</div>
    <div class="step"><strong>5. Tổng hợp:</strong> cân chỉnh dự báo track theo dự báo tổng từng tháng,
      sau đó cộng track lên album; tổng và các cấp chi tiết luôn nhất quán.</div>
    <p class="muted">Trang dùng các thẻ KPI, biểu đồ, bảng và CSS responsive để minh họa bố cục
      dashboard. Mở file HTML trực tiếp bằng trình duyệt; thư viện biểu đồ đã được nhúng offline.</p>
    <div class="downloads">
      <a href="revenue_forecast.xlsx">Tải workbook Excel</a>
      <a href="monthly_forecast.csv">Tải dự báo tổng theo tháng (CSV)</a>
      <a href="track_forecast.csv">Tải dự báo từng track (CSV)</a>
      <a href="album_forecast.csv">Tải dự báo từng album (CSV)</a>
    </div>
  </section>
  <section>
    <div class="eyebrow">07 · Kết quả chi tiết</div>
    <h2>Doanh thu dự báo từng track</h2>
    <p class="muted">Bảng đầy đủ gồm tất cả {len(tracks):,} track và từng tháng nằm trong
      <code>track_forecast.csv</code>; album tương ứng có trong <code>album_forecast.csv</code>.</p>
  </section>
</main>
<script>
  window.addEventListener("load", function () {{
    document.querySelectorAll(".js-plotly-plot").forEach(function (plot) {{
      Plotly.Plots.resize(plot);
    }});
  }});
</script>
</body>
</html>"""
    filename.write_text(page, encoding="utf-8")


def main():
    tracks, sales = load_sales_data()
    periods = pd.period_range(sales["month"].min(), sales["month"].max(), freq="M")
    if len(periods) <= max(LAG_PERIODS) + BACKTEST_HORIZON:
        raise ValueError("Cần nhiều hơn 24 tháng lịch sử để huấn luyện và kiểm định.")
    full_periods = periods.append(
        pd.period_range(
            periods[-1] + 1, periods[-1] + FORECAST_HORIZON, freq="M"
        )
    )
    history_length = len(periods)
    backtest_start = history_length - BACKTEST_HORIZON
    sold_track_ids = sorted(sales["track_id"].unique().tolist())
    historical_track_matrix = build_monthly_matrix(sales, sold_track_ids, periods)

    print(
        f"Đã tải {len(tracks):,} track, {len(sold_track_ids):,} track có giao dịch, "
        f"{history_length} tháng dữ liệu."
    )
    training_features, training_targets = make_features(
        historical_track_matrix,
        sold_track_ids,
        max(LAG_PERIODS),
        backtest_start,
    )
    actual_total = historical_track_matrix.sum(axis=0)
    validation_model = make_model()
    validation_model.fit(training_features, training_targets)
    validation_track_forecast = forecast_recursively(
        validation_model,
        historical_track_matrix,
        sold_track_ids,
        backtest_start,
        history_length,
    )
    total_features, total_targets = make_features(
        actual_total.reshape(1, -1),
        ["TOTAL"],
        max(LAG_PERIODS),
        backtest_start,
    )
    validation_total_model = make_model()
    validation_total_model.fit(total_features, total_targets)
    validation_total = forecast_recursively(
        validation_total_model,
        actual_total.reshape(1, -1),
        ["TOTAL"],
        backtest_start,
        history_length,
    )[0]
    validation_forecast = reconcile_to_total(
        validation_track_forecast, validation_total
    )

    sold_track_metadata = tracks.set_index("track_id").loc[sold_track_ids].reset_index()
    album_metadata = (
        tracks[["album_id", "album_name", "artist_name"]]
        .drop_duplicates("album_id")
        .sort_values("album_id")
        .reset_index(drop=True)
    )
    actual_album_matrix = roll_up_tracks(
        historical_track_matrix, sold_track_metadata, album_metadata
    )
    validation_track_extended = np.zeros(
        (len(sold_track_ids), history_length), dtype=float
    )
    validation_track_extended[:, :backtest_start] = historical_track_matrix[
        :, :backtest_start
    ]
    validation_track_extended[:, backtest_start:] = validation_forecast
    validation_album_matrix = roll_up_tracks(
        validation_track_extended, sold_track_metadata, album_metadata
    )
    metrics = pd.DataFrame(
        [
            calculate_metrics(
                historical_track_matrix[:, backtest_start:],
                validation_forecast,
                "Track",
            ),
            calculate_metrics(
                actual_album_matrix[:, backtest_start:],
                validation_album_matrix[:, backtest_start:],
                "Album",
            ),
            calculate_metrics(
                actual_total[backtest_start:],
                validation_total,
                "Tổng doanh thu",
            ),
        ]
    )

    print("Đang huấn luyện mô hình cuối cùng trên toàn bộ lịch sử...")
    final_features, final_targets = make_features(
        historical_track_matrix,
        sold_track_ids,
        max(LAG_PERIODS),
        history_length,
    )
    final_model = make_model()
    final_model.fit(final_features, final_targets)
    sold_track_future_raw = forecast_recursively(
        final_model,
        historical_track_matrix,
        sold_track_ids,
        history_length,
        history_length + FORECAST_HORIZON,
    )
    total_final_features, total_final_targets = make_features(
        actual_total.reshape(1, -1),
        ["TOTAL"],
        max(LAG_PERIODS),
        history_length,
    )
    total_final_model = make_model()
    total_final_model.fit(total_final_features, total_final_targets)
    total_future = forecast_recursively(
        total_final_model,
        actual_total.reshape(1, -1),
        ["TOTAL"],
        history_length,
        history_length + FORECAST_HORIZON,
    )[0]
    sold_track_future = reconcile_to_total(sold_track_future_raw, total_future)

    full_track_matrix = np.zeros(
        (len(tracks), history_length + FORECAST_HORIZON), dtype=float
    )
    track_positions = {
        track_id: position for position, track_id in enumerate(tracks["track_id"])
    }
    sold_positions = np.array([track_positions[track_id] for track_id in sold_track_ids])
    full_track_matrix[sold_positions, :history_length] = historical_track_matrix
    full_track_matrix[sold_positions, history_length:] = sold_track_future
    album_matrix = roll_up_tracks(full_track_matrix, tracks, album_metadata)
    total_history = full_track_matrix[:, :history_length].sum(axis=0)
    total_backtest = validation_total

    track_forecasts, album_forecasts = make_forecast_tables(
        tracks, album_metadata, full_track_matrix, album_matrix, full_periods, history_length
    )
    track_summary, album_summary = make_summary_tables(
        tracks, album_metadata, full_track_matrix, album_matrix, full_periods, history_length
    )
    monthly_forecast = pd.DataFrame(
        {
            "Tháng": [str(period) for period in full_periods],
            "Doanh thu thực tế (USD)": np.r_[
                total_history, np.full(FORECAST_HORIZON, np.nan)
            ],
            "Dự báo kiểm định (USD)": np.r_[
                np.full(backtest_start, np.nan),
                total_backtest,
                np.full(FORECAST_HORIZON, np.nan),
            ],
            "Dự báo tương lai (USD)": np.r_[
                np.full(history_length, np.nan), total_future
            ],
        }
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    track_forecasts.to_csv(RESULTS_DIR / "track_forecast.csv", index=False)
    album_forecasts.to_csv(RESULTS_DIR / "album_forecast.csv", index=False)
    monthly_forecast.to_csv(RESULTS_DIR / "monthly_forecast.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "model_evaluation.csv", index=False)
    with pd.ExcelWriter(RESULTS_DIR / "revenue_forecast.xlsx", engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Đánh giá mô hình", index=False)
        monthly_forecast.to_excel(writer, sheet_name="Theo tháng", index=False)
        track_summary.to_excel(writer, sheet_name="Theo track", index=False)
        album_summary.to_excel(writer, sheet_name="Theo album", index=False)
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                width = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                    width + 2, 42
                )

    export_html(
        RESULTS_DIR / "index.html",
        tracks,
        len(sold_track_ids),
        album_metadata,
        full_periods,
        total_history,
        total_backtest,
        total_future,
        metrics,
        track_summary,
        album_summary,
    )

    print(f"Đã xuất dashboard và dữ liệu tại: {RESULTS_DIR}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))


if __name__ == "__main__":
    main()
