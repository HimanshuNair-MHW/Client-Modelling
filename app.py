import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path

from prophet import Prophet
from causalimpact import CausalImpact
import plotly.graph_objs as go
from dateutil.easter import easter as easter_sunday


# ==========================================================
# Helpers: date parsing + cleaning
# ==========================================================

def standardize_to_daily(df: pd.DataFrame,
                         date_col: str,
                         agg_method: str) -> pd.DataFrame:
    """
    Robustly parse date column, drop bad dates, aggregate duplicates,
    and standardise to a daily DateTimeIndex with forward/back fill.
    """
    if date_col not in df.columns:
        st.error(f"❌ Date column '{date_col}' not found.")
        st.stop()

    tmp = df.copy()

    # Stringify and clean basic junk
    col = (
        tmp[date_col]
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "NaT": np.nan})
    )

    # Parse dates (dayfirst to be friendly to UK/Europe)
    tmp[date_col] = pd.to_datetime(col, errors="coerce", dayfirst=True)

    bad = tmp[date_col].isna().sum()
    if bad > 0:
        st.warning(f"⚠ {bad} rows had invalid dates and were dropped.")
        tmp = tmp.dropna(subset=[date_col])

    if tmp.empty:
        st.error("❌ No valid dates left after cleaning.")
        st.stop()

    tmp = tmp.sort_values(date_col)

    # Aggregate duplicates
    if tmp[date_col].duplicated().sum() > 0:
        if agg_method == "mean":
            tmp = tmp.groupby(date_col, as_index=False).mean(numeric_only=True)
        elif agg_method == "sum":
            tmp = tmp.groupby(date_col, as_index=False).sum(numeric_only=True)
        else:
            tmp = tmp.groupby(date_col, as_index=False).first()

    tmp = tmp.set_index(date_col).sort_index()

    # Build a complete daily index
    full_idx = pd.date_range(tmp.index.min(), tmp.index.max(), freq="D")
    tmp = tmp.reindex(full_idx)

    # Fill gaps
    tmp = tmp.ffill().bfill()

    return tmp


def clean_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.ffill().bfill().interpolate()
    s = s.fillna(0)
    return s


def build_holidays(index: pd.DatetimeIndex, flags: dict) -> pd.DataFrame | None:
    holidays = []
    years = sorted(set(index.year))

    for y in years:
        if flags.get("Christmas", False):
            holidays.append({"ds": pd.Timestamp(f"{y}-12-25"), "holiday": "Christmas"})
        if flags.get("Black Friday", False):
            nov = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="D")
            fridays = [d for d in nov if d.weekday() == 4]
            if len(fridays) >= 4:
                holidays.append({"ds": fridays[3], "holiday": "BlackFriday"})
        if flags.get("Easter", False):
            holidays.append({"ds": pd.Timestamp(easter_sunday(y)), "holiday": "Easter"})
        if flags.get("Cyber Monday", False):
            nov = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="D")
            fridays = [d for d in nov if d.weekday() == 4]
            if len(fridays) >= 4:
                cm = fridays[3] + pd.Timedelta(days=3)
                holidays.append({"ds": cm, "holiday": "CyberMonday"})

    return pd.DataFrame(holidays) if holidays else None


# ==========================================================
# PROPHET: pure forecast (version 2) with fit metrics
# ==========================================================

def run_prophet_forecast(df_daily: pd.DataFrame,
                         metric_col: str,
                         horizon_len: int,
                         horizon_unit: str,
                         holiday_flags: dict):
    """
    df_daily: daily-indexed dataframe with metric_col
    horizon_unit: 'Daily', 'Weekly', 'Monthly'
    Returns: (history_series, full_forecast_df, metrics_dict)
    """

    df_daily = df_daily.sort_index()
    base = df_daily[[metric_col]].copy()

    # Resample to modeling frequency
    if horizon_unit == "Daily":
        freq = "D"
        series = base
    elif horizon_unit == "Weekly":
        freq = "W"
        series = base.resample("W").mean()
    else:  # Monthly
        freq = "M"
        series = base.resample("M").mean()

    series = series.dropna()

    if len(series) < 10:
        st.error("❌ Not enough data to fit Prophet (need at least 10 points).")
        st.stop()

    # Prepare Prophet frame
    df_p = series.reset_index()
    date_col_name = df_p.columns[0]
    df_p = df_p.rename(columns={date_col_name: "ds", metric_col: "y"})
    df_p["y"] = clean_series(df_p["y"])

    # Build model
    m = Prophet(
        weekly_seasonality=True if horizon_unit in ["Daily", "Weekly"] else False,
        yearly_seasonality=True,
        daily_seasonality=False,
    )

    hol_df = build_holidays(series.index, holiday_flags)
    if hol_df is not None:
        m.holidays = hol_df

    try:
        m.fit(df_p)
    except Exception as e:
        st.error(f"❌ Prophet failed during fitting: {e}")
        st.stop()

    future = m.make_future_dataframe(periods=horizon_len, freq=freq)
    forecast = m.predict(future).set_index("ds")

    # Metrics on in-sample fit
    aligned = forecast.loc[series.index]
    actual = series[metric_col]
    pred = aligned["yhat"]

    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mape = float(
        np.mean(
            np.abs((actual - pred) / np.where(actual != 0, actual, np.nan))
        ) * 100
    )

    metrics = {
        "Model": "Prophet Forecast",
        "Frequency": horizon_unit,
        "Horizon length": horizon_len,
        "RMSE (in-sample)": rmse,
        "MAPE (in-sample, %)": mape,
    }

    return series, forecast, metrics


def plot_prophet_forecast(history: pd.DataFrame,
                          forecast: pd.DataFrame,
                          metric_col: str,
                          title: str):
    fig = go.Figure()

    # Actuals (history only)
    fig.add_trace(go.Scatter(
        x=history.index,
        y=history[metric_col],
        mode="lines",
        name="Actual"
    ))

    # Forecast mean (includes history & future)
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast["yhat"],
        mode="lines",
        name="Forecast"
    ))

    # Uncertainty band (simple + safe)
    if {"yhat_lower", "yhat_upper"}.issubset(forecast.columns):
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast["yhat_lower"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            opacity=0.2,
            name="Forecast interval"
        ))

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=metric_col)
    return fig


# ==========================================================
# BSTS / CausalImpact: uplift model (requires controls)
# ==========================================================

def _standardise_ci_summary(sd_raw, index_for_df):
    """
    Convert various possible CausalImpact summary_data / inferences
    objects into a DataFrame with at least:
      - 'actual'
      - 'predicted'
    and ideally:
      - 'point_effect'
    """
    # Case 1: already a DataFrame
    if isinstance(sd_raw, pd.DataFrame):
        df_sd = sd_raw.copy()
    else:
        # Try Series or array-like
        arr = np.asarray(sd_raw)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = [f"col{i}" for i in range(arr.shape[1])]
        df_sd = pd.DataFrame(arr, index=index_for_df, columns=cols)

    # Try to detect columns
    def find_col(candidates):
        for c in df_sd.columns:
            lc = c.lower()
            if any(term in lc for term in candidates):
                return c
        return None

    actual_col = find_col(["actual", "response", "observed", "y"])
    pred_col = find_col(["pred", "forecast", "expected", "mean"])
    effect_col = find_col(["effect"])

    if actual_col is None or pred_col is None:
        # Fallback: first two columns
        if len(df_sd.columns) >= 2:
            actual_col = df_sd.columns[0]
            pred_col = df_sd.columns[1]
        else:
            st.error("❌ Unexpected CausalImpact output: cannot identify actual & predicted.")
            st.stop()

    df_std = pd.DataFrame({
        "actual": df_sd[actual_col],
        "predicted": df_sd[pred_col]
    })

    # Add effect column if we can
    if effect_col is not None:
        df_std["point_effect"] = df_sd[effect_col]
    elif len(df_sd.columns) >= 3:
        df_std["point_effect"] = df_sd.iloc[:, 2]

    return df_std


def run_bsts_uplift(df_daily: pd.DataFrame,
                    metric_col: str,
                    control_cols: list[str],
                    pre_period: tuple[pd.Timestamp, pd.Timestamp],
                    post_period: tuple[pd.Timestamp, pd.Timestamp]):

    pre_start, pre_end = pre_period
    post_start, post_end = post_period

    df_daily = df_daily.sort_index()

    # Need at least one control for BSTS
    if len(control_cols) == 0:
        st.error("❌ BSTS uplift requires at least one control variable.")
        st.stop()

    cols = [metric_col] + control_cols
    ci_df = df_daily[cols].copy()

    for c in cols:
        ci_df[c] = clean_series(ci_df[c])

    try:
        ci = CausalImpact(ci_df, (pre_start, pre_end), (post_start, post_end))
    except Exception as e:
        st.error(f"❌ CausalImpact failed to run: {e}")
        st.stop()

    # Try different attributes for CI output
    sd_raw = getattr(ci, "summary_data", None)
    if sd_raw is None:
        sd_raw = getattr(ci, "inferences", None)
    if sd_raw is None:
        st.error("❌ Could not read CausalImpact output (no summary_data / inferences).")
        st.stop()

    sd = _standardise_ci_summary(sd_raw, ci_df.index)

    # Metrics
    actual = sd["actual"]
    pred = sd["predicted"]

    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(
            np.nanmean(
                np.abs((actual - pred) / np.where(actual != 0, actual, np.nan))
            ) * 100
        )

    # Post-period uplift
    post_mask = (sd.index >= post_start) & (sd.index <= post_end)
    sd_post = sd.loc[post_mask]

    uplift = float((sd_post["actual"] - sd_post["predicted"]).sum())
    pred_sum = float(sd_post["predicted"].sum())
    rel_uplift_pct = uplift / pred_sum * 100 if pred_sum != 0 else np.nan

    # p-value (if present)
    p_val = None
    for cand in ["p_value", "p", "tail_prob"]:
        if cand in sd.columns:
            p_val = float(sd[cand].iloc[-1])
            break

    metrics = {
        "Model": "BSTS / CausalImpact (uplift)",
        "RMSE (overall)": rmse,
        "MAPE (overall, %)": mape,
        "Total uplift (post)": uplift,
        "Relative uplift (post, %)": rel_uplift_pct,
        "p-value (if available)": p_val,
        "Controls used": control_cols,
    }

    # For plotting: actual vs predicted
    plot_df = pd.DataFrame({
        "Actual": actual,
        "Predicted": pred
    })

    if "point_effect" in sd.columns:
        plot_df["Effect"] = sd["point_effect"]

    return ci, plot_df, metrics


def plot_bsts_actual_vs_cf(plot_df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["Actual"],
        mode="lines",
        name="Actual"
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["Predicted"],
        mode="lines",
        name="Counterfactual"
    ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Metric")
    return fig


# ==========================================================
# STREAMLIT APP
# ==========================================================

def main():
    st.set_page_config(layout="wide", page_title="Forecast & Uplift App")
    st.title("📈 Forecast & Uplift App")
    st.caption("Prophet for pure forecast · BSTS (CausalImpact) for uplift with controls.")

    # ---------- File upload ----------
    uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if not uploaded:
        st.info("👈 Upload a CSV or Excel file to begin.")
        st.stop()

    ext = Path(uploaded.name).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        try:
            xl = pd.ExcelFile(uploaded)
        except Exception as e:
            st.error(f"❌ Failed to read Excel file: {e}")
            st.stop()
        sheet = st.sidebar.selectbox("Worksheet", xl.sheet_names)
        df = xl.parse(sheet)
    else:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"❌ Failed to read CSV file: {e}")
            st.stop()

    if df.empty:
        st.error("❌ Uploaded file is empty.")
        st.stop()

    st.subheader("Data preview")
    st.dataframe(df.head())

    # ---------- Column selection ----------
    date_col = st.sidebar.selectbox("Date column", df.columns)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("❌ No numeric columns found for KPI.")
        st.stop()

    metric_col = st.sidebar.selectbox("Metric (KPI)", numeric_cols)
    control_candidates = [c for c in df.columns if c not in [metric_col, date_col]]
    control_cols = st.sidebar.multiselect("Control variables (for BSTS uplift)", control_candidates)

    agg_method = st.sidebar.radio("If duplicate dates exist, aggregate using:", ["mean", "sum", "first"])

    # Standardize to daily index
    df_daily = standardize_to_daily(df, date_col, agg_method)

    min_ts, max_ts = df_daily.index.min(), df_daily.index.max()
    min_d, max_d = min_ts.date(), max_ts.date()

    # ---------- Model choice ----------
    model_choice = st.sidebar.radio("Model", ["Prophet forecast", "BSTS uplift"])

    # ---------- Prophet forecast UI ----------
    if model_choice == "Prophet forecast":
        st.sidebar.markdown("### Prophet Forecast Settings")
        granularity = st.sidebar.radio("Forecast granularity", ["Daily", "Weekly", "Monthly"])

        if granularity == "Daily":
            max_len = 365
            default_len = 30
        elif granularity == "Weekly":
            max_len = 104
            default_len = 12
        else:
            max_len = 36
            default_len = 6

        horizon_len = st.sidebar.slider(
            "Forecast length",
            min_value=1,
            max_value=max_len,
            value=default_len
        )

        st.sidebar.markdown("### Holidays")
        holiday_flags = {
            "Christmas": st.sidebar.checkbox("Christmas", True),
            "Black Friday": st.sidebar.checkbox("Black Friday", True),
            "Easter": st.sidebar.checkbox("Easter", True),
            "Cyber Monday": st.sidebar.checkbox("Cyber Monday", True),
        }

        if st.sidebar.button("🚀 Run Prophet forecast"):
            history, forecast, metrics = run_prophet_forecast(
                df_daily,
                metric_col=metric_col,
                horizon_len=horizon_len,
                horizon_unit=granularity,
                holiday_flags=holiday_flags
            )

            st.subheader("Prophet Forecast Results")
            st.json(metrics)
            fig = plot_prophet_forecast(history, forecast, metric_col, "Prophet forecast")
            st.plotly_chart(fig, use_container_width=True)

        st.stop()

    # ---------- BSTS uplift UI ----------
    st.sidebar.markdown("### BSTS Uplift Settings")

    span_days = (max_ts - min_ts).days
    default_pre_end = min_d + timedelta(days=int(span_days * 0.7))
    if default_pre_end >= max_d:
        default_pre_end = max_d - timedelta(days=1)
    default_post_start = default_pre_end + timedelta(days=1)

    pre_input = st.sidebar.date_input(
        "Pre-period (training)",
        value=(min_d, default_pre_end),
        min_value=min_d,
        max_value=max_d
    )
    post_input = st.sidebar.date_input(
        "Post-period (intervention)",
        value=(default_post_start, max_d),
        min_value=min_d,
        max_value=max_d
    )

    if not isinstance(pre_input, (list, tuple)) or len(pre_input) != 2:
        st.error("❌ Select both start and end for pre-period.")
        st.stop()
    if not isinstance(post_input, (list, tuple)) or len(post_input) != 2:
        st.error("❌ Select both start and end for post-period.")
        st.stop()

    pre_start, pre_end = map(pd.Timestamp, pre_input)
    post_start, post_end = map(pd.Timestamp, post_input)

    if pre_start >= pre_end:
        st.error("❌ Pre-period start must be before pre-period end.")
        st.stop()
    if post_start >= post_end:
        st.error("❌ Post-period start must be before post-period end.")
        st.stop()
    if pre_end >= post_start:
        st.error("❌ Pre-period must end before post-period starts.")
        st.stop()

    if st.sidebar.button("🚀 Run BSTS uplift"):
        ci, plot_df, metrics = run_bsts_uplift(
            df_daily,
            metric_col=metric_col,
            control_cols=control_cols,
            pre_period=(pre_start, pre_end),
            post_period=(post_start, post_end)
        )

        st.subheader("BSTS / CausalImpact Uplift Results")
        st.json(metrics)
        fig = plot_bsts_actual_vs_cf(plot_df, "Actual vs counterfactual (BSTS uplift)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**CausalImpact textual summary (if available)**")
        try:
            st.text(ci.summary())
        except Exception:
            st.text("Summary not available on this CausalImpact build.")


if __name__ == "__main__":
    main()
