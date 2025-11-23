import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

from prophet import Prophet
from causalimpact import CausalImpact
import plotly.graph_objs as go
from dateutil.easter import easter as easter_sunday


# ==========================================================
# Utility: robust date parsing + resampling to daily
# ==========================================================

def resample_to_daily(df: pd.DataFrame,
                      date_col: str,
                      freq_choice: str,
                      agg_method: str) -> pd.DataFrame:
    """
    - Robustly parse the date column (day-first, tolerate garbage).
    - Drop rows with unparseable dates.
    - Aggregate duplicates.
    - Resample to daily with forward fill.
    """

    if date_col not in df.columns:
        st.error(f"❌ Date column '{date_col}' not found in data.")
        st.stop()

    tmp = df.copy()

    # Convert to string and strip junk
    col = (
        tmp[date_col]
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "NaT": np.nan})
    )

    # Parse dates with dayfirst=True, errors coerced to NaT (no crash)
    tmp[date_col] = pd.to_datetime(col, errors="coerce", dayfirst=True)

    # Drop unparseable dates
    bad_dates = tmp[date_col].isna().sum()
    if bad_dates > 0:
        st.warning(f"⚠️ {bad_dates} rows had invalid dates and were removed.")
        tmp = tmp.dropna(subset=[date_col])

    if tmp.empty:
        st.error("❌ No valid dates left after cleaning. Please check your date column.")
        st.stop()

    # Sort
    tmp = tmp.sort_values(date_col)

    # Aggregate duplicates if needed
    if tmp[date_col].duplicated().sum() > 0:
        if agg_method == "mean":
            tmp = tmp.groupby(date_col, as_index=False).mean(numeric_only=True)
        elif agg_method == "sum":
            tmp = tmp.groupby(date_col, as_index=False).sum(numeric_only=True)
        else:
            tmp = tmp.groupby(date_col, as_index=False).first()

    # Set index
    tmp = tmp.set_index(date_col).sort_index()

    # Resample to daily regardless of original frequency
    # (Daily, Weekly, Monthly all become daily)
    tmp = tmp.resample("D").ffill()
    tmp = tmp.ffill().bfill()

    return tmp


# ==========================================================
# Holidays helper
# ==========================================================

def build_holidays(index: pd.DatetimeIndex, flags: dict) -> pd.DataFrame | None:
    holidays = []
    years = sorted(set(index.year))

    for y in years:
        if flags.get("Christmas", False):
            holidays.append({"ds": pd.Timestamp(f"{y}-12-25"), "holiday": "Christmas"})

        if flags.get("Black Friday", False):
            nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
            fridays = [d for d in nov if d.weekday() == 4]
            if len(fridays) >= 4:
                holidays.append({"ds": fridays[3], "holiday": "BlackFriday"})

        if flags.get("Easter", False):
            es = pd.Timestamp(easter_sunday(y))
            holidays.append({"ds": es, "holiday": "Easter"})

        if flags.get("Cyber Monday", False):
            nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
            fridays = [d for d in nov if d.weekday() == 4]
            if len(fridays) >= 4:
                cm = fridays[3] + pd.Timedelta(days=3)
                holidays.append({"ds": cm, "holiday": "CyberMonday"})

    if not holidays:
        return None

    return pd.DataFrame(holidays)


# ==========================================================
# Series cleaner for KPI + controls
# ==========================================================

def clean_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    # Forward & backward fill, then interpolate, then zero-fill as last resort
    s = s.fillna(method="ffill").fillna(method="bfill").interpolate()
    s = s.fillna(0)
    return s


# ==========================================================
# Prophet engine with hard safety firewall
# ==========================================================

def run_prophet(df: pd.DataFrame,
                metric_col: str,
                control_cols: list[str],
                pre_period: tuple[pd.Timestamp, pd.Timestamp],
                post_period: tuple[pd.Timestamp, pd.Timestamp],
                holiday_flags: dict):

    pre_start, pre_end = pre_period
    post_start, post_end = post_period

    df = df.sort_index()

    train = df.loc[pre_start:pre_end].copy()
    full  = df.loc[pre_start:post_end].copy()

    if train.empty:
        st.error("❌ Pre-period contains no data after cleaning. Adjust your dates.")
        st.stop()

    # ----- build Prophet input frames -----
    # reset_index() makes the datetime index the FIRST column – we don't assume its name
    train_p = train.reset_index()
    full_p  = full.reset_index()

    date_col_name = train_p.columns[0]  # first column is the datetime after reset_index

    train_p = train_p.rename(columns={date_col_name: "ds", metric_col: "y"})
    full_p  = full_p.rename(columns={date_col_name: "ds", metric_col: "y"})

    # clean KPI
    train_p["y"] = clean_series(train_p["y"])
    full_p["y"]  = clean_series(full_p["y"])

    # prophet model
    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False
    )

    # holidays
    hol_df = build_holidays(df.index, holiday_flags)
    if hol_df is not None:
        m.holidays = hol_df

    used_controls: list[str] = []

    # add regressors safely
    for c in control_cols:
        if c not in df.columns:
            continue

        tr_c = clean_series(train[c])
        fu_c = clean_series(full[c])

        # if the regressor is completely flat, it's not useful
        if tr_c.nunique() <= 1:
            st.warning(f"⚠️ Control '{c}' has no variation and was ignored.")
            continue

        train_p[c] = tr_c.values
        full_p[c]  = fu_c.values
        m.add_regressor(c)
        used_controls.append(c)

    # ------------- final safety firewall before fit() -------------
    # no duplicates on ds, no NaNs, no infs
    num_cols = ["y"] + used_controls

    train_p = (
        train_p
        .drop_duplicates(subset=["ds"])
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=num_cols)
        .sort_values("ds")
    )

    if len(train_p) < 14:
        st.error("❌ Not enough clean pre-period data after processing (need ≥ 14 days).")
        st.stop()

    try:
        m.fit(train_p)  # Prophet will ignore extra columns it doesn't need
    except Exception as e:
        st.error(f"❌ Prophet failed during model fitting: {e}")
        st.stop()

    # full frame for prediction
    full_p = (
        full_p
        .drop_duplicates(subset=["ds"])
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .sort_values("ds")
    )

    forecast = m.predict(full_p).set_index("ds")

    # merge with actuals
    result = full[[metric_col]].join(
        forecast[["yhat", "yhat_lower", "yhat_upper"]],
        how="left"
    )

    # pre-period metrics
    pre_res = result.loc[pre_start:pre_end]
    rmse = float(np.sqrt(np.mean((pre_res[metric_col] - pre_res["yhat"]) ** 2)))
    mape = float(
        np.mean(np.abs((pre_res[metric_col] - pre_res["yhat"]) / pre_res[metric_col])) * 100
    )

    # post-period uplift
    post_res = result.loc[post_start:post_end].copy()
    post_res["impact"] = post_res[metric_col] - post_res["yhat"]
    post_res["cum_impact"] = post_res["impact"].cumsum()

    total_uplift = float(post_res["impact"].sum())
    pred_sum = float(post_res["yhat"].sum())
    rel_uplift_pct = float(total_uplift / pred_sum * 100) if pred_sum != 0 else np.nan

    metrics = {
        "Model": "Prophet",
        "RMSE (pre)": rmse,
        "MAPE (pre, %)": mape,
        "Total uplift (post)": total_uplift,
        "Relative uplift (post, %)": rel_uplift_pct,
        "Controls used": used_controls,
    }

    return result, post_res, metrics



# ==========================================================
# BSTS / CausalImpact engine
# ==========================================================

def run_bsts(df: pd.DataFrame,
             metric_col: str,
             control_cols: list[str],
             pre_period: tuple[pd.Timestamp, pd.Timestamp],
             post_period: tuple[pd.Timestamp, pd.Timestamp]):

    pre_start, pre_end = pre_period
    post_start, post_end = post_period

    df = df.sort_index()

    cols = [metric_col] + list(control_cols)
    ci_df = df[cols].copy()

    # clean everything
    for c in cols:
        ci_df[c] = clean_series(ci_df[c])

    try:
        ci = CausalImpact(ci_df, [pre_start, pre_end], [post_start, post_end])
    except Exception as e:
        st.error(f"❌ CausalImpact failed to run on this data: {e}")
        st.stop()

    # some versions of causalimpact don't expose .summary_data
    try:
        sd = ci.summary_data
    except AttributeError:
        st.error("❌ This CausalImpact version does not provide 'summary_data'. Try the Prophet model instead.")
        st.stop()

    rmse = float(np.sqrt(np.mean((sd["actual"] - sd["predicted"]) ** 2)))
    mape = float(np.mean(np.abs((sd["actual"] - sd["predicted"]) / sd["actual"])) * 100)

    post_mask = (sd.index >= post_start) & (sd.index <= post_end)
    post_sd = sd.loc[post_mask]

    uplift = float(post_sd["point_effect"].sum())
    pred_sum = float(post_sd["predicted"].sum())
    rel_uplift_pct = float(uplift / pred_sum * 100) if pred_sum != 0 else np.nan

    p_value = None
    if "p_value" in sd.columns:
        p_value = float(sd["p_value"].iloc[-1])

    metrics = {
        "Model": "BSTS / CausalImpact",
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Total uplift (post)": uplift,
        "Relative uplift (post, %)": rel_uplift_pct,
        "p-value": p_value,
    }

    return ci, sd, metrics

# ==========================================================
# Plot helpers
# ==========================================================

def plot_actual_vs_cf(df: pd.DataFrame, metric_col: str, title: str = "Actual vs Counterfactual"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[metric_col], name="Actual"))
    fig.add_trace(go.Scatter(x=df.index, y=df["yhat"], name="Counterfactual"))
    if "yhat_lower" in df.columns and "yhat_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df.index, df.index[::-1]]),
            y=pd.concat([df["yhat_upper"], df["yhat_lower"][::-1]]),
            fill="toself",
            mode="lines",
            line=dict(width=0),
            opacity=0.2,
            showlegend=False,
            name="Confidence interval"
        ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=metric_col)
    return fig


# ==========================================================
# Streamlit app
# ==========================================================

def main():
    st.set_page_config(layout="wide", page_title="Counterfactual Impact App")
    st.title("📈 Counterfactual Impact App (Prophet + BSTS)")
    st.caption("Fully error-proof. Supports CSV & Excel · Daily / Weekly / Monthly.")

    # -------------------------------
    # File upload (CSV or Excel)
    # -------------------------------
    uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if not uploaded:
        st.info("👈 Upload a CSV or Excel file to begin.")
        st.stop()

    file_ext = Path(uploaded.name).suffix.lower()

    # Excel: let user choose sheet
    if file_ext in [".xlsx", ".xls"]:
        try:
            xl = pd.ExcelFile(uploaded)
        except Exception as e:
            st.error(f"❌ Failed to read Excel file: {e}")
            st.stop()

        sheet_name = st.sidebar.selectbox("Select worksheet", xl.sheet_names)
        df = xl.parse(sheet_name)
    else:
        # CSV
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"❌ Failed to read CSV file: {e}")
            st.stop()

    if df.empty:
        st.error("❌ Uploaded file appears to be empty.")
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Column & frequency selection
    # -------------------------------
    date_col = st.sidebar.selectbox("Date column", df.columns)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("❌ No numeric columns found for KPI. Please check your data.")
        st.stop()

    metric_col = st.sidebar.selectbox("Metric (KPI)", numeric_cols)

    control_candidates = [c for c in df.columns if c != metric_col and c != date_col]
    control_cols = st.sidebar.multiselect("Control variables (optional)", control_candidates)

    freq_choice = st.sidebar.radio("Data frequency (as uploaded)", ["Daily", "Weekly", "Monthly"])
    agg_method = st.sidebar.radio("If duplicate dates exist, aggregate using:", ["mean", "sum", "first"])

    # -------------------------------
    # Resample to daily
    # -------------------------------
    df_daily = resample_to_daily(df, date_col, freq_choice, agg_method)

    min_ts, max_ts = df_daily.index.min(), df_daily.index.max()
    min_date, max_date = min_ts.date(), max_ts.date()

    if (max_ts - min_ts).days < 1:
        st.error("❌ Not enough date span in data. Need at least 2 days.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Periods")

    span_days = (max_ts - min_ts).days
    default_pre_end = min_date + timedelta(days=max(13, span_days // 2))
    if default_pre_end > max_date:
        default_pre_end = max_date

    default_post_start = min(default_pre_end + timedelta(days=1), max_date)

    pre_period_input = st.sidebar.date_input(
        "Pre-period (training)",
        value=(min_date, default_pre_end),
        min_value=min_date,
        max_value=max_date
    )

    post_period_input = st.sidebar.date_input(
        "Post-period (intervention)",
        value=(default_post_start, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if not isinstance(pre_period_input, (list, tuple)) or len(pre_period_input) != 2:
        st.error("❌ Please select a start and end date for the pre-period.")
        st.stop()
    if not isinstance(post_period_input, (list, tuple)) or len(post_period_input) != 2:
        st.error("❌ Please select a start and end date for the post-period.")
        st.stop()

    pre_start = pd.Timestamp(pre_period_input[0])
    pre_end = pd.Timestamp(pre_period_input[1])
    post_start = pd.Timestamp(post_period_input[0])
    post_end = pd.Timestamp(post_period_input[1])

    if pre_start >= pre_end:
        st.error("❌ Pre-period start must be before pre-period end.")
        st.stop()
    if post_start >= post_end:
        st.error("❌ Post-period start must be before post-period end.")
        st.stop()
    if pre_end >= post_start:
        st.error("❌ Pre-period must end *before* post-period begins.")
        st.stop()

    model_choice = st.sidebar.radio("Model", ["Prophet", "BSTS / CausalImpact"])

    st.sidebar.markdown("### Holidays / High Demand")
    holiday_flags = {
        "Christmas": st.sidebar.checkbox("Christmas", True),
        "Black Friday": st.sidebar.checkbox("Black Friday", True),
        "Easter": st.sidebar.checkbox("Easter", True),
        "Cyber Monday": st.sidebar.checkbox("Cyber Monday", True),
    }

    run_btn = st.sidebar.button("🚀 Run model")

    if not run_btn:
        st.stop()

    # -------------------------------
    # Run selected model
    # -------------------------------
    if model_choice == "Prophet":
        st.subheader("Prophet Counterfactual Results")
        res_df, post_df, metrics = run_prophet(
            df_daily,
            metric_col=metric_col,
            control_cols=control_cols,
            pre_period=(pre_start, pre_end),
            post_period=(post_start, post_end),
            holiday_flags=holiday_flags
        )

        st.json(metrics)
        st.plotly_chart(plot_actual_vs_cf(res_df, metric_col), use_container_width=True)

    else:
        st.subheader("BSTS / CausalImpact Results")
        ci, sd, metrics = run_bsts(
            df_daily,
            metric_col=metric_col,
            control_cols=control_cols,
            pre_period=(pre_start, pre_end),
            post_period=(post_start, post_end)
        )

        st.json(metrics)

        # Plot using summary_data (rename columns)
        plot_df = sd.rename(columns={"actual": metric_col, "predicted": "yhat"})
        st.plotly_chart(plot_actual_vs_cf(plot_df, metric_col), use_container_width=True)

        st.markdown("**CausalImpact summary**")
        st.text(ci.summary())


if __name__ == "__main__":
    main()
