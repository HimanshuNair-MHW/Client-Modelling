import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from io import StringIO

from prophet import Prophet
from causalimpact import CausalImpact
import plotly.graph_objs as go

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dateutil.easter import easter as easter_sunday


# ==========================================================
# SAFE DATA PREP
# ==========================================================

def prepare_df(df: pd.DataFrame, date_col: str, agg_method: str):
    df = df.copy()

    # Ensure date is datetime
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except:
        st.error("❌ Your date column contains non-date values. Please clean the dataset.")
        st.stop()

    # Sort
    df = df.sort_values(date_col)

    # Handle duplicates safely
    if df[date_col].duplicated().sum() > 0:
        st.warning(f"⚠️ Duplicate dates detected. Aggregating using: {agg_method.upper()}")
        
        if agg_method == "mean":
            df = df.groupby(date_col, as_index=False).mean()
        elif agg_method == "sum":
            df = df.groupby(date_col, as_index=False).sum()
        elif agg_method == "first":
            df = df.groupby(date_col, as_index=False).first()

    # Set index
    df = df.set_index(date_col)

    # Ensure daily frequency
    try:
        df = df.asfreq("D")
    except:
        st.error("❌ Failed to resample to daily frequency. Check your date column formatting.")
        st.stop()

    # Forward fill missing days
    df = df.ffill()

    return df


# ==========================================================
# HOLIDAY HANDLING
# ==========================================================

def build_holiday_df(index: pd.DatetimeIndex, flags):
    holidays = []
    years = sorted(set(index.year))

    for y in years:

        if flags["christmas"]:
            dt = pd.Timestamp(f"{y}-12-25")
            holidays.append({"ds": dt, "holiday": "christmas"})

        if flags["black_friday"]:
            nov = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="D")
            fridays = [d for d in nov if d.weekday() == 4]
            if len(fridays) >= 4:
                holidays.append({"ds": fridays[3], "holiday": "black_friday"})

        if flags["easter"]:
            es = pd.Timestamp(easter_sunday(y))
            holidays.append({"ds": es, "holiday": "easter"})

        if flags["cyber_monday"] and flags["black_friday"]:
            nov = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="D")
            fridays = [d for d in nov if d.weekday() == 4]
            if len(fridays) >= 4:
                cm = fridays[3] + pd.Timedelta(days=3)
                holidays.append({"ds": cm, "holiday": "cyber_monday"})

    if len(holidays) == 0:
        return None

    return pd.DataFrame(holidays)


# ==========================================================
# PROPHET COUNTERFACTUAL
# ==========================================================

def run_prophet(df, metric_col, control_cols, pre_period, post_period, holidays):

    pre_start, pre_end = pre_period
    post_start, post_end = post_period

    train = df.loc[pre_start:pre_end]
    full = df.loc[pre_start:post_end]

    # Prophet needs at least 14 days
    if len(train) < 14:
        st.error("❌ Prophet requires at least 14 days of pre-period data.")
        st.stop()

    # Prepare input
    train_p = train.reset_index().rename(columns={"index": "ds", metric_col: "y"})
    full_p = full.reset_index().rename(columns={"index": "ds"})

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False
    )

    hol = build_holiday_df(df.index, holidays)
    if hol is not None:
        model.holidays = hol

    for c in control_cols:
        model.add_regressor(c)
        train_p[c] = train[c].values
        full_p[c] = full[c].values

    # Fit
    model.fit(train_p)

    # Forecast
    forecast = model.predict(full_p)
    forecast = forecast.set_index("ds")

    # Merge
    result = full[[metric_col]].join(
        forecast[["yhat", "yhat_lower", "yhat_upper"]],
        how="left"
    )

    # Fit metrics
    pre_res = result.loc[pre_start:pre_end]
    rmse = np.sqrt(np.mean((pre_res[metric_col] - pre_res["yhat"]) ** 2))
    mape = np.mean(np.abs((pre_res[metric_col] - pre_res["yhat"]) / pre_res[metric_col])) * 100

    # Impact
    post_res = result.loc[post_start:post_end].copy()
    post_res["impact"] = post_res[metric_col] - post_res["yhat"]
    post_res["cum_impact"] = post_res["impact"].cumsum()

    metrics = {
        "rmse": rmse,
        "mape": mape,
        "uplift": post_res["impact"].sum(),
        "rel_uplift": (post_res["impact"].sum() / post_res["yhat"].sum() * 100)
    }

    return result, post_res, metrics


# ==========================================================
# BSTS / CAUSAL IMPACT
# ==========================================================

def run_bsts(df, metric_col, control_cols, pre_period, post_period):
    pre_start, pre_end = pre_period
    post_start, post_end = post_period

    cols = [metric_col] + list(control_cols)
    ci_df = df[cols]

    ci = CausalImpact(ci_df, [pre_start, pre_end], [post_start, post_end])
    sd = ci.summary_data

    # Metrics
    rmse = np.sqrt(np.mean((sd["actual"] - sd["predicted"]) ** 2))
    mape = np.mean(np.abs((sd["actual"] - sd["predicted"]) / sd["actual"])) * 100
    uplift = sd.loc[post_start:post_end]["point_effect"].sum()
    rel_uplift = (uplift / sd.loc[post_start:post_end]["predicted"].sum() * 100)

    metrics = {
        "rmse": rmse,
        "mape": mape,
        "uplift": uplift,
        "rel_uplift": rel_uplift,
        "p_value": sd["p_value"].dropna().iloc[-1] if "p_value" in sd else None
    }

    return ci, sd, metrics


# ==========================================================
# CHART HELPERS
# ==========================================================

def plot_actual_vs_cf(df, metric_col):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[metric_col], name="Actual"))
    fig.add_trace(go.Scatter(x=df.index, y=df["yhat"], name="Counterfactual"))
    fig.update_layout(title="Actual vs Counterfactual")
    return fig


# ==========================================================
# STREAMLIT UI
# ==========================================================

def main():
    st.set_page_config(layout="wide", page_title="Counterfactual Model App")

    st.title("📈 Counterfactual Impact App (Prophet + BSTS)")
    st.caption("Stupid-proof, agency-ready version.")

    # -------------------------------------
    # Data Input
    # -------------------------------------
    src = st.sidebar.radio("Data source", ["Upload CSV"])

    df = None
    if src == "Upload CSV":
        file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)

    if df is None:
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head())

    date_col = st.sidebar.selectbox("Date column", df.columns)
    metric_col = st.sidebar.selectbox("Metric (KPI)", df.select_dtypes(include=[np.number]).columns)
    controls = st.sidebar.multiselect("Control variables", [c for c in df.columns if c != metric_col])

    agg_method = st.sidebar.radio("How to aggregate duplicate dates?", ["mean", "sum", "first"])

    df_clean = prepare_df(df, date_col, agg_method)

    min_date, max_date = df_clean.index.min(), df_clean.index.max()

    pre = st.sidebar.date_input("Pre-period", (min_date, max_date - pd.Timedelta(days=30)))
    post = st.sidebar.date_input("Post-period", (max_date - pd.Timedelta(days=29), max_date))

    # VALIDATION
    pre_start, pre_end = map(pd.Timestamp, pre)
    post_start, post_end = map(pd.Timestamp, post)

    if pre_start >= pre_end:
        st.error("❌ Pre-period start must be before end.")
        st.stop()

    if post_start >= post_end:
        st.error("❌ Post-period start must be before end.")
        st.stop()

    if pre_end >= post_start:
        st.error("❌ Pre-period must end *before* post-period begins.")
        st.stop()

    model = st.sidebar.radio("Model", ["Prophet", "BSTS"])

    holidays = {
        "christmas": st.sidebar.checkbox("Christmas", True),
        "black_friday": st.sidebar.checkbox("Black Friday", True),
        "easter": st.sidebar.checkbox("Easter", True),
        "cyber_monday": st.sidebar.checkbox("Cyber Monday", True)
    }

    if st.sidebar.button("🚀 Run model"):
        st.subheader("Results")

        if model == "Prophet":
            result, post_df, m = run_prophet(df_clean, metric_col, controls,
                                             (pre_start, pre_end), (post_start, post_end), holidays)

            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{m['rmse']:.2f}")
            col2.metric("MAPE", f"{m['mape']:.2f}%")
            col3.metric("Uplift", f"{m['uplift']:.2f}")

            st.plotly_chart(plot_actual_vs_cf(result, metric_col), use_container_width=True)

        else:
            ci, sd, m = run_bsts(df_clean, metric_col, controls, 
                                 (pre_start, pre_end), (post_start, post_end))

            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{m['rmse']:.2f}")
            col2.metric("MAPE", f"{m['mape']:.2f}%")
            col3.metric("Uplift", f"{m['uplift']:.2f}")

            st.write(ci.summary())

            st.plotly_chart(plot_actual_vs_cf(
                sd.rename(columns={"actual": metric_col,
                                   "predicted": "yhat"})),
                use_container_width=True
            )


if __name__ == "__main__":
    main()
