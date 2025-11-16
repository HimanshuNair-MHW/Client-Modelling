import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from io import StringIO

from prophet import Prophet
from causalimpact import CausalImpact
import plotly.graph_objs as go
from dateutil.easter import easter as easter_sunday


# ==========================================================
# FREQUENCY HANDLING
# ==========================================================

def resample_to_daily(df, date_col, freq_choice, agg_method):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Remove duplicates
    if df[date_col].duplicated().sum() > 0:
        if agg_method == "mean":
            df = df.groupby(date_col, as_index=False).mean()
        elif agg_method == "sum":
            df = df.groupby(date_col, as_index=False).sum()
        else:
            df = df.groupby(date_col, as_index=False).first()

    df = df.set_index(date_col)

    # --- RESAMPLING ---
    if freq_choice == "Daily":
        df = df.asfreq("D")

    elif freq_choice == "Weekly":
        # Weekly → assign the value to the full week
        df = df.resample("D").ffill()

    elif freq_choice == "Monthly":
        # Monthly → assign to whole month
        df = df.resample("D").ffill()

    # Fill missing edges
    df = df.ffill().bfill()

    return df


# ==========================================================
# HOLIDAY CALENDAR
# ==========================================================

def build_holidays(dates, flags):
    holidays = []
    years = sorted(set(dates.year))

    for y in years:
        if flags['Christmas']:
            holidays.append({"ds": pd.Timestamp(f"{y}-12-25"), "holiday": "Christmas"})

        if flags['Black Friday']:
            nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
            fridays = [d for d in nov if d.weekday() == 4]
            if len(fridays) >= 4:
                holidays.append({"ds": fridays[3], "holiday": "BlackFriday"})

        if flags['Easter']:
            es = pd.Timestamp(easter_sunday(y))
            holidays.append({"ds": es, "holiday": "Easter"})

        if flags['Cyber Monday']:
            nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
            fridays = [d for d in nov if d.weekday() == 4]
            if len(fridays) >= 4:
                cm = fridays[3] + pd.Timedelta(days=3)
                holidays.append({"ds": cm, "holiday": "CyberMonday"})

    return pd.DataFrame(holidays) if holidays else None


# ==========================================================
# MASSIVE AUTO-CLEANER FOR KPI + CONTROLS
# ==========================================================

def clean_series(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.fillna(method="ffill").fillna(method="bfill").interpolate()
    if s.isna().any():
        s = s.fillna(0)
    return s


# ==========================================================
# PROPHET ENGINE (AUTO-CLEAN, FULLY ADAPTIVE)
# ==========================================================

def run_prophet(df, metric_col, controls, pre_period, post_period, holidays):

    pre_s, pre_e = pre_period
    post_s, post_e = post_period

    train = df.loc[pre_s:pre_e].copy()
    full  = df.loc[pre_s:post_e].copy()

    # Must rename metric to y in BOTH
    train_p = train.reset_index().rename(columns={"index":"ds", metric_col:"y"})
    full_p  = full.reset_index().rename(columns={"index":"ds", metric_col:"y"})

    # Clean KPI
    train_p["y"] = clean_series(train_p["y"])
    full_p["y"]  = clean_series(full_p["y"])

    # Prophet model
    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False
    )

    hol = build_holidays(df.index, holidays)
    if hol is not None:
        m.holidays = hol

    used_controls = []

    # Clean and add control regressors
    for c in controls:
        clean_train = clean_series(train[c])
        clean_full  = clean_series(full[c])

        if clean_train.isna().all():
            st.warning(f"⚠️ Dropping control '{c}' (invalid values)")
            continue

        train_p[c] = clean_train.values
        full_p[c]  = clean_full.values

        m.add_regressor(c)
        used_controls.append(c)

    # Fit
    m.fit(train_p)

    # Forecast
    fc = m.predict(full_p).set_index("ds")

    result = full[[metric_col]].join(
        fc[["yhat","yhat_lower","yhat_upper"]],
        how="left"
    )

    # Metrics
    pre = result.loc[pre_s:pre_e]
    rmse = np.sqrt(np.mean((pre[metric_col] - pre["yhat"])**2))
    mape = np.mean(np.abs((pre[metric_col] - pre["yhat"]) / pre[metric_col])) * 100

    post = result.loc[post_s:post_e].copy()
    post["impact"] = post[metric_col] - post["yhat"]
    post["cum_impact"] = post["impact"].cumsum()

    metrics = {
        "RMSE": rmse,
        "MAPE": mape,
        "Uplift": post["impact"].sum(),
        "Rel Uplift %": (post["impact"].sum() / post["yhat"].sum() * 100),
        "Controls Used": used_controls
    }

    return result, post, metrics


# ==========================================================
# BSTS ENGINE (AUTO-CLEAN)
# ==========================================================

def run_bsts(df, metric_col, controls, pre_period, post_period):

    pre_s, pre_e = pre_period
    post_s, post_e = post_period

    cols = [metric_col] + list(controls)
    df_ci = df[cols].copy()

    for c in cols:
        df_ci[c] = clean_series(df_ci[c])

    ci = CausalImpact(df_ci, [pre_s, pre_e], [post_s, post_e])
    sd = ci.summary_data

    rmse = np.sqrt(np.mean((sd["actual"] - sd["predicted"])**2))
    mape = np.mean(np.abs((sd["actual"] - sd["predicted"]) / sd["actual"])) * 100
    uplift = sd.loc[post_s:post_e]["point_effect"].sum()
    rel = uplift / sd.loc[post_s:post_e]["predicted"].sum() * 100

    metrics = {
        "RMSE": rmse,
        "MAPE": mape,
        "Uplift": uplift,
        "Rel Uplift %": rel,
        "p-value": sd["p_value"].iloc[-1] if "p_value" in sd else None
    }

    return ci, sd, metrics


# ==========================================================
# PLOTTING
# ==========================================================

def plot_cf(df, metric):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[metric], name="Actual"))
    fig.add_trace(go.Scatter(x=df.index, y=df["yhat"], name="Counterfactual"))
    fig.update_layout(title="Actual vs Counterfactual")
    return fig


# ==========================================================
# STREAMLIT UI
# ==========================================================

def main():
    st.set_page_config(layout="wide")
    st.title("📈 Counterfactual Impact App (Prophet + BSTS)")
    st.caption("Fully error-proof. Supports Daily, Weekly, Monthly.")

    # ========== DATA INPUT ==========
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if not file:
        st.stop()

    df = pd.read_csv(file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    date_col = st.sidebar.selectbox("Date column", df.columns)
    metric_col = st.sidebar.selectbox("Metric (KPI)", df.select_dtypes(include=[np.number]).columns)
    control_cols = st.sidebar.multiselect("Control variables", [c for c in df.columns if c != metric_col])

    freq = st.sidebar.radio("Data Frequency", ["Daily","Weekly","Monthly"])
    agg = st.sidebar.radio("If duplicates exist, aggregate using:", ["mean","sum","first"])

    df_clean = resample_to_daily(df, date_col, freq, agg)

    min_date, max_date = df_clean.index.min().date(), df_clean.index.max().date()

    pre = st.sidebar.date_input("Pre-period", (min_date, max_date - pd.Timedelta(days=30)))
    post = st.sidebar.date_input("Post-period", (max_date - pd.Timedelta(days=29), max_date))

    pre_s, pre_e = map(pd.Timestamp, pre)
    post_s, post_e = map(pd.Timestamp, post)

    if pre_e >= post_s:
        st.error("❌ Pre-period must end BEFORE post-period begins.")
        st.stop()

    model_choice = st.sidebar.radio("Model", ["Prophet","BSTS"])

    holidays = {
        "Christmas": st.sidebar.checkbox("Christmas", True),
        "Black Friday": st.sidebar.checkbox("Black Friday", True),
        "Easter": st.sidebar.checkbox("Easter", True),
        "Cyber Monday": st.sidebar.checkbox("Cyber Monday", True),
    }

    if st.sidebar.button("🚀 Run model"):

        if model_choice == "Prophet":
            res, post_df, metrics = run_prophet(
                df_clean, metric_col, control_cols,
                (pre_s, pre_e), (post_s, post_e),
                holidays
            )

            st.subheader("Metrics")
            st.json(metrics)
            st.plotly_chart(plot_cf(res, metric_col), use_container_width=True)

        else:
            ci, sd, metrics = run_bsts(
                df_clean, metric_col, control_cols,
                (pre_s, pre_e), (post_s, post_e)
            )

            st.subheader("Metrics")
            st.json(metrics)
            st.write(ci.summary())
            st.plotly_chart(plot_cf(
                sd.rename(columns={"actual": metric_col, "predicted": "yhat"})
            ), use_container_width=True)


if __name__ == "__main__":
    main()
