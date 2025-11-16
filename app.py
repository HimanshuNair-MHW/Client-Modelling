import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from io import StringIO

from prophet import Prophet
from causalimpact import CausalImpact
import plotly.graph_objs as go

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dateutil.easter import easter as easter_sunday


# =========================================
# Helpers – Data loading
# =========================================

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


@st.cache_data
def load_google_sheet(sheet_url: str, worksheet_name: str = None) -> pd.DataFrame:
    """
    Reads a Google Sheet using a service account stored in st.secrets.
    You need to add a secret called 'gcp_service_account' with the JSON key.
    """
    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_dict = st.secrets["gcp_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scopes)
    client = gspread.authorize(creds)

    sh = client.open_by_url(sheet_url)
    if worksheet_name:
        ws = sh.worksheet(worksheet_name)
    else:
        ws = sh.sheet1

    data = ws.get_all_records()
    df = pd.DataFrame(data)
    return df


def prepare_df(df: pd.DataFrame, date_col: str):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    df = df.asfreq("D")  # daily frequency, forward-fill below
    df = df.ffill()
    return df


# =========================================
# Helpers – Holiday calendar
# =========================================

def build_holiday_df(index: pd.DatetimeIndex,
                     add_christmas=True,
                     add_black_friday=True,
                     add_easter=True,
                     add_cyber_monday=True) -> pd.DataFrame:
    holidays = []

    years = sorted(set(index.year))

    for y in years:
        # Christmas (25 Dec)
        if add_christmas:
            dt = pd.Timestamp(date(y, 12, 25))
            if dt in index:
                holidays.append({"ds": dt, "holiday": "christmas"})

        # Black Friday = 4th Friday in November
        if add_black_friday:
            nov = pd.date_range(start=date(y, 11, 1), end=date(y, 11, 30), freq="D")
            fridays = [d for d in nov if d.weekday() == 4]  # 4 = Friday
            if len(fridays) >= 4:
                bf = fridays[3]
                if bf in index:
                    holidays.append({"ds": bf, "holiday": "black_friday"})

        # Easter Sunday
        if add_easter:
            es = pd.Timestamp(easter_sunday(y))
            if es in index:
                holidays.append({"ds": es, "holiday": "easter"})

        # Cyber Monday = Monday after Black Friday
        if add_cyber_monday and add_black_friday:
            nov = pd.date_range(start=date(y, 11, 1), end=date(y, 11, 30), freq="D")
            fridays = [d for d in nov if d.weekday() == 4]
            if len(fridays) >= 4:
                bf = fridays[3]
                cm = bf + pd.Timedelta(days=3)  # Fri + 3 = Mon
                if cm in index:
                    holidays.append({"ds": cm, "holiday": "cyber_monday"})

    if not holidays:
        return None

    holidays_df = pd.DataFrame(holidays)
    return holidays_df


# =========================================
# Prophet Counterfactual
# =========================================

def run_prophet_counterfactual(df: pd.DataFrame,
                               metric_col: str,
                               control_cols,
                               pre_period,
                               post_period,
                               holiday_flags):

    # Split pre/post
    pre_start, pre_end = pre_period
    post_start, post_end = post_period

    train = df.loc[pre_start:pre_end]
    full_period = df.loc[pre_start:post_end]

    # Prepare Prophet dataframe
    train_prophet = train.reset_index().rename(columns={"index": "ds", metric_col: "y"})
    full_prophet = full_period.reset_index().rename(columns={"index": "ds"})

    # Add regressors if any
    model = Prophet(weekly_seasonality=True,
                    yearly_seasonality=True,
                    daily_seasonality=False)

    # Add holidays
    holidays_df = build_holiday_df(
        df.index,
        add_christmas=holiday_flags["christmas"],
        add_black_friday=holiday_flags["black_friday"],
        add_easter=holiday_flags["easter"],
        add_cyber_monday=holiday_flags["cyber_monday"]
    )
    if holidays_df is not None:
        model.holidays = holidays_df

    for col in control_cols:
        model.add_regressor(col)

    # Subset columns for train/full
    regressor_cols = list(control_cols)
    if regressor_cols:
        train_prophet[regressor_cols] = train[regressor_cols].reset_index(drop=True)
        full_prophet[regressor_cols] = full_period[regressor_cols].reset_index(drop=True)

    # Fit on pre-period only
    model.fit(train_prophet)

    # Forecast across pre + post
    forecast = model.predict(full_prophet)

    # Join forecast with actuals
    forecast = forecast.set_index("ds")
    result = full_period[[metric_col]].join(
        forecast[["yhat", "yhat_lower", "yhat_upper"]],
        how="left"
    )

    # Metrics on pre-period (fit quality)
    pre_result = result.loc[pre_start:pre_end]
    y_true = pre_result[metric_col]
    y_pred = pre_result["yhat"]

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100

    # Uplift on post-period
    post_result = result.loc[post_start:post_end]
    post_result = post_result.copy()
    post_result["impact"] = post_result[metric_col] - post_result["yhat"]
    post_result["cum_impact"] = post_result["impact"].cumsum()

    total_pred = post_result["yhat"].sum()
    total_impact = post_result["impact"].sum()
    rel_uplift = (total_impact / total_pred) * 100 if total_pred != 0 else np.nan

    metrics = {
        "model": "Prophet",
        "rmse_pre": rmse,
        "mape_pre": mape,
        "total_impact": total_impact,
        "rel_uplift_pct": rel_uplift,
        "p_value": None  # Prophet doesn't give a p-value directly
    }

    return result, post_result, metrics


# =========================================
# CausalImpact Counterfactual (BSTS)
# =========================================

def run_causalimpact_counterfactual(df: pd.DataFrame,
                                    metric_col: str,
                                    control_cols,
                                    pre_period,
                                    post_period):
    pre_start, pre_end = pre_period
    post_start, post_end = post_period

    # Build CausalImpact data: first column = response, others = controls
    cols = [metric_col] + list(control_cols)
    ci_df = df[cols].copy()

    # CausalImpact expects a DataFrame indexed by time
    # pre/post period given as dates
    ci = CausalImpact(ci_df, [pre_start, pre_end], [post_start, post_end])

    summary_df = ci.summary_data  # contains actual, predicted, point_effect, etc.

    # Basic metrics (training fit is inside the model; we'll use whole summary)
    rmse = np.sqrt(np.mean((summary_df["actual"] - summary_df["predicted"]) ** 2))
    mape = np.mean(
        np.abs((summary_df["actual"] - summary_df["predicted"]) /
               np.where(summary_df["actual"] == 0, np.nan, summary_df["actual"]))
    ) * 100

    # Cumulative impact & relative effect for the post-period
    post_mask = (summary_df.index >= post_start) & (summary_df.index <= post_end)
    post_summary = summary_df.loc[post_mask]
    total_impact = post_summary["point_effect"].sum()
    total_pred = post_summary["predicted"].sum()
    rel_uplift = (total_impact / total_pred) * 100 if total_pred != 0 else np.nan

    # p_value from summary_data (if available)
    p_value = None
    if "p_value" in summary_df.columns:
        # Usually last row is cumulative
        p_value = summary_df["p_value"].iloc[-1]

    metrics = {
        "model": "CausalImpact (BSTS)",
        "rmse": rmse,
        "mape": mape,
        "total_impact": total_impact,
        "rel_uplift_pct": rel_uplift,
        "p_value": p_value,
    }

    return ci, summary_df, metrics


# =========================================
# Plotly Chart Helpers
# =========================================

def plot_actual_vs_counterfactual(result_df, metric_col, title="Actual vs Counterfactual"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result_df.index,
        y=result_df[metric_col],
        mode="lines",
        name="Actual"
    ))
    fig.add_trace(go.Scatter(
        x=result_df.index,
        y=result_df["yhat"],
        mode="lines",
        name="Counterfactual"
    ))
    if "yhat_lower" in result_df.columns and "yhat_upper" in result_df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([result_df.index, result_df.index[::-1]]),
            y=pd.concat([result_df["yhat_upper"], result_df["yhat_lower"][::-1]]),
            fill="toself",
            opacity=0.2,
            line=dict(width=0),
            showlegend=False,
            name="Confidence interval"
        ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=metric_col)
    return fig


def plot_impact_series(post_df, title="Pointwise impact"):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=post_df.index,
        y=post_df["impact"],
        name="Impact"
    ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Impact")
    return fig


def plot_cumulative_impact(post_df, title="Cumulative impact"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=post_df.index,
        y=post_df["cum_impact"],
        mode="lines",
        name="Cumulative impact"
    ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Cumulative impact")
    return fig


# =========================================
# Streamlit App
# =========================================

def main():
    st.set_page_config(page_title="Counterfactual Model App", layout="wide")
    st.title("📈 Counterfactual Impact & Uplift App")
    st.caption("Prophet or BSTS (CausalImpact) – with holidays, validation metrics, and uplift.")

    st.sidebar.header("1. Data Source")

    data_source = st.sidebar.radio("Select data source", ["Upload CSV", "Google Sheet"])

    df = None
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            df = load_csv(uploaded_file)
    else:
        sheet_url = st.sidebar.text_input("Google Sheet URL")
        sheet_name = st.sidebar.text_input("Worksheet name (optional)", value="")
        if st.sidebar.button("Load Google Sheet"):
            if sheet_url:
                try:
                    df = load_google_sheet(sheet_url, worksheet_name=sheet_name or None)
                except Exception as e:
                    st.error(f"Error loading Google Sheet: {e}")

    if df is None:
        st.info("👈 Load your data to begin.")
        st.stop()

    st.subheader("Preview of data")
    st.dataframe(df.head())

    # Basic config
    st.sidebar.header("2. Model Configuration")

    date_col = st.sidebar.selectbox("Date column", df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metric_col = st.sidebar.selectbox("KPI / Response column", numeric_cols)
    control_candidates = [c for c in numeric_cols if c != metric_col]
    control_cols = st.sidebar.multiselect("Control variables (optional)", control_candidates)

    # Prepare datetime index
    df_prepared = prepare_df(df, date_col)
    min_date, max_date = df_prepared.index.min().date(), df_prepared.index.max().date()

    st.sidebar.markdown("**Pre-period (training)**")
    pre_period = st.sidebar.date_input(
        "Pre-period range",
        value=(min_date, max_date if (max_date - min_date).days > 30 else min_date),
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(pre_period, tuple):
        pre_start, pre_end = pre_period
    else:
        st.sidebar.error("Please select both start and end of pre-period.")
        st.stop()

    st.sidebar.markdown("**Post-period (intervention)**")
    post_period = st.sidebar.date_input(
        "Post-period range",
        value=(pre_end, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(post_period, tuple):
        post_start, post_end = post_period
    else:
        st.sidebar.error("Please select both start and end of post-period.")
        st.stop()

    model_type = st.sidebar.radio("Model type", ["Prophet", "BSTS / CausalImpact"])

    st.sidebar.header("3. Holidays / High Demand")
    holiday_flags = {
        "christmas": st.sidebar.checkbox("Christmas", value=True),
        "black_friday": st.sidebar.checkbox("Black Friday", value=True),
        "easter": st.sidebar.checkbox("Easter", value=True),
        "cyber_monday": st.sidebar.checkbox("Cyber Monday", value=True),
    }

    run_button = st.sidebar.button("🚀 Run counterfactual")

    if not run_button:
        st.stop()

    # Ensure datetime index
    df_model = df_prepared.copy()

    # Convert date_input (date) to Timestamp for slicing
    pre_start_ts = pd.Timestamp(pre_start)
    pre_end_ts = pd.Timestamp(pre_end)
    post_start_ts = pd.Timestamp(post_start)
    post_end_ts = pd.Timestamp(post_end)

    if model_type == "Prophet":
        st.subheader("Prophet Counterfactual Results")
        result_df, post_df, metrics = run_prophet_counterfactual(
            df_model,
            metric_col=metric_col,
            control_cols=control_cols,
            pre_period=(pre_start_ts, pre_end_ts),
            post_period=(post_start_ts, post_end_ts),
            holiday_flags=holiday_flags
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE (pre)", f"{metrics['rmse_pre']:.2f}")
        col2.metric("MAPE (pre)", f"{metrics['mape_pre']:.2f}%")
        col3.metric("Total uplift (post)", f"{metrics['total_impact']:.2f}")

        col4, col5 = st.columns(2)
        col4.metric("Relative uplift (post)", f"{metrics['rel_uplift_pct']:.2f}%")
        col5.metric("P-value", "N/A")

        st.plotly_chart(
            plot_actual_vs_counterfactual(result_df, metric_col),
            use_container_width=True
        )

        st.plotly_chart(
            plot_impact_series(post_df),
            use_container_width=True
        )

        st.plotly_chart(
            plot_cumulative_impact(post_df),
            use_container_width=True
        )

        # Download option
        csv_buf = StringIO()
        result_df.to_csv(csv_buf)
        st.download_button(
            "Download full results as CSV",
            csv_buf.getvalue(),
            file_name="prophet_counterfactual_results.csv",
            mime="text/csv"
        )

    else:
        st.subheader("CausalImpact (BSTS) Results")

        ci, summary_df, metrics = run_causalimpact_counterfactual(
            df_model,
            metric_col=metric_col,
            control_cols=control_cols,
            pre_period=(pre_start_ts, pre_end_ts),
            post_period=(post_start_ts, post_end_ts),
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{metrics['rmse']:.2f}")
        col2.metric("MAPE", f"{metrics['mape']:.2f}%")
        col3.metric("Total uplift (post)", f"{metrics['total_impact']:.2f}")

        col4, col5 = st.columns(2)
        col4.metric("Relative uplift (post)", f"{metrics['rel_uplift_pct']:.2f}%")
        col5.metric("P-value", f"{metrics['p_value']:.3f}" if metrics['p_value'] is not None else "N/A")

        # CausalImpact has built-in plotting, but we'll also build our own basic plot.
        st.markdown("**CausalImpact textual summary**")
        st.text(ci.summary())

        # Build result-like df for plotting (actual vs predicted)
        # summary_df already indexed by time
        plot_df = summary_df[["actual", "predicted", "predicted_lower", "predicted_upper"]].copy()
        plot_df = plot_df.rename(columns={
            "actual": metric_col,
            "predicted": "yhat",
            "predicted_lower": "yhat_lower",
            "predicted_upper": "yhat_upper"
        })

        st.plotly_chart(
            plot_actual_vs_counterfactual(plot_df, metric_col),
            use_container_width=True
        )

        # Simple impact & cumulative using summary_df
        impact_df = summary_df.copy()
        impact_df["cum_effect"] = impact_df["point_effect"].cumsum()

        post_mask = (impact_df.index >= post_start_ts) & (impact_df.index <= post_end_ts)
        post_impact_df = impact_df.loc[post_mask]

        impact_fig = go.Figure()
        impact_fig.add_trace(go.Bar(
            x=post_impact_df.index,
            y=post_impact_df["point_effect"],
            name="Point effect"
        ))
        impact_fig.update_layout(title="Pointwise impact (post-period)", xaxis_title="Date", yaxis_title="Impact")
        st.plotly_chart(impact_fig, use_container_width=True)

        cum_fig = go.Figure()
        cum_fig.add_trace(go.Scatter(
            x=post_impact_df.index,
            y=post_impact_df["cum_effect"],
            mode="lines",
            name="Cumulative effect"
        ))
        cum_fig.update_layout(title="Cumulative impact (post-period)", xaxis_title="Date", yaxis_title="Cumulative impact")
        st.plotly_chart(cum_fig, use_container_width=True)

        # Download option
        csv_buf = StringIO()
        summary_df.to_csv(csv_buf)
        st.download_button(
            "Download CausalImpact summary as CSV",
            csv_buf.getvalue(),
            file_name="causalimpact_summary.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
