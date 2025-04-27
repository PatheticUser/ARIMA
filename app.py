import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="ETH Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* Modern styling */
    .main-header {
        font-size: 2.5rem;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .sub-header {
        font-size: 1.3rem;
        font-weight: 500;
        margin-top: 0;
    }
    
    /* Cards */
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
    }
    
    /* Risk indicators */
    .risk-low {
        font-weight: bold;
    }
    .risk-medium {
        font-weight: bold;
    }
    .risk-high {
        font-weight: bold;
    }
    
    /* Hide Streamlit branding */
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


def create_logo():
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.add_patch(
        plt.Polygon(
            [[50, 10], [90, 50], [50, 90], [10, 50]],
            fill=True,
            alpha=0.8,
        )
    )
    ax.add_patch(
        plt.Polygon(
            [[50, 25], [75, 50], [50, 75], [25, 50]],
            fill=True,
            color="white",
            alpha=1.0,
        )
    )

    ax.axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)

    img_str = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{img_str}"


@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model, None
    except FileNotFoundError:
        return None, f"Model file not found at {model_path}. Please check the path."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def generate_sample_data():
    start_date = datetime(2017, 1, 1)
    end_date = datetime.now()
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    n = len(date_range)

    trend = np.linspace(100, 3000, n)

    cycle1 = 1000 * np.sin(np.linspace(0, 2 * np.pi, n))
    cycle2 = 1500 * np.sin(np.linspace(0, 1.5 * np.pi, n))

    weekly = 100 * np.sin(np.linspace(0, 52 * 2 * np.pi, n))

    noise = np.random.normal(0, 200, n)

    prices = trend + cycle1 + cycle2 + weekly + noise
    prices = np.maximum(prices, 50)

    df = pd.DataFrame(
        {
            "Date": date_range,
            "Price": prices,
            "Volume": np.random.uniform(1000000, 10000000, n),
        }
    )

    sentiment = np.sin(np.linspace(0, 8 * np.pi, n)) * 0.7 + np.random.normal(0, 0.3, n)
    sentiment = np.clip(sentiment, -1, 1)
    df["Sentiment"] = sentiment

    df.set_index("Date", inplace=True)
    return df


def generate_future_predictions(model, historical_data, days=90, confidence_level=95):
    current_date = datetime.now()
    future_dates = pd.date_range(start=current_date, periods=days, freq="D")

    if model is not None:
        try:
            has_exog = (
                hasattr(model, "exog_names")
                and model.exog_names is not None
                and len(model.exog_names) > 0
            )

            if has_exog:
                exog_count = (
                    len(model.exog_names) if hasattr(model, "exog_names") else 1
                )
                exog = np.zeros((days, exog_count))
                forecast = model.get_forecast(steps=days, exog=exog)
            else:
                forecast = model.get_forecast(steps=days)

            predictions = forecast.predicted_mean
            conf_int = forecast.conf_int(alpha=(100 - confidence_level) / 100)
            lower_bound = conf_int.iloc[:, 0]
            upper_bound = conf_int.iloc[:, 1]

        except Exception as e:
            st.warning(f"Using simulated predictions instead: {str(e)}")
            model = None

    if model is None:
        last_price = historical_data["Price"].iloc[-1]

        trend_factor = 1.0003
        trend = np.array([last_price * (trend_factor**i) for i in range(days)])

        cycle = 0.2 * trend * np.sin(np.linspace(0, 1.5 * np.pi, days))

        weekly = 0.05 * trend * np.sin(np.linspace(0, 52 * 2 * np.pi, days))

        time_factor = np.linspace(1, 2, days)
        noise = np.random.normal(0, 0.01 * trend * time_factor)

        predictions = trend + cycle + weekly + noise

        z_score = {80: 1.282, 85: 1.440, 90: 1.645, 95: 1.96, 99: 2.576}
        selected_z = z_score.get(confidence_level, 1.96)

        std_dev = historical_data["Price"].iloc[-30:].std() * time_factor
        lower_bound = predictions - selected_z * std_dev
        upper_bound = predictions + selected_z * std_dev

        predictions = pd.Series(predictions, index=future_dates)
        lower_bound = pd.Series(lower_bound, index=future_dates)
        upper_bound = pd.Series(upper_bound, index=future_dates)

    forecast_df = pd.DataFrame(
        {
            "Predicted_Price": predictions,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
        }
    )

    n = len(forecast_df)
    forecast_df["Predicted_Volume"] = np.random.uniform(1000000, 10000000, n)

    price_changes = np.diff(np.append([0], predictions)) / predictions
    lagged_sentiment = np.roll(price_changes, 5)
    sentiment = 0.7 * lagged_sentiment + 0.3 * np.random.normal(0, 0.2, n)
    sentiment = np.clip(sentiment, -1, 1)
    forecast_df["Predicted_Sentiment"] = sentiment

    return forecast_df


def calculate_technical_indicators(data):
    price_col = "Price" if "Price" in data.columns else "Predicted_Price"

    data["MA_7"] = data[price_col].rolling(window=7).mean()
    data["MA_30"] = data[price_col].rolling(window=30).mean()

    delta = data[price_col].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    data["EMA_12"] = data[price_col].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data[price_col].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    return data


def generate_scenarios(base_forecast):
    bullish = base_forecast.copy()
    bearish = base_forecast.copy()

    growth_factor = np.linspace(1, 1.3, len(bullish))
    bullish["Predicted_Price"] = base_forecast["Predicted_Price"] * growth_factor

    decline_factor = np.linspace(1, 0.7, len(bearish))
    bearish["Predicted_Price"] = base_forecast["Predicted_Price"] * decline_factor

    bullish["Lower_Bound"] = bullish["Predicted_Price"] * 0.85
    bullish["Upper_Bound"] = bullish["Predicted_Price"] * 1.15

    bearish["Lower_Bound"] = bearish["Predicted_Price"] * 0.85
    bearish["Upper_Bound"] = bearish["Predicted_Price"] * 1.15

    return {"base": base_forecast, "bullish": bullish, "bearish": bearish}


def calculate_risk_score(volatility, trend, sentiment):
    vol_score = min(1, volatility / 0.1)
    trend_score = (trend + 0.1) / 0.2
    sent_score = (sentiment + 1) / 2

    risk_score = (
        vol_score * 0.5 + (1 - trend_score) * 0.3 + (1 - sent_score) * 0.2
    ) * 100
    return min(100, max(0, risk_score))


def get_risk_category(score):
    if score < 33:
        return "Low", "risk-low"
    elif score < 66:
        return "Medium", "risk-medium"
    else:
        return "High", "risk-high"


def generate_market_insights(historical_data, forecast_data):
    last_price = historical_data["Price"].iloc[-1]
    end_price = forecast_data["Predicted_Price"].iloc[-1]

    expected_return = (end_price / last_price - 1) * 100

    daily_returns = historical_data["Price"].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100

    forecast_daily_returns = forecast_data["Predicted_Price"].pct_change().dropna()
    forecast_volatility = forecast_daily_returns.std() * np.sqrt(252) * 100

    risk_free_rate = 2.0
    sharpe_ratio = (expected_return - risk_free_rate) / forecast_volatility

    cumulative_returns = (1 + forecast_daily_returns).cumprod()
    max_return = cumulative_returns.cummax()
    drawdown = (cumulative_returns / max_return - 1) * 100
    max_drawdown = drawdown.min()

    recent_volatility = daily_returns.iloc[-30:].std() * np.sqrt(252)
    recent_trend = daily_returns.iloc[-30:].mean() * 30
    recent_sentiment = (
        historical_data["Sentiment"].iloc[-30:].mean()
        if "Sentiment" in historical_data
        else 0
    )
    risk_score = calculate_risk_score(recent_volatility, recent_trend, recent_sentiment)
    risk_category, risk_class = get_risk_category(risk_score)

    return {
        "expected_return": expected_return,
        "volatility": volatility,
        "forecast_volatility": forecast_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "risk_class": risk_class,
    }


def main():
    with st.sidebar:
        logo_url = create_logo()
        st.markdown(
            f"""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <img src="{logo_url}" width="50">
            <h1 style="margin-left: 10px; margin-bottom: 0;">ETH Predictor</h1>
        </div>
        """,
            unsafe_allow_html=True,
        )

        model_path = "Model/sarimax_model.pkl"
        model, error_message = load_model(model_path)

        if error_message:
            st.warning(error_message)
            st.info("Using simulated model for demonstration.")
        else:
            st.success("SARIMAX model loaded successfully!")

        prediction_days = st.slider("Prediction Horizon (Days)", 30, 120, 90)

        confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)

        scenario = st.radio(
            "Market Scenario", ["Base Case", "Bullish", "Bearish"], index=0
        )

        show_ma = st.checkbox("Show Moving Averages", True)
        show_volume = st.checkbox("Show Volume", False)

        st.markdown(
            """
        **ETH Predictor** uses time series modeling to forecast Ethereum prices.
        
        **Model:** SARIMAX
        **Training Data:** 2017-2024
        """
        )

    st.markdown(
        '<p class="main-header">Ethereum Price Forecast</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Advanced Analytics & Future Predictions</p>',
        unsafe_allow_html=True,
    )

    historical_data = generate_sample_data()

    historical_data = calculate_technical_indicators(historical_data)

    forecast_data = generate_future_predictions(
        model, historical_data, days=prediction_days, confidence_level=confidence_level
    )

    forecast_data = calculate_technical_indicators(forecast_data)

    scenarios = generate_scenarios(forecast_data)

    if scenario == "Bullish":
        forecast_data = scenarios["bullish"]
    elif scenario == "Bearish":
        forecast_data = scenarios["bearish"]
    else:
        forecast_data = scenarios["base"]

    insights = generate_market_insights(historical_data, forecast_data)

    tab1, tab2, tab3 = st.tabs(
        ["Price Forecast", "Technical Analysis", "Market Insights"]
    )

    with tab1:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            current_price = historical_data["Price"].iloc[-1]
            predicted_end_price = forecast_data["Predicted_Price"].iloc[-1]
            price_change = ((predicted_end_price / current_price) - 1) * 100
            st.metric("Current Price", f"${current_price:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                f"Predicted Price (Day {prediction_days})",
                f"${predicted_end_price:.2f}",
                f"{price_change:.2f}%",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            highest_price = forecast_data["Predicted_Price"].max()
            highest_date = (
                forecast_data["Predicted_Price"].idxmax().strftime("%b %d, %Y")
            )
            st.metric(
                "Highest Predicted", f"${highest_price:.2f}", f"on {highest_date}"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            volatility = (
                forecast_data["Predicted_Price"].pct_change().std() * np.sqrt(252) * 100
            )
            st.metric("Predicted Volatility", f"{volatility:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader(f"ETH Price Forecast (Next {prediction_days} Days)")

        fig = go.Figure()

        historical_end = historical_data.index[-1]
        historical_start = historical_end - timedelta(days=90)
        historical_subset = historical_data.loc[historical_start:historical_end]

        fig.add_trace(
            go.Scatter(
                x=historical_subset.index,
                y=historical_subset["Price"],
                mode="lines",
                name="Historical",
            )
        )

        fig.add_shape(
            type="line",
            x0=historical_end,
            y0=forecast_data["Lower_Bound"].min() * 0.9,
            x1=historical_end,
            y1=forecast_data["Upper_Bound"].max() * 1.1,
            line=dict(color="gray", width=1, dash="dash"),
        )

        fig.add_annotation(
            x=historical_end,
            y=forecast_data["Upper_Bound"].max() * 1.05,
            text="Forecast Start",
            showarrow=False,
            yshift=10,
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_data.index,
                y=forecast_data["Predicted_Price"],
                mode="lines",
                name="Forecast",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_data.index.tolist() + forecast_data.index.tolist()[::-1],
                y=forecast_data["Upper_Bound"].tolist()
                + forecast_data["Lower_Bound"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(58, 123, 213, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name=f"{confidence_level}% Confidence Interval",
                showlegend=True,
            )
        )

        if show_ma:
            if "MA_7" in historical_subset.columns:
                fig.add_trace(
                    go.Scatter(
                        x=historical_subset.index,
                        y=historical_subset["MA_7"],
                        mode="lines",
                        name="7-Day MA (Historical)",
                    )
                )

            if "MA_30" in historical_subset.columns:
                fig.add_trace(
                    go.Scatter(
                        x=historical_subset.index,
                        y=historical_subset["MA_30"],
                        mode="lines",
                        name="30-Day MA (Historical)",
                    )
                )

            if "MA_7" in forecast_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_data.index,
                        y=forecast_data["MA_7"],
                        mode="lines",
                        name="7-Day MA (Forecast)",
                    )
                )

            if "MA_30" in forecast_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_data.index,
                        y=forecast_data["MA_30"],
                        mode="lines",
                        name="30-Day MA (Forecast)",
                    )
                )

        fig.update_layout(
            height=600,
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            margin=dict(l=0, r=0, t=10, b=0),
        )

        fig.update_xaxes(title_text="Date", gridcolor="rgba(220,220,220,0.5)")
        fig.update_yaxes(title_text="Price (USD)", gridcolor="rgba(220,220,220,0.5)")

        st.plotly_chart(fig, use_container_width=True)

        if show_volume:
            st.subheader("Trading Volume")

            fig_volume = go.Figure()

            fig_volume.add_trace(
                go.Bar(
                    x=historical_subset.index,
                    y=historical_subset["Volume"],
                    name="Historical Volume",
                    marker_color="rgba(58, 123, 213, 0.5)",
                )
            )

            fig_volume.add_trace(
                go.Bar(
                    x=forecast_data.index,
                    y=forecast_data["Predicted_Volume"],
                    name="Predicted Volume",
                    marker_color="rgba(0, 210, 255, 0.5)",
                )
            )

            fig_volume.update_layout(
                height=300,
                xaxis_title="Date",
                yaxis_title="Volume",
                template="plotly_white",
                margin=dict(l=0, r=0, t=10, b=0),
            )

            st.plotly_chart(fig_volume, use_container_width=True)

        st.subheader("Detailed Price Predictions")

        selected_date = st.slider(
            "Select Date",
            min_value=forecast_data.index.min().to_pydatetime(),
            max_value=forecast_data.index.max().to_pydatetime(),
            value=forecast_data.index.min().to_pydatetime(),
            format="YYYY-MM-DD",
        )

        closest_date = forecast_data.index[
            forecast_data.index.get_indexer([selected_date], method="nearest")[0]
        ]

        selected_prediction = forecast_data.loc[closest_date]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                f"Predicted Price on {closest_date.strftime('%b %d, %Y')}",
                f"${selected_prediction['Predicted_Price']:.2f}",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Lower Bound",
                f"${selected_prediction['Lower_Bound']:.2f}",
                f"{confidence_level}% Confidence Interval",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Upper Bound",
                f"${selected_prediction['Upper_Bound']:.2f}",
                f"{confidence_level}% Confidence Interval",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Weekly Price Summary")

        weekly_summary = forecast_data.resample("W").agg(
            {
                "Predicted_Price": ["mean", "min", "max", "std"],
                "Lower_Bound": "mean",
                "Upper_Bound": "mean",
            }
        )

        weekly_summary.columns = [
            "Average",
            "Minimum",
            "Maximum",
            "Std Dev",
            "Lower Bound",
            "Upper Bound",
        ]
        weekly_summary.index = weekly_summary.index.strftime("%b %d, %Y")

        formatted_summary = weekly_summary.copy()
        for col in formatted_summary.columns:
            formatted_summary[col] = formatted_summary[col].map("${:.2f}".format)

        st.dataframe(formatted_summary, use_container_width=True)

        csv = forecast_data.to_csv()
        st.download_button(
            label="Download Complete Forecast Data",
            data=csv,
            file_name="eth_price_forecast.csv",
            mime="text/csv",
        )

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Relative Strength Index (RSI)")

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data["RSI"],
                    mode="lines",
                    name="RSI",
                )
            )

            fig.add_shape(
                type="line",
                x0=forecast_data.index[0],
                y0=70,
                x1=forecast_data.index[-1],
                y1=70,
                line=dict(color="red", width=1, dash="dash"),
            )

            fig.add_shape(
                type="line",
                x0=forecast_data.index[0],
                y0=30,
                x1=forecast_data.index[-1],
                y1=30,
                line=dict(color="green", width=1, dash="dash"),
            )

            fig.add_annotation(
                x=forecast_data.index[0],
                y=70,
                text="Overbought",
                showarrow=False,
                yshift=10,
                font=dict(size=10, color="red"),
            )

            fig.add_annotation(
                x=forecast_data.index[0],
                y=30,
                text="Oversold",
                showarrow=False,
                yshift=-10,
                font=dict(size=10, color="green"),
            )

            fig.update_layout(
                height=300,
                xaxis_title="Date",
                yaxis_title="RSI",
                yaxis=dict(range=[0, 100]),
                template="plotly_white",
                margin=dict(l=0, r=0, t=10, b=0),
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("MACD Indicator")

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data["MACD"],
                    mode="lines",
                    name="MACD",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data["MACD_Signal"],
                    mode="lines",
                    name="Signal",
                )
            )

            histogram_y = forecast_data["MACD"] - forecast_data["MACD_Signal"]
            colors = ["green" if val >= 0 else "red" for val in histogram_y]

            fig.add_trace(
                go.Bar(
                    x=forecast_data.index,
                    y=histogram_y,
                    name="Histogram",
                    marker_color=colors,
                )
            )

            fig.update_layout(
                height=300,
                xaxis_title="Date",
                yaxis_title="MACD",
                template="plotly_white",
                margin=dict(l=0, r=0, t=10, b=0),
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Support and Resistance Levels")

        prices = forecast_data["Predicted_Price"].values

        support_levels = []
        resistance_levels = []

        for i in range(2, len(prices) - 2):
            if (
                prices[i - 2]
                > prices[i - 1]
                > prices[i]
                < prices[i + 1]
                < prices[i + 2]
            ):
                support_levels.append(prices[i])

            if (
                prices[i - 2]
                < prices[i - 1]
                < prices[i]
                > prices[i + 1]
                > prices[i + 2]
            ):
                resistance_levels.append(prices[i])

        support_levels = sorted(set([round(level, 2) for level in support_levels]))[:3]
        resistance_levels = sorted(
            set([round(level, 2) for level in resistance_levels]), reverse=True
        )[:3]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            for i, level in enumerate(support_levels):
                st.markdown(f"**S{i+1}:** ${level:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            for i, level in enumerate(resistance_levels):
                st.markdown(f"**R{i+1}:** ${level:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=insights["risk_score"],
                    title={"text": "Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "steps": [
                            {"range": [0, 33], "color": "rgba(76, 175, 80, 0.3)"},
                            {"range": [33, 66], "color": "rgba(255, 152, 0, 0.3)"},
                            {"range": [66, 100], "color": "rgba(244, 67, 54, 0.3)"},
                        ],
                        "threshold": {
                            "thickness": 0.75,
                            "value": insights["risk_score"],
                        },
                    },
                )
            )

            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))

            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f"**Risk Category:** <span class='{insights['risk_class']}'>{insights['risk_category']}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)

            st.markdown(f"**Expected Return:** {insights['expected_return']:.2f}%")
            st.markdown(f"**Volatility (Historical):** {insights['volatility']:.2f}%")
            st.markdown(
                f"**Volatility (Forecast):** {insights['forecast_volatility']:.2f}%"
            )
            st.markdown(f"**Sharpe Ratio:** {insights['sharpe_ratio']:.2f}")
            st.markdown(f"**Maximum Drawdown:** {insights['max_drawdown']:.2f}%")

            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Market Sentiment Analysis")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=forecast_data.index,
                y=forecast_data["Predicted_Sentiment"],
                mode="lines",
                name="Sentiment",
            )
        )

        fig.add_shape(
            type="line",
            x0=forecast_data.index[0],
            y0=0,
            x1=forecast_data.index[-1],
            y1=0,
            line=dict(color="black", width=1, dash="dash"),
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_data.index,
                y=[0] * len(forecast_data),
                fill="tonexty",
                fillcolor="rgba(76, 175, 80, 0.2)",
                line=dict(width=0),
                name="Positive Sentiment",
            )
        )

        fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            yaxis=dict(range=[-1, 1]),
            template="plotly_white",
            margin=dict(l=0, r=0, t=10, b=0),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Scenario Comparison")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=scenarios["base"].index,
                y=scenarios["base"]["Predicted_Price"],
                mode="lines",
                name="Base Case",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=scenarios["bullish"].index,
                y=scenarios["bullish"]["Predicted_Price"],
                mode="lines",
                name="Bullish",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=scenarios["bearish"].index,
                y=scenarios["bearish"]["Predicted_Price"],
                mode="lines",
                name="Bearish",
            )
        )

        fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            margin=dict(l=0, r=0, t=10, b=0),
        )

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
