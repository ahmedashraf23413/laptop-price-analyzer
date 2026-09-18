# ============================================
# 📦 Imports
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder


# ============================================
# ⚙️ Page Config
# ============================================
st.set_page_config(
    page_title="Laptop Price Analyzer",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# 🎨 Custom CSS
# ============================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    h1 {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff2d95);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        font-size: 3rem !important;
        text-align: center;
        padding: 20px 0;
        letter-spacing: 2px;
    }

    h2, h3 {
        color: #00d4ff !important;
        font-weight: 700 !important;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
        transition: all 0.3s ease;
        text-align: center;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(123, 47, 247, 0.4);
        border-color: #7b2ff7;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #b8b8d1;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .metric-icon { font-size: 2.5rem; margin-bottom: 10px; }

    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 15px 40px !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        letter-spacing: 2px !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4) !important;
    }

    .stButton > button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 12px 35px rgba(123, 47, 247, 0.6) !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid rgba(0, 212, 255, 0.3);
    }

    [data-testid="stSidebar"] * { color: #e0e0ff !important; }

    .prediction-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(123, 47, 247, 0.2));
        border: 2px solid #00d4ff;
        border-radius: 25px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 0 40px rgba(0, 212, 255, 0.5);
    }

    .prediction-value {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff2d95);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 15px 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #b8b8d1;
        font-weight: 700;
        padding: 12px 25px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7) !important;
        color: white !important;
    }

    hr { border-color: rgba(0, 212, 255, 0.3); margin: 30px 0; }

    .badge {
        display: inline-block;
        padding: 6px 15px;
        border-radius: 20px;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
        margin: 3px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# 📂 Data Loading & Processing
# ============================================
@st.cache_data
def load_data():
    possible_paths = [
        "Laptop_price.csv",
        "laptop_price.csv",
        "labtop data/Laptop_price.csv",
        r"D:\data preprocessing\labtop data\Laptop_price.csv",
        "../Laptop_price.csv",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)

    st.warning("⚠️ `Laptop_price.csv` not found — using sample data for demo")
    np.random.seed(42)
    n = 1000
    brands = ["Asus", "Acer", "Lenovo", "HP", "Dell"]
    return pd.DataFrame({
        "Brand": np.random.choice(brands, n),
        "Processor_Speed": np.random.uniform(1.5, 4.0, n),
        "RAM_Size": np.random.choice([4, 8, 16, 32], n),
        "Storage_Capacity": np.random.choice([256, 512, 1000], n),
        "Screen_Size": np.random.uniform(11.0, 17.0, n),
        "Weight": np.random.uniform(2.0, 5.0, n),
        "Price": np.random.uniform(8500, 33500, n),
    })


@st.cache_data
def clean_data(df):
    cols = ["Processor_Speed", "Storage_Capacity", "Screen_Size", "Weight", "Price", "RAM_Size"]
    for col in cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
    return df.reset_index(drop=True)


@st.cache_resource
def train_model(df):
    data_x = df.drop("Price", axis=1)
    data_y = df["Price"]

    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, random_state=55, test_size=0.20
    )

    one_hot = OneHotEncoder(handle_unknown="ignore")
    a_data = one_hot.fit_transform(x_train[["Brand"]]).toarray()
    b_data = one_hot.transform(x_test[["Brand"]]).toarray()
    new_cols = one_hot.get_feature_names_out(["Brand"])

    a_df = pd.DataFrame(a_data, index=x_train.index, columns=new_cols)
    b_df = pd.DataFrame(b_data, index=x_test.index, columns=new_cols)

    x_train = pd.concat([a_df, x_train], axis=1).drop(columns=["Brand"])
    x_test = pd.concat([b_df, x_test], axis=1).drop(columns=["Brand"])

    model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "y_test": y_test,
        "y_test_pred": y_test_pred,
    }

    return model, one_hot, metrics, x_train.columns.tolist()


# ============================================
# 🚀 Load & Train
# ============================================
raw_data = load_data()
data = clean_data(raw_data.copy())
model, encoder, metrics, feature_cols = train_model(data)


# ============================================
# 🎯 Header
# ============================================
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1>💻 Laptop Price Analyzer</h1>
    <p style="color: #b8b8d1; font-size: 1.2rem; letter-spacing: 3px;">
        Smart Analysis of Laptop Prices using Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ============================================
# ⚙️ Sidebar
# ============================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 15px 0;">
        <h2 style="color: #00d4ff;">⚙️ Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Navigation")
    page = st.radio(
        "Select page:",
        ["🏠 Overview", "📈 Data Analysis", "🤖 Price Prediction", "🎯 Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📋 Dataset Info")

    st.markdown(f"""
    <div class="metric-card" style="padding: 15px;">
        <div class="metric-label">Total Rows</div>
        <div class="metric-value" style="font-size: 1.8rem;">{len(data):,}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card" style="padding: 15px;">
        <div class="metric-label">Total Columns</div>
        <div class="metric-value" style="font-size: 1.8rem;">{len(data.columns)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏷️ Available Brands")
    for b in sorted(data["Brand"].unique().tolist()):
        st.markdown(f'<span class="badge">{b}</span>', unsafe_allow_html=True)


# ============================================
# 🏠 Overview Page
# ============================================
if page == "🏠 Overview":
    st.markdown("## 📊 Market Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">💰</div>
            <div class="metric-label">Average Price</div>
            <div class="metric-value">{data['Price'].mean():,.0f}</div>
            <div style="color: #b8b8d1; font-size: 0.9rem;">EGP</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-label">Max Price</div>
            <div class="metric-value">{data['Price'].max():,.0f}</div>
            <div style="color: #b8b8d1; font-size: 0.9rem;">EGP</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📉</div>
            <div class="metric-label">Min Price</div>
            <div class="metric-value">{data['Price'].min():,.0f}</div>
            <div style="color: #b8b8d1; font-size: 0.9rem;">EGP</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🏪</div>
            <div class="metric-label">Total Brands</div>
            <div class="metric-value">{data['Brand'].nunique()}</div>
            <div style="color: #b8b8d1; font-size: 0.9rem;">Brands</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🏷️ Brand Distribution")
        brand_counts = data["Brand"].value_counts().reset_index()
        brand_counts.columns = ["Brand", "Count"]

        fig = px.pie(
            brand_counts, values="Count", names="Brand", hole=0.55,
            color_discrete_sequence=px.colors.sequential.Plasma_r
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Statistics")
        st.dataframe(
            data.describe().round(2),
            use_container_width=True,
            height=400
        )

    st.markdown("---")
    st.markdown("### 📋 Data Sample")
    st.dataframe(data.head(10), use_container_width=True)


# ============================================
# 📈 Data Analysis Page
# ============================================
elif page == "📈 Data Analysis":
    st.markdown("## 📈 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "💵 Price & Brand", "⚙️ Specifications",
        "📐 Size & Weight", "🔗 Correlations"
    ])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.box(data, x="Brand", y="Price", color="Brand",
                         title="<b>Price Distribution by Brand</b>",
                         color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            avg_price = data.groupby("Brand")["Price"].mean().reset_index().sort_values("Price")
            fig = px.bar(avg_price, x="Price", y="Brand", orientation="h",
                         color="Price", color_continuous_scale="Plasma",
                         title="<b>Average Price per Brand</b>", text="Price")
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.box(data, x="RAM_Size", y="Price", color="RAM_Size",
                         title="<b>Impact of RAM Size on Price</b>",
                         color_discrete_sequence=px.colors.sequential.Plasma)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(data, x="Storage_Capacity", y="Price", color="Storage_Capacity",
                         title="<b>Impact of Storage Capacity on Price</b>",
                         color_discrete_sequence=px.colors.sequential.Viridis)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(data, x="Processor_Speed", y="Price",
                         color="Brand", size="RAM_Size",
                         title="<b>Processor Speed vs Price</b>",
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         opacity=0.7)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"), height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.scatter(data, x="Screen_Size", y="Weight",
                         color="Brand", size="Price",
                         title="<b>Screen Size vs Weight (point size = Price)</b>",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         opacity=0.7, hover_data=["Price"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"), height=500)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(data, x="Screen_Size", nbins=30,
                               title="<b>Screen Size Distribution</b>",
                               color_discrete_sequence=["#00d4ff"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(data, x="Weight", nbins=30,
                               title="<b>Weight Distribution</b>",
                               color_discrete_sequence=["#7b2ff7"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### 🔥 Correlation Matrix")
        corr = data.select_dtypes(include=[np.number]).corr()

        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="<b>Correlation Matrix Between Variables</b>")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"), height=600)
        st.plotly_chart(fig, use_container_width=True)


# ============================================
# 🤖 Price Prediction Page
# ============================================
elif page == "🤖 Price Prediction":
    st.markdown("## 🤖 Laptop Price Prediction")
    st.markdown("### Enter the laptop specifications to get an estimated price")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏷️ Brand & Specs")
        brand = st.selectbox("🏢 Brand", sorted(data["Brand"].unique().tolist()))
        processor_speed = st.slider(
            "⚡ Processor Speed (GHz)",
            float(data["Processor_Speed"].min()),
            float(data["Processor_Speed"].max()),
            float(data["Processor_Speed"].mean()),
            step=0.1
        )
        ram_size = st.selectbox(
            "🧠 RAM Size (GB)",
            sorted(data["RAM_Size"].unique().tolist())
        )

    with col2:
        st.markdown("#### 📐 Dimensions & Storage")
        storage = st.selectbox(
            "💾 Storage Capacity (GB)",
            sorted(data["Storage_Capacity"].unique().tolist())
        )
        screen_size = st.slider(
            "📺 Screen Size (inch)",
            float(data["Screen_Size"].min()),
            float(data["Screen_Size"].max()),
            float(data["Screen_Size"].mean()),
            step=0.1
        )
        weight = st.slider(
            "⚖️ Weight (kg)",
            float(data["Weight"].min()),
            float(data["Weight"].max()),
            float(data["Weight"].mean()),
            step=0.1
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Predict Price Now", use_container_width=True):
        input_df = pd.DataFrame({
            "Brand": [brand],
            "Processor_Speed": [processor_speed],
            "RAM_Size": [ram_size],
            "Storage_Capacity": [storage],
            "Screen_Size": [screen_size],
            "Weight": [weight],
        })

        brand_encoded = encoder.transform(input_df[["Brand"]]).toarray()
        brand_cols = encoder.get_feature_names_out(["Brand"])
        brand_df = pd.DataFrame(brand_encoded, columns=brand_cols)

        input_final = pd.concat([brand_df, input_df.drop("Brand", axis=1)], axis=1)
        input_final = input_final[feature_cols]

        prediction = model.predict(input_final)[0]

        st.markdown("---")
        st.markdown(f"""
        <div class="prediction-box">
            <div style="font-size: 1.5rem; color: #00d4ff; font-weight: 700; letter-spacing: 3px;">
                💰 Estimated Price
            </div>
            <div class="prediction-value">{prediction:,.0f}</div>
            <div style="font-size: 1.2rem; color: #b8b8d1;">EGP</div>
            <br>
            <div style="color: #b8b8d1; font-size: 0.95rem;">
                📊 Prediction Range: {prediction * 0.9:,.0f} - {prediction * 1.1:,.0f} EGP
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        market_avg = data["Price"].mean()
        diff = prediction - market_avg

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Market Average</div>
                <div class="metric-value" style="font-size: 1.8rem;">{market_avg:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            color = "#00ff88" if diff < 0 else "#ff4757"
            arrow = "▼" if diff < 0 else "▲"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Difference</div>
                <div class="metric-value" style="font-size: 1.8rem; color: {color} !important;
                     -webkit-text-fill-color: {color};">{arrow} {abs(diff):,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            status = "Below Average" if diff < 0 else "Above Average"
            color = "#00ff88" if diff < 0 else "#ff4757"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Evaluation</div>
                <div style="font-size: 1.2rem; color: {color}; font-weight: 800; margin-top: 15px;">
                    {status}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================
# 🎯 Model Performance Page
# ============================================
elif page == "🎯 Model Performance":
    st.markdown("## 🎯 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">Train R²</div>
            <div class="metric-value">{metrics['train_r2']*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">✅</div>
            <div class="metric-label">Test R²</div>
            <div class="metric-value">{metrics['test_r2']*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📏</div>
            <div class="metric-label">Train MAE</div>
            <div class="metric-value" style="font-size: 1.8rem;">{metrics['train_mae']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📐</div>
            <div class="metric-label">Test MAE</div>
            <div class="metric-value" style="font-size: 1.8rem;">{metrics['test_mae']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 R² Comparison")
        fig = go.Figure(data=[
            go.Bar(name="Train", x=["R²"], y=[metrics['train_r2']],
                   marker_color="#00d4ff",
                   text=[f"{metrics['train_r2']*100:.2f}%"], textposition="outside"),
            go.Bar(name="Test", x=["R²"], y=[metrics['test_r2']],
                   marker_color="#7b2ff7",
                   text=[f"{metrics['test_r2']*100:.2f}%"], textposition="outside")
        ])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"), barmode="group",
                          height=400, yaxis=dict(range=[0, 1.1]))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📉 MAE Comparison")
        fig = go.Figure(data=[
            go.Bar(name="Train", x=["MAE"], y=[metrics['train_mae']],
                   marker_color="#00d4ff",
                   text=[f"{metrics['train_mae']:,.0f}"], textposition="outside"),
            go.Bar(name="Test", x=["MAE"], y=[metrics['test_mae']],
                   marker_color="#ff2d95",
                   text=[f"{metrics['test_mae']:,.0f}"], textposition="outside")
        ])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"), barmode="group", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏆 Model Quality Assessment")

    r2 = metrics['test_r2']
    if r2 >= 0.95:
        grade, color, emoji = "Excellent", "#00ff88", "🌟"
    elif r2 >= 0.90:
        grade, color, emoji = "Very Good", "#00d4ff", "⭐"
    elif r2 >= 0.80:
        grade, color, emoji = "Good", "#7b2ff7", "✨"
    elif r2 >= 0.70:
        grade, color, emoji = "Fair", "#ffa502", "👍"
    else:
        grade, color, emoji = "Needs Improvement", "#ff4757", "⚠️"

    st.markdown(f"""
    <div class="metric-card" style="padding: 30px;">
        <div style="font-size: 4rem;">{emoji}</div>
        <div class="metric-label">Model Rating</div>
        <div class="metric-value" style="color: {color} !important;
             -webkit-text-fill-color: {color};">{grade}</div>
        <div style="color: #b8b8d1; margin-top: 15px; font-size: 1.1rem;">
            The model explains <b style="color: #00d4ff;">{r2*100:.2f}%</b> of the price variance
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Feature Importance
    st.markdown("### 🎯 Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})

    translations = {
        "Processor_Speed": "Processor Speed",
        "RAM_Size": "RAM Size",
        "Storage_Capacity": "Storage Capacity",
        "Screen_Size": "Screen Size",
        "Weight": "Weight",
        "Brand_Acer": "Brand: Acer",
        "Brand_Asus": "Brand: Asus",
        "Brand_Dell": "Brand: Dell",
        "Brand_HP": "Brand: HP",
        "Brand_Lenovo": "Brand: Lenovo",
    }
    feat_df["Feature_EN"] = feat_df["Feature"].map(translations).fillna(feat_df["Feature"])
    feat_df = feat_df.sort_values("Importance", ascending=True)

    fig = px.bar(feat_df, x="Importance", y="Feature_EN", orientation="h",
                 color="Importance", color_continuous_scale="Plasma", text="Importance")
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="white"),
                      coloraxis_showscale=False, height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Prediction Error Distribution")

    errors = metrics["y_test"] - metrics["y_test_pred"]

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(x=errors, nbins=40,
                           title="<b>Residuals Distribution</b>",
                           color_discrete_sequence=["#00d4ff"])
        fig.add_vline(x=0, line_dash="dash", line_color="#ff2d95", line_width=2)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(x=metrics["y_test_pred"], y=metrics["y_test"],
                         title="<b>Predicted vs Actual Values</b>",
                         color_discrete_sequence=["#7b2ff7"], opacity=0.6)
        min_val = min(metrics["y_test"].min(), metrics["y_test_pred"].min())
        max_val = max(metrics["y_test"].max(), metrics["y_test_pred"].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Perfect Fit",
            line=dict(color="#ff2d95", dash="dash", width=2)
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"),
                          xaxis_title="Predicted",
                          yaxis_title="Actual")
        st.plotly_chart(fig, use_container_width=True)


# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 30px 0; color: #b8b8d1;">
    <p style="font-size: 1.1rem; letter-spacing: 2px;">
        💻 <b style="color: #00d4ff;">Laptop Price Analyzer</b>
    </p>
    <p style="font-size: 0.9rem;">
        Built with
        <span class""")