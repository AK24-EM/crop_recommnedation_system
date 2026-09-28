import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AgroSense AI | Precision Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Gradient & Hero */
    .hero-container {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.04) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px -10px rgba(16, 185, 129, 0.15);
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        line-height: 1.6;
        margin-bottom: 16px;
    }
    
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }

    /* Result Card */
    .result-card {
        background: linear-gradient(145deg, #0d2319, #081710);
        border: 1px solid rgba(52, 211, 153, 0.35);
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 12px 35px -5px rgba(16, 185, 129, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: "";
        position: absolute;
        top: -50px;
        right: -50px;
        width: 150px;
        height: 150px;
        background: radial-gradient(circle, rgba(52, 211, 153, 0.25) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    
    .crop-name-highlight {
        font-size: 2.4rem;
        font-weight: 800;
        color: #a7f3d0;
        text-transform: capitalize;
        letter-spacing: -0.5px;
    }
    
    /* Metric & Analysis Cards */
    .metric-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 16px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        border-color: rgba(52, 211, 153, 0.4);
    }
    
    .metric-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    
    .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    
    /* Tip Box */
    .tip-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #10b981;
        padding: 14px 18px;
        border-radius: 0 12px 12px 0;
        margin-top: 14px;
    }
    
    /* Tabs custom styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CONSTANTS & CROP METADATA
# ---------------------------------------------------------
CROP_METADATA = {
    'apple': {'emoji': '🍎', 'category': 'Fruit / Temperate', 'desc': 'Thrives in temperate, cooler climates with deep, well-drained soil rich in potassium and phosphorus.'},
    'banana': {'emoji': '🍌', 'category': 'Fruit / Tropical', 'desc': 'Demands warm, humid tropical conditions with rich organic soil, abundant nitrogen, and high moisture.'},
    'blackgram': {'emoji': '🫘', 'category': 'Pulse / Legume', 'desc': 'Hardy leguminous crop that enriches soil nitrogen; flourishes in warm, loamy soils with moderate rainfall.'},
    'chickpea': {'emoji': '🧆', 'category': 'Pulse / Legume', 'desc': 'Cool-season legume ideal for semi-arid zones with deep, aerated neutral soils and low humidity.'},
    'coconut': {'emoji': '🥥', 'category': 'Plantation / Coastal', 'desc': 'Thrives along humid coastal belts with heavy rainfall, abundant sunshine, and sandy-loam soils.'},
    'coffee': {'emoji': '☕', 'category': 'Plantation / Cash Crop', 'desc': 'Cultivated in high-altitude shade with mild temperatures, rich volcanic soil, and steady rainfall.'},
    'cotton': {'emoji': '🌾', 'category': 'Fiber / Cash Crop', 'desc': 'Demands long frost-free periods, warm days, moderate rainfall, and deep fertile black or alluvial soils.'},
    'grapes': {'emoji': '🍇', 'category': 'Fruit / Viticulture', 'desc': 'Grows best in warm, sunny climates with well-drained soils having ample potassium and moderate pH.'},
    'jute': {'emoji': '🌿', 'category': 'Fiber / Commercial', 'desc': 'Golden fiber requiring warm, monsoon climates with heavy rainfall, high humidity, and alluvial clay-loam.'},
    'kidneybeans': {'emoji': '🫘', 'category': 'Pulse / Legume', 'desc': 'Protein-rich pulse that prefers mild temperatures, slightly acidic to neutral soils, and steady moisture.'},
    'lentil': {'emoji': '🍲', 'category': 'Pulse / Legume', 'desc': 'Winter pulse crop performing exceptionally in moderate temperatures and well-drained neutral loam.'},
    'maize': {'emoji': '🌽', 'category': 'Cereal / Grain', 'desc': 'Versatile staple grain requiring warm sunny weather, fertile soils rich in nitrogen, and well-spaced irrigation.'},
    'mango': {'emoji': '🥭', 'category': 'Fruit / Tropical', 'desc': 'Tropical king fruit requiring warm climates with distinct dry spells for flowering and deep, well-drained soil.'},
    'mothbeans': {'emoji': '🌱', 'category': 'Pulse / Arid', 'desc': 'Exceptionally drought-resistant legume ideal for arid zones in sandy, low-moisture soils.'},
    'mungbean': {'emoji': '🥣', 'category': 'Pulse / Legume', 'desc': 'Short-duration summer legume suitable for warm climates and loam soils with moderate rainfall.'},
    'muskmelon': {'emoji': '🍈', 'category': 'Fruit / Cucurbit', 'desc': 'Loves warm, sunny weather and sandy loam soils with moderate watering and low humidity during ripening.'},
    'orange': {'emoji': '🍊', 'category': 'Citrus Fruit', 'desc': 'Citrus crop requiring subtropical warmth, well-drained loamy soil, and balanced moisture without waterlogging.'},
    'papaya': {'emoji': '🍈', 'category': 'Fruit / Tropical', 'desc': 'Fast-growing tropical fruit requiring warm temperatures, high humidity, and rich porous soil with good drainage.'},
    'pigeonpeas': {'emoji': '🫘', 'category': 'Pulse / Legume', 'desc': 'Deep-rooted drought-hardy legume suitable for semi-arid rainfed cropping with moderate fertility.'},
    'pomegranate': {'emoji': '🍎', 'category': 'Fruit / Subtropical', 'desc': 'Flourishes in semi-arid regions with hot dry summers, cool winters, and neutral well-drained soils.'},
    'rice': {'emoji': '🌾', 'category': 'Cereal / Food Crop', 'desc': 'Primary staple grain requiring abundant water, high humidity, warm tropical climate, and heavy clayey loam.'},
    'watermelon': {'emoji': '🍉', 'category': 'Fruit / Cucurbit', 'desc': 'Warm-season vine demanding prolonged sunny days, sandy loam soils, and controlled moisture for sweetness.'},
}

FEATURE_LABELS = {
    'N': 'Nitrogen (N) [kg/ha]',
    'P': 'Phosphorus (P) [kg/ha]',
    'K': 'Potassium (K) [kg/ha]',
    'temperature': 'Temperature [°C]',
    'humidity': 'Relative Humidity [%]',
    'ph': 'Soil pH',
    'rainfall': 'Rainfall [mm]'
}

# ---------------------------------------------------------
# CACHED LOADERS
# ---------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(base_dir, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le

@st.cache_data
def load_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, 'Crop_recommendation.csv'))
    return df

try:
    model, scaler, le = load_ml_assets()
    df = load_dataset()
    crop_stats = df.groupby('label').agg(['mean', 'min', 'max']).round(2)
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# ---------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🌱 AgroSense AI — Intelligent Crop Recommendation</div>
    <div class="hero-subtitle">
        Empowering farmers and agronomists with precision machine learning (Random Forest • 99.55% accuracy). 
        Analyze multi-dimensional soil chemistry, environmental conditions, and climatic parameters to forecast the optimal crop yield.
    </div>
    <div>
        <span class="badge-pill">🌾 22 Crop Varieties</span>
        <span class="badge-pill">🧪 7 Agro-Climatic Indicators</span>
        <span class="badge-pill">⚡ Real-time Multi-Model Confidence</span>
        <span class="badge-pill">🎯 99.55% Model Accuracy</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR CONTROLS & PRESET SCENARIOS
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Soil & Climate Configuration")

# Quick Presets
preset_choice = st.sidebar.selectbox(
    "⚡ Quick Agro-Climatic Presets",
    [
        "Custom Input",
        "Tropical Monsoon (Rice, Jute)",
        "Semi-Arid Dryland (Chickpea, Mothbeans)",
        "Highland Plantation (Coffee)",
        "Humid Coastal (Coconut, Banana)",
        "Rich Alluvial Plain (Cotton, Maize)",
        "Orchard / High Potassium (Apple, Grapes)"
    ]
)

presets = {
    "Custom Input": {"N": 50, "P": 50, "K": 50, "temp": 25.0, "hum": 70.0, "ph": 6.5, "rain": 100.0},
    "Tropical Monsoon (Rice, Jute)": {"N": 90, "P": 45, "K": 40, "temp": 26.5, "hum": 84.0, "ph": 6.7, "rain": 210.0},
    "Semi-Arid Dryland (Chickpea, Mothbeans)": {"N": 35, "P": 65, "K": 80, "temp": 22.0, "hum": 18.0, "ph": 7.2, "rain": 75.0},
    "Highland Plantation (Coffee)": {"N": 100, "P": 30, "K": 30, "temp": 25.0, "hum": 58.0, "ph": 6.8, "rain": 160.0},
    "Humid Coastal (Coconut, Banana)": {"N": 25, "P": 20, "K": 30, "temp": 27.5, "hum": 95.0, "ph": 6.0, "rain": 175.0},
    "Rich Alluvial Plain (Cotton, Maize)": {"N": 120, "P": 45, "K": 20, "temp": 24.0, "hum": 80.0, "ph": 6.9, "rain": 80.0},
    "Orchard / High Potassium (Apple, Grapes)": {"N": 20, "P": 135, "K": 200, "temp": 22.5, "hum": 92.0, "ph": 6.0, "rain": 110.0},
}

current_preset = presets[preset_choice]

st.sidebar.markdown("#### 🧪 Soil Macro-Nutrients (NPK)")
n_input = st.sidebar.slider("Nitrogen (N) [kg/ha]", min_value=0, max_value=140, value=int(current_preset["N"]), help="Essential for vegetative leaf & stem growth")
p_input = st.sidebar.slider("Phosphorus (P) [kg/ha]", min_value=5, max_value=145, value=int(current_preset["P"]), help="Essential for root development and flowering")
k_input = st.sidebar.slider("Potassium (K) [kg/ha]", min_value=5, max_value=205, value=int(current_preset["K"]), help="Essential for drought tolerance & enzyme activation")

st.sidebar.markdown("#### ⛅ Environmental & Climate Factors")
temp_input = st.sidebar.slider("Temperature (°C)", min_value=8.0, max_value=45.0, value=float(current_preset["temp"]), step=0.5, help="Ambient temperature")
hum_input = st.sidebar.slider("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=float(current_preset["hum"]), step=1.0, help="Atmospheric moisture content")
ph_input = st.sidebar.slider("Soil pH Level", min_value=3.5, max_value=10.0, value=float(current_preset["ph"]), step=0.1, help="3.5-5.5 Acidic | 6.0-7.5 Neutral/Optimal | >7.5 Alkaline")
rain_input = st.sidebar.slider("Rainfall (mm)", min_value=20.0, max_value=300.0, value=float(current_preset["rain"]), step=1.0, help="Annual or seasonal precipitation")

# Predict button
predict_btn = st.sidebar.button("🌾 Run Precision Recommendation", type="primary", use_container_width=True)

# ---------------------------------------------------------
# INFERENCE LOGIC
# ---------------------------------------------------------
user_input_df = pd.DataFrame([[n_input, p_input, k_input, temp_input, hum_input, ph_input, rain_input]], 
                             columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

user_scaled = scaler.transform(user_input_df)
pred_idx = model.predict(user_scaled)[0]
top_crop = le.inverse_transform([pred_idx])[0]
all_probas = model.predict_proba(user_scaled)[0]

# Sorted probabilities
top_indices = np.argsort(all_probas)[::-1]
top_crops_list = [(le.inverse_transform([i])[0], all_probas[i]) for i in top_indices]
top_prob = top_crops_list[0][1] * 100

crop_info = CROP_METADATA.get(top_crop, {'emoji': '🌱', 'category': 'General Crop', 'desc': 'Standard commercial crop.'})

# ---------------------------------------------------------
# TABS INTERFACE
# ---------------------------------------------------------
tab_rec, tab_radar, tab_analytics, tab_compare, tab_model = st.tabs([
    "🎯 Crop Recommendation",
    "🕸️ Soil Radar Match",
    "📊 Soil & Climate EDA",
    "🌿 Crop Encyclopedia & Compare",
    "🧠 Model Diagnostics"
])

# ---------------------------------------------------------
# TAB 1: CROP RECOMMENDATION
# ---------------------------------------------------------
with tab_rec:
    col_rec_left, col_rec_right = st.columns([1.1, 0.9], gap="large")
    
    with col_rec_left:
        st.markdown(f"""
        <div class="result-card">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                <span class="badge-pill">✅ Optimal Agro-Match Found</span>
                <span style="color: #6ee7b7; font-weight: 700; font-size: 1rem;">Confidence: {top_prob:.1f}%</span>
            </div>
            <div style="display: flex; align-items: baseline; gap: 14px;">
                <span style="font-size: 3.5rem;">{crop_info['emoji']}</span>
                <div>
                    <div class="crop-name-highlight">{top_crop}</div>
                    <div style="color: #94a3b8; font-size: 0.95rem; font-weight: 500;">Category: {crop_info['category']}</div>
                </div>
            </div>
            <p style="color: #cbd5e1; line-height: 1.6; margin-top: 16px; font-size: 0.95rem;">
                {crop_info['desc']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Agronomic Diagnostic & Fertilizer Guidance
        st.markdown("#### 🔬 Agronomic Field Guidance")
        
        crop_mean = df[df['label'] == top_crop].mean(numeric_only=True)
        
        diff_n = n_input - crop_mean['N']
        diff_p = p_input - crop_mean['P']
        diff_k = k_input - crop_mean['K']
        
        c1, c2, c3 = st.columns(3)
        with c1:
            n_status = "Optimal" if abs(diff_n) <= 15 else ("Surplus (+)" if diff_n > 15 else "Deficient (-)")
            n_color = "#34d399" if abs(diff_n) <= 15 else ("#f59e0b" if diff_n > 15 else "#ef4444")
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Nitrogen (N)</div>
                <div class="metric-val" style="color: {n_color};">{n_status}</div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Input: {n_input} | Ideal: {crop_mean['N']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            p_status = "Optimal" if abs(diff_p) <= 15 else ("Surplus (+)" if diff_p > 15 else "Deficient (-)")
            p_color = "#34d399" if abs(diff_p) <= 15 else ("#f59e0b" if diff_p > 15 else "#ef4444")
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Phosphorus (P)</div>
                <div class="metric-val" style="color: {p_color};">{p_status}</div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Input: {p_input} | Ideal: {crop_mean['P']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            k_status = "Optimal" if abs(diff_k) <= 15 else ("Surplus (+)" if diff_k > 15 else "Deficient (-)")
            k_color = "#34d399" if abs(diff_k) <= 15 else ("#f59e0b" if diff_k > 15 else "#ef4444")
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Potassium (K)</div>
                <div class="metric-val" style="color: {k_color};">{k_status}</div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Input: {k_input} | Ideal: {crop_mean['K']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # pH Classification
        if ph_input < 6.0:
            ph_category = "Slightly Acidic to Acidic (Lime application may be required for sensitive crops)"
        elif 6.0 <= ph_input <= 7.5:
            ph_category = "Neutral to Ideal (Prime nutrient absorption zone)"
        else:
            ph_category = "Alkaline / Saline (Gypsum application or organic compost recommended)"
            
        st.markdown(f"""
        <div class="tip-box">
            <strong style="color: #34d399;">🌾 Soil pH Health:</strong> {ph_category}<br>
            <strong style="color: #34d399;">💧 Moisture Adequacy:</strong> Current rainfall ({rain_input:.0f} mm) vs Ideal average for {top_crop.capitalize()} ({crop_mean['rainfall']:.0f} mm).
        </div>
        """, unsafe_allow_html=True)
        
    with col_rec_right:
        st.markdown("#### 📊 Top Alternative Candidates & Confidence")
        
        # Plotly Bar Chart for top 5 crops
        top5_crops = [x[0].capitalize() for x in top_crops_list[:5]]
        top5_probs = [x[1] * 100 for x in top_crops_list[:5]]
        
        fig_prob = px.bar(
            x=top5_probs,
            y=top5_crops,
            orientation='h',
            labels={'x': 'Prediction Probability (%)', 'y': 'Crop'},
            color=top5_probs,
            color_continuous_scale=[[0, '#064e3b'], [0.5, '#059669'], [1.0, '#34d399']],
            text=[f"{p:.1f}%" for p in top5_probs]
        )
        fig_prob.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis=dict(range=[0, 105], showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=20, t=10, b=10),
            coloraxis_showscale=False,
            font=dict(color='#e2e8f0', family='Plus Jakarta Sans'),
            height=320
        )
        fig_prob.update_traces(textposition='outside', marker_line_color='rgba(255,255,255,0.15)', marker_line_width=1)
        st.plotly_chart(fig_prob, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: RADAR / SPIDER CHART COMPARISON
# ---------------------------------------------------------
with tab_radar:
    st.markdown("### 🕸️ Condition Fit Radar: Your Farm vs. Ideal Benchmark")
    st.markdown(f"Comparing current soil & climate inputs against the typical baseline for **{top_crop.capitalize()}**.")
    
    # Normalized radar chart (0 - 100% of maximum in dataset)
    radar_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    max_vals = df[radar_features].max()
    
    user_normalized = [user_input_df[feat].iloc[0] / max_vals[feat] * 100 for feat in radar_features]
    crop_mean_normalized = [crop_mean[feat] / max_vals[feat] * 100 for feat in radar_features]
    
    # Close polygon
    radar_labels = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temp', 'Humidity', 'pH', 'Rainfall']
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=user_normalized + [user_normalized[0]],
        theta=radar_labels + [radar_labels[0]],
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.35)',
        line=dict(color='#10b981', width=2),
        name='Your Farm Input'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=crop_mean_normalized + [crop_mean_normalized[0]],
        theta=radar_labels + [radar_labels[0]],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.2)',
        line=dict(color='#60a5fa', width=2, dash='dash'),
        name=f'Ideal {top_crop.capitalize()} Profile'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color='#94a3b8', gridcolor='rgba(255,255,255,0.1)'),
            angularaxis=dict(color='#cbd5e1', gridcolor='rgba(255,255,255,0.1)')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='#e2e8f0'), orientation='h', y=-0.15, x=0.25),
        height=480,
        margin=dict(l=40, r=40, t=30, b=50)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: EXPLORATORY DATA ANALYTICS
# ---------------------------------------------------------
with tab_analytics:
    st.markdown("### 📊 Multi-Crop Environmental Intelligence")
    
    col_eda1, col_eda2 = st.columns([1, 1], gap="large")
    
    with col_eda1:
        feature_to_plot = st.selectbox("Select Dimension to Inspect Across All Crops:", list(FEATURE_LABELS.keys()), 
                                       format_func=lambda x: FEATURE_LABELS[x], index=6)
        
        # Sort crops by mean of selected feature
        crop_order = df.groupby('label')[feature_to_plot].mean().sort_values(ascending=False).index
        
        fig_box = px.box(
            df,
            x='label',
            y=feature_to_plot,
            category_orders={'label': crop_order},
            color_discrete_sequence=['#10b981'],
            labels={'label': 'Crop', feature_to_plot: FEATURE_LABELS[feature_to_plot]},
            title=f"Distribution of {FEATURE_LABELS[feature_to_plot]} by Crop"
        )
        fig_box.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(tickangle=-45, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            height=420
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_eda2:
        st.markdown("#### 🧪 Interactive Soil Nutrient Matrix (N vs P vs K)")
        fig_scatter = px.scatter(
            df,
            x='N',
            y='P',
            color='label',
            size='K',
            hover_name='label',
            labels={'N': 'Nitrogen (N)', 'P': 'Phosphorus (P)', 'K': 'Potassium (K)'},
            title="Nitrogen vs Phosphorus with Potassium as Bubble Scale"
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            showlegend=False,
            height=440
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: ENCYCLOPEDIA & DUAL COMPARISON
# ---------------------------------------------------------
with tab_compare:
    st.markdown("### 🌿 Crop Agronomic Encyclopedia & Side-by-Side Comparison")
    
    col_c1, col_c2 = st.columns(2)
    crop_names = sorted(df['label'].unique())
    
    with col_c1:
        crop_a = st.selectbox("Select First Crop (Crop A):", crop_names, index=crop_names.index('rice'))
    with col_c2:
        crop_b = st.selectbox("Select Second Crop (Crop B):", crop_names, index=crop_names.index('coffee'))
        
    meta_a = CROP_METADATA.get(crop_a, {'emoji': '🌱', 'category': 'General', 'desc': ''})
    meta_b = CROP_METADATA.get(crop_b, {'emoji': '🌱', 'category': 'General', 'desc': ''})
    
    stats_a = df[df['label'] == crop_a].describe()
    stats_b = df[df['label'] == crop_b].describe()
    
    col_info_a, col_info_b = st.columns(2, gap="medium")
    
    with col_info_a:
        st.markdown(f"""
        <div class="result-card">
            <h3>{meta_a['emoji']} {crop_a.capitalize()}</h3>
            <p style="color: #94a3b8; font-size: 0.85rem;">{meta_a['category']}</p>
            <p style="color: #cbd5e1; font-size: 0.9rem;">{meta_a['desc']}</p>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <div style="font-size: 0.85rem; line-height: 1.8;">
                <strong>Optimal N:</strong> {stats_a.loc['mean', 'N']:.1f} kg/ha<br>
                <strong>Optimal P:</strong> {stats_a.loc['mean', 'P']:.1f} kg/ha<br>
                <strong>Optimal K:</strong> {stats_a.loc['mean', 'K']:.1f} kg/ha<br>
                <strong>Avg Temp:</strong> {stats_a.loc['mean', 'temperature']:.1f} °C<br>
                <strong>Avg Humidity:</strong> {stats_a.loc['mean', 'humidity']:.1f} %<br>
                <strong>Avg pH:</strong> {stats_a.loc['mean', 'ph']:.2f}<br>
                <strong>Avg Rainfall:</strong> {stats_a.loc['mean', 'rainfall']:.1f} mm
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_info_b:
        st.markdown(f"""
        <div class="result-card" style="border-color: rgba(96, 165, 250, 0.35);">
            <h3>{meta_b['emoji']} {crop_b.capitalize()}</h3>
            <p style="color: #94a3b8; font-size: 0.85rem;">{meta_b['category']}</p>
            <p style="color: #cbd5e1; font-size: 0.9rem;">{meta_b['desc']}</p>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <div style="font-size: 0.85rem; line-height: 1.8;">
                <strong>Optimal N:</strong> {stats_b.loc['mean', 'N']:.1f} kg/ha<br>
                <strong>Optimal P:</strong> {stats_b.loc['mean', 'P']:.1f} kg/ha<br>
                <strong>Optimal K:</strong> {stats_b.loc['mean', 'K']:.1f} kg/ha<br>
                <strong>Avg Temp:</strong> {stats_b.loc['mean', 'temperature']:.1f} °C<br>
                <strong>Avg Humidity:</strong> {stats_b.loc['mean', 'humidity']:.1f} %<br>
                <strong>Avg pH:</strong> {stats_b.loc['mean', 'ph']:.2f}<br>
                <strong>Avg Rainfall:</strong> {stats_b.loc['mean', 'rainfall']:.1f} mm
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 5: MODEL DIAGNOSTICS & EXPLAINABILITY
# ---------------------------------------------------------
with tab_model:
    st.markdown("### 🧠 Model Performance & Feature Weights")
    
    col_m1, col_m2 = st.columns([1, 1], gap="large")
    
    with col_m1:
        st.markdown("""
        <div class="metric-box" style="margin-bottom: 20px;">
            <div class="metric-title">Trained Estimator</div>
            <div class="metric-val" style="color: #34d399;">Random Forest Classifier</div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px;">
                • 200 Decision Trees with Gini Impurity<br>
                • Stratified 80/20 Train-Test Split<br>
                • 5-Fold Stratified Cross-Validation<br>
                • Test Accuracy: <strong style="color: #34d399;">99.55%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m2:
        # Random forest feature importances
        if hasattr(model, 'feature_importances_'):
            feat_imp = pd.DataFrame({
                'Feature': [FEATURE_LABELS[f] for f in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']],
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=True)
            
            fig_imp = px.bar(
                feat_imp,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Greens',
                title="Random Forest Feature Importance",
                labels={'Importance': 'Relative Importance Gini Score'}
            )
            fig_imp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                coloraxis_showscale=False,
                height=340,
                margin=dict(l=10, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_imp, use_container_width=True)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 10px 0;">
    🌱 AgroSense AI Engine • Precision Agriculture & Crop Recommendation Platform • Powered by Scikit-Learn & Streamlit
</div>
""", unsafe_allow_html=True)
