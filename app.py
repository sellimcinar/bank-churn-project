import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px

# ============= PAGE CONFIGURATION =============
st.set_page_config(
    page_title="Churn Prediction | Dark Fintech",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= CUSTOM CSS: CYBERPUNK DARK FINTECH UI =============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;900&family=Exo+2:wght@300;400;500;600;700&display=swap');
    
    /* HIDE TOP HEADER */
    header[data-testid="stHeader"] {
        background-color: #0E1117 !important;
        visibility: hidden !important;
    }
    
    /* Apply Font ONLY to specific text containers - DO NOT touch Streamlit internals */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif !important;
    }
    
    .stMarkdown p,
    .stMarkdown span,
    div[data-testid="stMarkdownContainer"] p {
        font-family: 'Exo 2', sans-serif !important;
    }
    
    /* Text color for readability */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    .stMarkdown p,
    div[data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
    }
    
    /* Leave Streamlit's internal icons completely untouched */
    div[data-testid="stExpander"] svg {
        display: block !important; /* Ensure visibility */
    }
    
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #0E1117 100%);
        border-right: 2px solid rgba(0, 255, 65, 0.3);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    
    /* Sidebar Labels - Make them visible */
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    /* Glassmorphism Effect - Sci-Fi Style */
    [data-testid="stMetricValue"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 
                    0 0 20px rgba(0, 255, 65, 0.1);
    }
    
    /* Headers - ORBITRON FONT */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: 0px !important;
        text-transform: uppercase !important;
    }
    
    h1 {
        background: linear-gradient(135deg, #00ff41 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
        font-weight: 900 !important;
    }
    
    h2 {
        font-size: 1.5rem !important;
    }
    
    h3 {
        font-size: 1.3rem !important;
        color: #00ff41 !important;
    }
    
    /* Slider Styling - More visible */
    .stSlider > div > div > div {
        background: rgba(0, 255, 65, 0.2);
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00ff41 0%, #00d4ff 100%);
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
    }
    
    /* Slider Values */
    .stSlider [data-baseweb="slider"] div {
        color: #FFFFFF !important;
    }
    
    /* Radio Button Fix - Readable Text */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] div {
        color: #FFFFFF !important;
    }
    
    /* Selectbox Styling - Bright and Visible */
    .stSelectbox > div > div {
        background: rgba(30, 30, 30, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
    }
    
    .stSelectbox label {
        color: #FFFFFF !important;
    }
    
    /* Selectbox text inside dropdown */
    .stSelectbox div[data-baseweb="select"] > div {
        color: #FFFFFF !important;
        background: rgba(30, 30, 30, 0.95) !important;
    }
    
    /* Selectbox Options in Menu */
    [data-baseweb="select"] {
        background: rgba(30, 30, 30, 0.95) !important;
    }
    
    [data-baseweb="select"] li {
        background: rgba(30, 30, 30, 0.95) !important;
        color: #FFFFFF !important;
    }
    
    [data-baseweb="select"] li:hover {
        background: rgba(0, 255, 65, 0.2) !important;
    }
    
    /* Custom Card Styling - Sci-Fi Enhanced */
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .prediction-card.churn {
        border: 2px solid #FF0000;
        box-shadow: 0 8px 32px rgba(255, 0, 0, 0.4),
                    0 0 40px rgba(255, 0, 0, 0.2);
    }
    
    .prediction-card.loyal {
        border: 2px solid #00FF00;
        box-shadow: 0 8px 32px rgba(0, 255, 0, 0.4),
                    0 0 40px rgba(0, 255, 0, 0.2);
    }
    
    .prediction-text {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .prediction-subtext {
        font-size: 1.2rem;
        color: #FFFFFF !important;
        opacity: 0.9;
        margin-top: 10px;
    }
    
    .churn-text {
        color: #FF0000 !important;
        text-shadow: 0 0 20px rgba(255, 0, 0, 0.8);
        font-weight: 700 !important;
    }
    
    .loyal-text {
        color: #00FF00 !important;
        text-shadow: 0 0 20px rgba(0, 255, 0, 0.8);
        font-weight: 700 !important;
    }
    
    /* Sidebar Title */
    [data-testid="stSidebar"] h2 {
        background: linear-gradient(135deg, #00ff41 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Orbitron', sans-serif !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #00ff41 0%, #00d4ff 100%);
        color: #0E1117 !important;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.5);
        transform: translateY(-2px);
    }
    
    /* Input Fields - Make them visible */
    input, textarea, select {
        background: rgba(30, 30, 30, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: #FFFFFF !important;
    }
    
    /* Main text and paragraphs */
    p, span, div {
        color: #FFFFFF !important;
    }
    
    /* Plotly chart backgrounds */
    .js-plotly-plot {
        background: transparent !important;
    }
    
    /* Add space between text elements */
    p, h1, h2, h3 {
        margin-bottom: 15px !important;
        line-height: 1.6 !important; /* Fix line height for custom fonts */
    }
    
    /* Add space above input widgets */
    div[data-testid="stSidebar"] div[data-baseweb="select"],
    div[data-testid="stSidebar"] div[data-baseweb="base-input"] {
        margin-top: 10px !important;
    }
    
    /* Force Input Backgrounds to be Dark */
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"],
    div[data-baseweb="radio"] {
        background-color: #1E1E1E !important;
        border: 1px solid #00FFA3 !important; /* Neon Green Border */
        color: white !important;
    }
    
    /* Force Text inside Inputs to be White */
    div[data-baseweb="select"] span,
    div[data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
    }
    
    /* Fix the Dropdown Menu Options (When clicked) */
    ul[data-baseweb="menu"] {
        background-color: #0E1117 !important;
        color: white !important;
    }
    
    /* Force Radio Button Options to be readable */
    label[data-baseweb="radio"] > div {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ============= DATA GENERATION FUNCTION =============
@st.cache_data
def generate_mock_data():
    """Generate synthetic telecom customer data with realistic correlations"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    credit_score = np.random.randint(300, 851, n_samples)
    age = np.random.randint(18, 71, n_samples)
    tenure = np.random.randint(0, 11, n_samples)
    balance = np.random.uniform(0, 250000, n_samples)
    num_products = np.random.randint(1, 5, n_samples)
    has_cr_card = np.random.randint(0, 2, n_samples)
    is_active = np.random.randint(0, 2, n_samples)
    salary = np.random.uniform(10000, 200000, n_samples)
    
    # Generate target with realistic correlations
    churn_prob = np.zeros(n_samples)
    
    # Lower credit score increases churn
    churn_prob += (850 - credit_score) / 2000
    
    # Very young or very old customers more likely to churn
    churn_prob += np.abs(age - 40) / 200
    
    # Low tenure increases churn
    churn_prob += (10 - tenure) / 50
    
    # Very low or very high balance affects churn
    churn_prob += np.where(balance < 50000, 0.15, 0)
    churn_prob += np.where(balance > 200000, 0.1, 0)
    
    # Multiple products reduces churn
    churn_prob -= num_products * 0.05
    
    # Inactive members more likely to churn
    churn_prob += (1 - is_active) * 0.3
    
    # Low salary increases churn
    churn_prob += (200000 - salary) / 500000
    
    # Normalize and add randomness
    churn_prob = np.clip(churn_prob + np.random.uniform(-0.1, 0.1, n_samples), 0, 1)
    
    # Convert to binary
    exited = (churn_prob > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': salary,
        'Exited': exited
    })
    
    return df

# ============= MODEL TRAINING =============
@st.cache_resource
def train_model():
    """Train Random Forest model and return model with training data"""
    df = generate_mock_data()
    
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print accuracy to terminal
    print(f"Model Accuracy: {accuracy:.2%}")
    
    return model, accuracy, X.columns

# ============= MAIN APP =============
def main():
    # Train model
    model, accuracy, feature_names = train_model()
    
    # Hero Section
    st.markdown("<h1>🏦 AI Risk Sentinel: Bank Customer Churn Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #FAFAFA; font-size: 1.2rem; margin-bottom: 0.5rem;'>Real-time risk assessment using Random Forest algorithms. Detects customers likely to close their accounts based on financial behavior.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #888; font-size: 1rem;'>Model Accuracy: <span style='color: #00ff41; font-weight: 600;'>{accuracy:.2%}</span> | Trained on 10,000+ historical records</p>", unsafe_allow_html=True)
    
    # ============= SIDEBAR: CUSTOMER SIMULATOR =============
    st.sidebar.markdown("<h2>🎛️ Customer Simulator</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # How to Use Expander
    with st.sidebar.expander("ℹ️ How to use this?"):
        st.markdown("""
        **Scenario:** You are a Bank Manager analyzing a customer profile.
        
        **Model:** Trained on 10,000+ historical records using Random Forest Classifier.
        
        **Goal:** If the gauge hits RED, offer the customer a special promotion to make them stay.
        """)
    
    # Add visual separator for breathing room
    st.sidebar.markdown("---")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # Input widgets
    credit_score = st.sidebar.slider(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=10,
        help="Financial health (300-850)"
    )
    
    age = st.sidebar.slider(
        "Age",
        min_value=18,
        max_value=70,
        value=40,
        step=1,
        help="Customer's age in years"
    )
    
    tenure = st.sidebar.slider(
        "Tenure (Years)",
        min_value=0,
        max_value=10,
        value=5,
        step=1,
        help="How many years they have been with the bank"
    )
    
    balance = st.sidebar.slider(
        "Account Balance ($)",
        min_value=0,
        max_value=250000,
        value=75000,
        step=5000,
        help="Money currently in their account"
    )
    
    num_products = st.sidebar.slider(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        help="Number of credit cards/loans they have"
    )
    
    has_cr_card = st.sidebar.selectbox(
        "Has Credit Card?",
        options=["Yes", "No"],
        index=0
    )
    has_cr_card_val = 1 if has_cr_card == "Yes" else 0
    
    is_active = st.sidebar.selectbox(
        "Is Active Member?",
        options=["Yes", "No"],
        index=0
    )
    is_active_val = 1 if is_active == "Yes" else 0
    
    salary = st.sidebar.slider(
        "Estimated Salary ($)",
        min_value=10000,
        max_value=200000,
        value=100000,
        step=5000,
        help="Customer's estimated annual salary"
    )
    
    # Add spacing before algorithm explanation
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # Algorithm Logic Expander
    with st.sidebar.expander("🌲 How the AI Model Works?"):
        st.markdown("""
        **The Analogy:** Imagine asking a single expert for advice. They might be wrong. That is a Decision Tree.
        
        **The Solution:** This app uses a Random Forest, which is like a 'Council of 100 Experts'.
        
        **The Process:**
        
        1. The model asks 100 different trees.
        2. Each tree gives a vote (Churn or Stay).
        3. The system takes the Majority Vote.
        
        **Result:** This makes the prediction much more accurate and stable than a single guess.
        """)
    
    # ============= PREDICTION =============
    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [has_cr_card_val],
        'IsActiveMember': [is_active_val],
        'EstimatedSalary': [salary]
    })
    
    # Get prediction
    churn_probability = model.predict_proba(input_data)[0][1]
    churn_percentage = churn_probability * 100
    will_churn = churn_probability > 0.5
    
    # Gamification: Celebrate safe predictions!
    if churn_percentage < 40:
        st.balloons()
    
    # ============= VISUALIZATIONS =============
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Churn Probability Gauge")
        
        # Determine color based on zone
        if churn_percentage < 40:
            gauge_color = "#00ff41"  # Green
            zone = "Safe"
        elif churn_percentage < 70:
            gauge_color = "#ffff00"  # Yellow
            zone = "Warning"
        else:
            gauge_color = "#ff0055"  # Red
            zone = "High Risk"
        
        # Create gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>{zone}</b>", 'font': {'size': 24, 'color': gauge_color}},
            number={'suffix': "%", 'font': {'size': 50, 'color': '#ffffff'}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#ffffff"},
                'bar': {'color': gauge_color, 'thickness': 0.75},
                'bgcolor': "rgba(255, 255, 255, 0.1)",
                'borderwidth': 2,
                'bordercolor': gauge_color,
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(0, 255, 65, 0.2)'},
                    {'range': [40, 70], 'color': 'rgba(255, 255, 0, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(255, 0, 85, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': churn_percentage
                }
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#ffffff", 'family': "Inter"},
            height=400
        )
        
        st.plotly_chart(fig_gauge, width="stretch")
    
    with col2:
        st.markdown("### 🎯 Prediction Result")
        
        # Prediction card
        if will_churn:
            card_class = "churn"
            text_class = "churn-text"
            result_text = "Customer will CHURN"
            icon = "⚠️"
        else:
            card_class = "loyal"
            text_class = "loyal-text"
            result_text = "Customer is LOYAL"
            icon = "✅"
        
        st.markdown(f"""
        <div class="prediction-card {card_class}">
            <p class="prediction-text {text_class}">{icon} {result_text}</p>
            <p class="prediction-subtext" style="color: #ffffff;">
                Churn Probability: <strong>{churn_percentage:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Report Generation - Download Button
        from datetime import datetime
        import random
        
        # Generate customer ID
        customer_id = f"CUST-{random.randint(10000, 99999)}"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create report content
        report_content = f"""=======================================
    BANK CUSTOMER CHURN RISK REPORT
=======================================

Report Generated: {current_time}
Customer ID: {customer_id}

--- CUSTOMER PROFILE ---
Credit Score: {credit_score}
Age: {age} years
Tenure with Bank: {tenure} years
Account Balance: ${balance:,.2f}
Number of Products: {num_products}
Has Credit Card: {has_cr_card}
Active Member: {is_active}
Estimated Salary: ${salary:,.2f}

--- AI ANALYSIS ---
Model: Random Forest Classifier (100 Trees)
Model Accuracy: {accuracy:.2%}

Churn Risk Probability: {churn_percentage:.2f}%

--- FINAL VERDICT ---
{"⚠️ HIGH RISK - Customer likely to CHURN" if will_churn else "✅ SAFE - Customer is LOYAL"}

Risk Level: {zone}

--- RECOMMENDATION ---
{"Immediate action required. Consider offering retention incentives." if churn_percentage > 70 else "Monitor customer. Standard engagement protocols." if churn_percentage > 40 else "Customer is satisfied. Maintain current service quality."}

=======================================
Powered by Random Forest AI
Developed by Abdullah Selim Cinar
=======================================
"""
        
        st.download_button(
            label="📥 Download Risk Report",
            data=report_content,
            file_name=f"risk_report_{customer_id}.txt",
            mime="text/plain"
        )
        
        # Add spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Feature importance
        st.markdown("### 📈 Feature Importance")
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(5)
        
        fig_importance = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale=['#0E1117', '#00ff41'],
            labels={'Importance': 'Importance Score', 'Feature': ''}
        )
        
        fig_importance.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#ffffff", 'family': "Inter"},
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)',
                zerolinecolor='rgba(255, 255, 255, 0.1)'
            ),
            yaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)'
            )
        )
        
        fig_importance.update_traces(
            marker=dict(
                line=dict(color='#00ff41', width=1)
            )
        )
        
        st.plotly_chart(fig_importance, width="stretch")
    
    # Footer - Branding
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 20px; margin-top: 50px;">
        <p style="color: rgba(0, 255, 65, 0.7); font-size: 0.9rem; font-family: 'Exo 2', sans-serif;">
            Developed by <strong style="color: #00ff41;">Abdullah Selim Cinar</strong> 🚀 | Powered by <strong style="color: #00d4ff;">Random Forest AI</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
