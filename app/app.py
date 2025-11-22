"""
Vehicle Price Prediction Dashboard
Production-ready Streamlit application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from predictor import VehiclePricePredictor
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    .sample-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache data loading
@st.cache_resource
def load_artifacts():
    """Load all necessary artifacts."""
    try:
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        artifacts_dir = os.path.join(current_dir, 'artifacts')
        
        # Load predictor
        predictor = VehiclePricePredictor(
            model_path=os.path.join(artifacts_dir, 'xgboost_optimized.pkl'),
            encoders_path=os.path.join(artifacts_dir, 'label_encoders.pkl'),
            metadata_path=os.path.join(artifacts_dir, 'model_metadata.pkl')
        )
        
        # Load lookup tables
        with open(os.path.join(artifacts_dir, 'lookup_tables.pkl'), 'rb') as f:
            lookups = pickle.load(f)
        
        # Load dashboard data
        with open(os.path.join(artifacts_dir, 'dashboard_data.pkl'), 'rb') as f:
            dashboard_data = pickle.load(f)
        
        # Load samples
        with open(os.path.join(artifacts_dir, 'sample_predictions.pkl'), 'rb') as f:
            samples = pickle.load(f)
        
        return predictor, lookups, dashboard_data, samples
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.stop()

# Load artifacts
predictor, lookups, dashboard_data, samples = load_artifacts()

# Sidebar navigation
st.sidebar.title("🚗 Vehicle Price Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🎯 Price Prediction", "📊 Model Insights", "🔍 Market Analysis", "📈 Sample Predictions", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
st.sidebar.metric("Test MAE", f"${dashboard_data['overall_metrics']['mae']:,.0f}")
st.sidebar.metric("Test R²", f"{dashboard_data['overall_metrics']['r2']:.4f}")
st.sidebar.metric("Test MAPE", f"{dashboard_data['overall_metrics']['mape']:.2f}%")

# ==================== PAGE 1: PRICE PREDICTION ====================
if page == "🎯 Price Prediction":
    st.title("🎯 Vehicle Price Prediction")
    st.markdown("Enter vehicle details to get an instant price prediction powered by XGBoost AI.")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vehicle Information")
        
        year = st.number_input(
            "Year",
            min_value=1980,
            max_value=2025,
            value=2015,
            help="Manufacturing year"
        )
        
        make = st.selectbox(
            "Make",
            options=sorted(lookups['categorical_options']['make']),
            help="Vehicle manufacturer"
        )
        
        model = st.selectbox(
            "Model",
            options=sorted(lookups['categorical_options']['model_grouped']),
            help="Vehicle model"
        )
        
        body = st.selectbox(
            "Body Type",
            options=sorted(lookups['categorical_options']['body']),
            help="Vehicle body style"
        )
        
        transmission = st.selectbox(
            "Transmission",
            options=sorted(lookups['categorical_options']['transmission']),
            help="Transmission type"
        )
    
    with col2:
        st.subheader("Condition & Location")
        
        odometer = st.number_input(
            "Odometer (miles)",
            min_value=0,
            max_value=500000,
            value=50000,
            step=1000,
            help="Total miles driven"
        )
        
        condition = st.slider(
            "Condition Score",
            min_value=1.0,
            max_value=49.0,
            value=35.0,
            step=1.0,
            help="1-49 scale (higher is better)"
        )
        
        state = st.selectbox(
            "State",
            options=sorted(lookups['categorical_options']['state']),
            help="Location state"
        )
        
        color = st.selectbox(
            "Exterior Color",
            options=sorted(lookups['categorical_options']['color'])
        )
        
        interior = st.selectbox(
            "Interior Color",
            options=sorted(lookups['categorical_options']['interior'])
        )
    
    # Additional inputs
    st.subheader("Market Information")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        mmr = st.number_input(
            "MMR (Manheim Market Report)",
            min_value=0.0,
            max_value=200000.0,
            value=12000.0,
            step=100.0,
            help="Market reference price"
        )
    
    with col4:
        seller = st.selectbox(
            "Seller Type",
            options=sorted(lookups['categorical_options']['seller_grouped'])
        )
    
    with col5:
        trim = st.selectbox(
            "Trim Level",
            options=sorted(lookups['categorical_options']['trim_grouped'])
        )
    
    # Predict button
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        
        # Prepare input
        vehicle_input = {
            'year': int(year),
            'make': make,
            'model_grouped': model,
            'body': body,
            'transmission': transmission,
            'odometer': int(odometer),
            'condition': float(condition),
            'state': state,
            'color': color,
            'interior': interior,
            'seller_grouped': seller,
            'trim_grouped': trim,
            'mmr': float(mmr)
        }
        
        # Make prediction
        with st.spinner("Analyzing vehicle..."):
            result = predictor.predict(vehicle_input)
        
        if result['success']:
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h1 style='margin:0; font-size: 3em;'>${result['predicted_price']:,.0f}</h1>
                <p style='margin:10px 0 0 0; font-size: 1.2em;'>Predicted Selling Price</p>
                <p style='margin:5px 0 0 0; opacity: 0.9;'>Confidence: {result['confidence']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparison with MMR
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Price",
                    f"${result['predicted_price']:,.0f}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "MMR (Market Value)",
                    f"${mmr:,.0f}",
                    delta=None
                )
            
            with col3:
                diff = result['predicted_price'] - mmr
                st.metric(
                    "Difference",
                    f"${abs(diff):,.0f}",
                    delta=f"${diff:,.0f}",
                    delta_color="normal"
                )
            
            # Additional insights
            st.markdown("### 📊 Prediction Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Vehicle Summary:**
                - {year} {make} {model}
                - {odometer:,} miles
                - Condition: {condition}/49
                - Location: {state}
                """)
            
            with col2:
                if diff > 0:
                    st.success(f"✅ This vehicle may sell **above** market value by ${diff:,.0f}")
                elif diff < 0:
                    st.warning(f"⚠️ This vehicle may sell **below** market value by ${abs(diff):,.0f}")
                else:
                    st.info("ℹ️ This vehicle is priced at market value")
        
        else:
            st.error("Prediction failed. Please check your inputs.")
            for error in result.get('errors', []):
                st.error(f"- {error}")

# ==================== PAGE 2: MODEL INSIGHTS ====================
elif page == "📊 Model Insights":
    st.title("📊 Model Insights & Performance")
    
    # Overall metrics
    st.header("Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Mean Absolute Error",
            f"${dashboard_data['overall_metrics']['mae']:,.0f}",
            help="Average prediction error"
        )
    
    with col2:
        st.metric(
            "R² Score",
            f"{dashboard_data['overall_metrics']['r2']:.4f}",
            help="Variance explained (closer to 1 is better)"
        )
    
    with col3:
        st.metric(
            "MAPE",
            f"{dashboard_data['overall_metrics']['mape']:.2f}%",
            help="Mean Absolute Percentage Error"
        )
    
    with col4:
        st.metric(
            "Total Predictions",
            f"{dashboard_data['overall_metrics']['total_predictions']:,}",
            help="Training dataset size"
        )
    
    st.markdown("---")
    
    # Feature importance
    st.header("Feature Importance")
    st.markdown("Understanding which factors drive vehicle prices")
    
    feature_imp = pd.DataFrame(dashboard_data['feature_importance']).head(10)
    
    fig = px.bar(
        feature_imp,
        x='importance',
        y='feature',
        orientation='h',
        title="Top 10 Most Important Features",
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500, showlegend=False)
    fig.update_yaxes(categoryorder='total ascending')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.info(f"""
    **Key Insights:**
    - **{feature_imp.iloc[0]['feature']}** is the most important feature ({feature_imp.iloc[0]['importance']*100:.1f}% importance)
    - Top 3 features account for {feature_imp.head(3)['importance'].sum()*100:.1f}% of prediction power
    - Model uses 17 features total for predictions
    """)

# ==================== PAGE 3: MARKET ANALYSIS ====================
elif page == "🔍 Market Analysis":
    st.title("🔍 Market Analysis")
    
    tab1, tab2, tab3 = st.tabs(["By State", "By Make", "By Body Type"])
    
    with tab1:
        st.subheader("Performance by State")
        
        state_df = pd.DataFrame(dashboard_data['state_performance']).T.reset_index()
        state_df.columns = ['State', 'MAE', 'MAPE', 'Count', 'Avg_Price']
        state_df = state_df.sort_values('Count', ascending=False).head(15)
        
        fig = px.bar(
            state_df,
            x='State',
            y='MAPE',
            title="Model Accuracy by State (Top 15)",
            labels={'MAPE': 'MAPE (%)', 'State': 'State'},
            color='MAPE',
            color_continuous_scale='RdYlGn_r',
            hover_data=['MAE', 'Count']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(state_df, use_container_width=True)
    
    with tab2:
        st.subheader("Performance by Make")
        
        make_df = pd.DataFrame(dashboard_data['make_performance']).T.reset_index()
        make_df.columns = ['Make', 'MAE', 'MAPE', 'Count', 'Avg_Price']
        make_df = make_df.sort_values('Count', ascending=False).head(15)
        
        fig = px.scatter(
            make_df,
            x='Avg_Price',
            y='MAPE',
            size='Count',
            color='Make',
            title="Accuracy vs Average Price by Make",
            labels={'Avg_Price': 'Average Price ($)', 'MAPE': 'MAPE (%)'},
            hover_data=['MAE']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(make_df, use_container_width=True)
    
    with tab3:
        st.subheader("Performance by Body Type")
        
        body_df = pd.DataFrame(dashboard_data['body_performance']).T.reset_index()
        body_df.columns = ['Body', 'MAE', 'MAPE', 'Count', 'Avg_Price']
        body_df = body_df.sort_values('Count', ascending=False).head(10)
        
        fig = px.bar(
            body_df,
            x='Body',
            y='Count',
            title="Vehicle Count by Body Type",
            labels={'Count': 'Number of Vehicles', 'Body': 'Body Type'},
            color='Avg_Price',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(body_df, use_container_width=True)

# ==================== PAGE 4: SAMPLE PREDICTIONS ====================
elif page == "📈 Sample Predictions":
    st.title("📈 Sample Predictions Showcase")
    st.markdown("See how the model performs across different vehicle types")
    
    for sample in samples:
        st.markdown(f"""
        <div class="sample-card">
            <h3>{sample['category']}</h3>
            <p><strong>{sample['year']} {sample['make']} {sample['model']} ({sample['body']})</strong></p>
            <p>📍 {sample['state']} | 🛣️ {sample['odometer']:,} miles | ⭐ Condition: {sample['condition']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Actual Price", f"${sample['actual_price']:,.0f}")
        
        with col2:
            st.metric("Predicted Price", f"${sample['predicted_price']:,.0f}")
        
        with col3:
            st.metric("Error", f"${abs(sample['error']):,.0f}")
        
        with col4:
            st.metric("Error %", f"{abs(sample['error_pct']):.1f}%")
        
        st.markdown("---")

# ==================== PAGE 5: ABOUT ====================
else:
    st.title("ℹ️ About This Dashboard")
    
    st.markdown("""
    ## Vehicle Price Prediction System
    
    This production-ready dashboard uses **XGBoost machine learning** to predict vehicle selling prices 
    with high accuracy based on multiple factors.
    
    ### Model Information
    
    - **Algorithm:** XGBoost Regressor (Optimized with Optuna)
    - **Training Data:** 558,825 vehicle sales records (2014-2015)
    - **Features:** 17 features (8 numeric, 9 categorical)
    - **Performance:** 96.82% R², $887 MAE, 12.30% MAPE
    
    ### Key Features Used
    
    1. **MMR (61.7% importance)** - Manheim Market Report value
    2. **Body Type (9.0%)** - Vehicle body style
    3. **Age-Odometer Interaction (8.5%)** - Wear indicator
    4. **Make (5.9%)** - Vehicle manufacturer
    5. **Model (3.0%)** - Specific model
    
    ### How It Works
    
    1. Enter vehicle details in the prediction form
    2. Model analyzes 17 different features
    3. Advanced XGBoost algorithm computes price
    4. Instant prediction with confidence score
    
    ### Accuracy by Segment
    
    - **Best Range:** $30k-$50k (3.19% MAPE)
    - **Best State:** Pennsylvania (7.05% MAPE)
    - **Best Make:** BMW (7.61% MAPE)
    - **Best Age:** 2-5 years old (5.85% MAPE)
    
    ### Technology Stack
    
    - **Frontend:** Streamlit
    - **ML Framework:** XGBoost, scikit-learn
    - **Optimization:** Optuna (50 trials)
    - **Monitoring:** Custom drift detection system
    - **Visualization:** Plotly, Matplotlib
    
    ### Project Structure
    
    This system was built through 5 comprehensive notebooks:
    1. Data Overview + EDA
    2. Data Cleaning + Preprocessing
    3. Modeling + Hyperparameter Tuning
    4. Explainability + Fairness Analysis
    5. Monitoring + Drift Detection
    
    ### Contact & Support
    
    Built with ❤️ for production ML deployment
    
    ---
    
    **Version:** 1.0.0 | **Last Updated:** 2025
    """)
    
    st.success("🎉 Ready for production deployment!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Vehicle Price Prediction Dashboard | Powered by XGBoost AI</div>",
    unsafe_allow_html=True
)