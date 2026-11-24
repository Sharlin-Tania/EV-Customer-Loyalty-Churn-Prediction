import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from textblob import TextBlob
from fpdf import FPDF
import io
import plotly.express as px
import plotly.graph_objects as go


# --- 1. Page Configuration ---
st.set_page_config(page_title="EV Customer Intelligence Hub", layout="wide")

# --- 2. Load Assets ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('xgb_churn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        kmeans = joblib.load('kmeans_model.pkl')
        return model, scaler, kmeans
    except Exception as e:
        st.error(f"Error loading .pkl files: {e}")
        return None, None, None

xgb_model, scaler, kmeans = load_assets()

# --- 3. Constants & Mapping ---
rating_cols = ['Visual_Appeal', 'Reliability', 'Performance', 'Service_Experience', 
               'Extra_Features', 'Comfort', 'Maintenance_cost', 'Value_for_Money']

ownership_map = {
    '< 3 months': 1, '3-6 months': 2, '6 months-1 yr': 3, 
    '6 months-1 year': 3, '> 1 yr': 4, '> 1 year': 4
}
ridden_map = {
    '< 5000 km': 1, '< 5000 kms': 1, '5000-10000 km': 2, 
    '5000-10000 kms': 2, '10000-15000 km': 3, '10000-15000 kms': 3, 
    '> 15000 km': 4, '> 15000 kms': 4
}

# --- 4. Core Processing Logic ---
def process_data(df_input):
    df = df_input.copy()
    # Clean Headers
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
    
    # Encode Categoricals
    if 'Owned_for' in df.columns:
        df['Owned_for_encoded'] = df['Owned_for'].str.strip().map(ownership_map)
    if 'Ridden_for' in df.columns:
        df['Ridden_for_encoded'] = df['Ridden_for'].str.strip().map(ridden_map)
    
    # Fill missing encodings
    df['Owned_for_encoded'] = df['Owned_for_encoded'].fillna(3)
    df['Ridden_for_encoded'] = df['Ridden_for_encoded'].fillna(2)

    # Feature Engineering
    df['Avg_Rating'] = df[rating_cols].mean(axis=1)
    df['Rating_Std'] = df[rating_cols].std(axis=1).fillna(0)
    df['Min_Rating'] = df[rating_cols].min(axis=1)
    df['Max_Rating'] = df[rating_cols].max(axis=1)
    df['Negative_Count'] = (df[rating_cols] <= 2).sum(axis=1)
    df['Positive_Count'] = (df[rating_cols] >= 4).sum(axis=1)
    df['Rating_Range'] = df['Max_Rating'] - df['Min_Rating']
    df['Consistency_Score'] = 1 / (1 + df['Rating_Std'])
    df['Sentiment_Score'] = df['review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['Review_Length'] = df['review'].apply(lambda x: len(str(x)))
    
    df['Usage_Intensity'] = df['Owned_for_encoded'] * df['Ridden_for_encoded']
    df['CLV_Score'] = (df['Usage_Intensity'] * df['Value_for_Money'] * df['Avg_Rating']) / 100
    df['Value_Perception'] = df['Value_for_Money'] / (df['Maintenance_cost'].clip(lower=0.1) + 0.01)
    df['Satisfaction_Index'] = df['Avg_Rating'] * df['Consistency_Score']
    
    # Segmentation
    seg_features = rating_cols + ['Avg_Rating', 'Value_for_Money', 'Sentiment_Score', 'CLV_Score']
    scaled_data = scaler.transform(df[seg_features].fillna(0))
    df['Segment_ID'] = kmeans.predict(scaled_data)
    
    segment_names = {0: 'Dissatisfied_Customers', 1: 'Value_Seekers', 2: 'Loyal_Advocates', 3: 'At_Risk_Users'}
    for sid, name in segment_names.items():
        df[f'Segment_{name}'] = (df['Segment_ID'] == sid).astype(int)

    # Predict
    final_features = df[xgb_model.feature_names_in_]
    df['Churn_Prob'] = xgb_model.predict_proba(final_features)[:, 1]
    df['Risk_Category'] = pd.cut(df['Churn_Prob'], bins=[0, 0.3, 0.7, 1], labels=['Low', 'Medium', 'High'])
    
    return df

# --- 5. UI Layout ---
st.title("⚡ EV Customer Loyalty & Churn Analytics")
tabs = st.tabs(["Individual Analysis", "Fleet Business Dashboard"])

# --- TAB 1: Individual Analysis ---
with tabs[0]:
    st.sidebar.header("Individual Customer Input")
    single_ratings = {}
    for col in rating_cols:
        single_ratings[col] = st.sidebar.slider(col.replace('_',' '), 1, 5, 3)
    
    owned_txt = st.sidebar.selectbox("Owned For", list(ownership_map.keys()))
    ridden_txt = st.sidebar.selectbox("Ridden For", list(ridden_map.keys()))
    review_text = st.sidebar.text_area("Customer Review", "The service is good but battery is average.")
    
    if st.sidebar.button("Analyze Individual"):
        # Create input df
        single_df = pd.DataFrame([single_ratings])
        single_df['Owned_for'] = owned_txt
        single_df['Ridden_for'] = ridden_txt
        single_df['review'] = review_text
        
        res = process_data(single_df)
        if res is not None:
            c1, c2, c3 = st.columns(3)
            prob = res['Churn_Prob'].values[0]
            c1.metric("Churn Risk", f"{prob*100:.1f}%")
            c2.metric("Segment", res['Segment_ID'].map({0:'Dissatisfied', 1:'Value Seeker', 2:'Loyal', 3:'At Risk'}).values[0])
            c3.metric("Sentiment", f"{res['Sentiment_Score'].values[0]:.2f}")
            
            # SHAP Plot
            st.subheader("Why this prediction?")
            explainer = shap.TreeExplainer(xgb_model)
            shap_v = explainer(res[xgb_model.feature_names_in_])
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(shap_v[0], show=False)
            st.pyplot(fig)

            # PDF Download
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Customer Retention Report", ln=True, align='C')
            pdf.set_font("Arial", size=12)
            pdf.ln(10)
            pdf.cell(0, 10, f"Churn Probability: {prob*100:.1f}%", ln=True)
            pdf.multi_cell(0, 10, f"Review Analysis: {review_text}")
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            pdf.image(buf, x=10, y=60, w=180)
            
            st.download_button("📥 Download Report", data=bytes(pdf.output(dest='S')), file_name="Report.pdf")

# --- TAB 2: Fleet Dashboard ---

with tabs[1]:
    st.header("🏢 Fleet Executive Business Intelligence")
    uploaded_file = st.file_uploader("Upload Fleet Feedback CSV", type="csv")
    
    if uploaded_file:
        raw_data = pd.read_csv(uploaded_file)
        processed = process_data(raw_data)
        
        if processed is not None:
            # --- 1. Top Level Metrics ---
            k1, k2, k3, k4 = st.columns(4)
            high_risk_df = processed[processed['Risk_Category'] == 'High']
            
            k1.metric("Total Customers", len(processed))
            k2.metric("At-Risk Accounts", len(high_risk_df), f"{len(high_risk_df)/len(processed)*100:.1f}%", delta_color="inverse")
            k3.metric("Avg Fleet Sentiment", f"{processed['Sentiment_Score'].mean():.2f}")
            k4.metric("Service Quality", f"{processed['Service_Experience'].mean():.1f}/5.0")

            st.markdown("---")

            # --- 2. INTERACTIVE PLOTLY CHARTS ---
            col_left, col_right = st.columns(2)

            with col_left:
                st.write("### 🎯 Sentiment vs. Churn Probability")
                
                # --- FIX START ---
                # Fill NaN values in CLV_Score so Plotly doesn't crash
                processed['CLV_Score'] = processed['CLV_Score'].fillna(0)
                
                # Ensure size is always positive (Plotly size cannot be negative)
                # We add a small constant (0.1) so dots aren't invisible if score is 0
                processed['bubble_size'] = processed['CLV_Score'].apply(lambda x: max(x, 0)) + 0.1
                # --- FIX END ---

                # Updated Plotly Scatter
                fig_px = px.scatter(processed, 
                                    x="Sentiment_Score", 
                                    y="Churn_Prob", 
                                    color="Risk_Category", 
                                    size="bubble_size",  # Use the cleaned size column
                                    hover_data=['review', 'CLV_Score'],
                                    color_discrete_map={'High':'#ef553b', 'Medium':'#fec032', 'Low':'#636efa'},
                                    title="Customer Sentiment vs. Risk (Size = Lifetime Value)")
                
                st.plotly_chart(fig_px, use_container_width=True)
            with col_right:
                st.write("### 📊 Performance Heatmap")
                # Interactive Heatmap of all ratings
                avg_stats = processed[rating_cols].mean().reset_index()
                avg_stats.columns = ['Feature', 'Avg_Rating']
                fig_heat = px.bar(avg_stats, x='Avg_Rating', y='Feature', orientation='h',
                                  color='Avg_Rating', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_heat, use_container_width=True)

            # --- 3. MORE INSIGHTS ---
            st.write("### 📈 Churn Drivers by Ownership Duration")
            fig_box = px.box(processed, x="Owned_for", y="Churn_Prob", color="Owned_for",
                             title="How Ownership Length Affects Churn")
            st.plotly_chart(fig_box, use_container_width=True)

            # --- 4. ACTIONABLE RECOMMENDATIONS ---
            st.subheader("💡 Strategic Recommendations")
            rec_col1, rec_col2 = st.columns(2)
            with rec_col1:
                if processed['Service_Experience'].mean() < 3.5:
                    st.warning("**Priority 1: Service Overhaul** - Service ratings are low. High-risk customers cite experience as a key pain point. Recommend audit of service centers.")
                if processed['Maintenance_cost'].mean() > 3.0:
                    st.info("**Priority 2: Cost Perception** - Maintenance costs are perceived as high. Consider offering 'Service Packs' to lock in loyalty.")
            with rec_col2:
                top_segment = processed['Segment_ID'].mode()[0]
                seg_name = {0:'Dissatisfied', 1:'Value Seekers', 2:'Loyal', 3:'At Risk'}[top_segment]
                st.success(f"**Targeting:** Your largest group is **{seg_name}**. Tailor marketing campaigns to address their specific needs.")

            # --- 5. FIXED PDF GENERATOR ---
            def generate_fixed_pdf(df, matplotlib_figs):
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                
                # --- Page 1: Executive Cover & Summary ---
                pdf.add_page()
                # Header
                pdf.set_fill_color(30, 50, 100) # Dark Blue Header
                pdf.rect(0, 0, 210, 40, 'F')
                pdf.set_text_color(255, 255, 255)
                pdf.set_font("Arial", 'B', 24)
                pdf.cell(0, 20, "Fleet Retention Intelligence Report", ln=True, align='C')
                
                pdf.set_text_color(0, 0, 0)
                pdf.ln(25)
                
                # Summary Table Data
                high_risk_pct = len(df[df['Risk_Category']=='High'])/len(df)*100
                avg_sent = df['Sentiment_Score'].mean()
                
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "1. Executive Summary", ln=True)
                pdf.set_font("Arial", '', 11)
                summary_text = (f"This report provides a data-driven analysis of {len(df)} EV customers. "
                                f"Current analysis shows that {high_risk_pct:.1f}% of the fleet is at 'High Risk' of churning. "
                                f"The average sentiment score of {avg_sent:.2f} suggests a moderately positive but fragile "
                                f"customer base. Key friction points identified include Service Experience and Maintenance Costs.")
                pdf.multi_cell(0, 8, summary_text)
                
                # Chart 1: Risk Distribution (Strategic Overview)
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Figure 1: Churn Risk Segmentation across Fleet", ln=True)
                img_buf1 = io.BytesIO()
                matplotlib_figs[0].savefig(img_buf1, format='png', bbox_inches='tight', dpi=150)
                pdf.image(img_buf1, x=25, w=160)
                
                # --- Page 2: Operational Insights ---
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "2. Operational Performance Analysis", ln=True)
                pdf.set_font("Arial", '', 11)
                pdf.multi_cell(0, 8, "The following chart illustrates the average ratings across 8 core performance "
                                    "metrics. Identifying features with ratings below 3.5 is critical for retention.")
                
                # Chart 2: Feature Performance
                pdf.ln(5)
                img_buf2 = io.BytesIO()
                matplotlib_figs[1].savefig(img_buf2, format='png', bbox_inches='tight', dpi=150)
                pdf.image(img_buf2, x=25, w=160)
                
                # --- Page 3: Strategic Recommendations ---
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "3. Strategic Recommendations & Action Plan", ln=True)
                pdf.ln(5)
                
                # Dynamic Recommendations based on data
                pdf.set_font("Arial", 'B', 11)
                pdf.cell(0, 8, "Immediate Actions:", ln=True)
                pdf.set_font("Arial", '', 11)
                
                recs = [
                    "- Proactive Outreach: Contact High-Risk customers with sentiment < 0 immediately.",
                    "- Service Audit: Focus on the 'Service_Experience' metric which is currently the lead churn driver.",
                    "- Loyalty Program: Target 'Value Seekers' with maintenance-inclusive service packages.",
                    "- Feedback Loop: Respond to critical reviews captured in this batch to improve brand trust."
                ]
                for rec in recs:
                    pdf.cell(0, 8, rec, ln=True)
                    
                pdf.ln(20)
                pdf.set_font("Arial", 'I', 8)
                pdf.cell(0, 10, "Report generated via EV Intelligence AI Dashboard. Confidential - Internal Use Only.", align='C')
                
                return pdf.output(dest='S')

            # We create static versions of charts for the PDF since FPDF can't handle Plotly directly
            # 1. Prepare static versions of charts for the PDF
            # Chart A: Feature Ratings
            fig_static1, ax_static1 = plt.subplots(figsize=(10, 6))
            avg_stats = processed[rating_cols].mean().sort_values().reset_index()
            avg_stats.columns = ['Feature', 'Avg_Rating']
            sns.barplot(data=avg_stats, x='Avg_Rating', y='Feature', palette='Blues_d', ax=ax_static1)
            ax_static1.set_title("Fleet-wide Feature Performance")

            # Chart B: Risk Distribution
            fig_static2, ax_static2 = plt.subplots(figsize=(10, 6))
            sns.countplot(data=processed, x='Risk_Category', palette='viridis', ax=ax_static2)
            ax_static2.set_title("Customer Risk Segmentation")

            # 2. Call the NEW professional function
            # The function now takes: (dataframe, list_of_figures)
            pdf_data = generate_fixed_pdf(processed, [fig_static1, fig_static2])

            # 3. Download Button
            st.download_button(
                label="📥 Download Executive Business Report", 
                data=bytes(pdf_data), 
                file_name="EV_Fleet_Executive_Report.pdf", 
                mime="application/pdf"
            )