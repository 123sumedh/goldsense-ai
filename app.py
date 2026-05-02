import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import librosa
import soundfile as sf
import pandas as pd
import plotly.graph_objects as go

# Local imports
from src.image_processing.quality_check import ImageQualityChecker
from src.image_processing.enhancement import ImageEnhancer
from src.image_processing.features import ColorFeatureExtractor
from src.computer_vision.jewelry_detector import JewelryDetector
from src.computer_vision.hallmark_ocr import HallmarkDetector
from src.computer_vision.purity_classifier import PurityClassifier
from src.audio_processing.noise_reduction import AudioDenoiser
from src.audio_processing.audio_features import AudioFeatureExtractor
from src.fusion.multimodal_fusion import MultimodalFusion
from src.fusion.decision_engine import DecisionEngine

# Page config
st.set_page_config(
    page_title="GoldSense AI",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for finance theme
st.markdown("""
<style>
    .main {
        background-color: #0A1628;
    }
    .stApp {
        background: linear-gradient(135deg, #0A1628 0%, #142847 100%);
    }
    h1, h2, h3 {
        color: #D4AF37 !important;
    }
    .stButton>button {
        background-color: #D4AF37;
        color: #0A1628;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1E3A5F;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #D4AF37;
    }
</style>
""", unsafe_allow_html=True)

# Initialize models (cached)
@st.cache_resource
def load_models():
    return {
        'quality': ImageQualityChecker(),
        'enhancer': ImageEnhancer(),
        'features': ColorFeatureExtractor(),
        'detector': JewelryDetector(),
        'hallmark': HallmarkDetector(),
        'purity': PurityClassifier(),
        'audio_denoiser': AudioDenoiser(),
        'audio_features': AudioFeatureExtractor(),
        'fusion': MultimodalFusion(),
        'decision': DecisionEngine(),
    }

models = load_models()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/gold-bars.png", width=80)
    st.markdown("# GoldSense AI")
    st.markdown("*AI-Powered Remote Gold Assessment*")
    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("""
    1. 📸 Upload jewelry photo
    2. 🎤 (Optional) Upload tap-test audio
    3. 📝 Add declared info
    4. 🤖 Get AI assessment in seconds
    """)
    st.markdown("---")
    st.markdown("### Built For")
    st.markdown("**TENZORX National AI Hackathon**")
    st.markdown("Organised by Poonawalla Fincorp")

# Main header
st.markdown("# 🪙 GoldSense AI")
st.markdown("### Remote Gold Assessment for Lending — Powered by Multimodal AI")
st.markdown("---")

# Layout: 3 columns for input
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("## 📸 Upload Inputs")
    
    uploaded_image = st.file_uploader(
        "Jewelry Photo (Required)",
        type=['png', 'jpg', 'jpeg'],
        help="Place jewelry on a flat surface with good lighting. Include a ₹10 coin for scale (optional)."
    )
    
    uploaded_audio = st.file_uploader(
        "Tap-Test Audio (Optional)",
        type=['wav', 'mp3', 'm4a'],
        help="Record yourself tapping the jewelry on a hard surface 3-5 times."
    )
    
    st.markdown("### 📝 Customer Declared Information")
    declared_weight = st.number_input("Declared Weight (grams)", min_value=0.0, value=0.0, step=0.5)
    declared_purity = st.selectbox(
        "Declared Purity",
        options=['Not specified', '24K (999)', '22K (916)', '18K (750)', '14K (585)']
    )
    has_bill = st.checkbox("Customer has purchase bill")
    
    analyze_btn = st.button("🔍 Run AI Assessment", type="primary", use_container_width=True)

with col_right:
    if uploaded_image:
        st.markdown("## 🖼️ Image Preview")
        image = Image.open(uploaded_image)
        st.image(image, use_column_width=True)
    else:
        st.markdown("## 🖼️ Image Preview")
        st.info("👈 Upload a jewelry photo to begin")

# Analysis section
if analyze_btn and uploaded_image:
    with st.spinner("🔄 Running multimodal AI pipeline..."):
        # Convert image
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        if image_np.shape[-1] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # =========================================
        # STAGE 1: Image Quality Check
        # =========================================
        st.markdown("---")
        st.markdown("## 🎯 Analysis Pipeline")
        
        with st.expander("📊 Stage 1: Image Quality Assessment", expanded=True):
            quality = models['quality'].assess(image_bgr)
            
            cols = st.columns(4)
            cols[0].metric("Quality Score", f"{quality['quality_score']}/100")
            cols[1].metric("Blur Score", f"{quality['blur_score']:.0f}")
            cols[2].metric("Brightness", f"{quality['brightness']:.0f}")
            cols[3].metric("Resolution", f"{quality['resolution'][1]}×{quality['resolution'][0]}")
            
            if quality['issues']:
                for issue in quality['issues']:
                    st.warning(f"⚠️ {issue}")
            else:
                st.success("✅ Image quality acceptable")
        
        # =========================================
        # STAGE 2: Image Enhancement
        # =========================================
        with st.expander("🎨 Stage 2: Image Enhancement Pipeline"):
            stages = models['enhancer'].enhance_pipeline(image_bgr)
            
            cols = st.columns(len(stages))
            for i, (name, img) in enumerate(stages.items()):
                with cols[i]:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                             caption=name.replace('_', ' ').title(),
                             use_column_width=True)
            
            enhanced_img = stages['final']
        
        # =========================================
        # STAGE 3: Object Detection
        # =========================================
        with st.expander("🔍 Stage 3: Jewelry Detection"):
            detection = models['detector'].detect(enhanced_img)
            
            col1, col2 = st.columns(2)
            col1.metric("Jewelry Type", detection['jewelry_type'])
            col2.metric("Detection Confidence", f"{detection['confidence']:.2%}")
        
        # =========================================
        # STAGE 4: Hallmark OCR
        # =========================================
        with st.expander("📜 Stage 4: Hallmark Detection (OCR)"):
            hallmark = models['hallmark'].detect(enhanced_img)
            
            if hallmark['detected']:
                st.success(f"✅ Hallmark detected: **{hallmark.get('declared_purity', 'Unknown')}**")
                if hallmark['bis_certified']:
                    st.success("✅ BIS certification mark found")
                
                df = pd.DataFrame(hallmark['hallmarks'])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("ℹ️ No clear hallmark detected — relying on visual analysis")
        
        # =========================================
        # STAGE 5: Color & Texture Features
        # =========================================
        with st.expander("🎨 Stage 5: Visual Feature Extraction"):
            features = models['features'].extract(enhanced_img)
            
            cols = st.columns(4)
            cols[0].metric("Hue (°)", f"{features['hue_mean']:.1f}")
            cols[1].metric("Saturation", f"{features['saturation_mean']:.0f}")
            cols[2].metric("LAB-b (yellow)", f"{features['lab_b_mean']:.0f}")
            cols[3].metric("Highlight ratio", f"{features['highlight_ratio']:.2%}")
            
            with st.container():
                st.markdown("**Texture features (GLCM):**")
                cols = st.columns(3)
                cols[0].metric("Contrast", f"{features['contrast']:.2f}")
                cols[1].metric("Homogeneity", f"{features['homogeneity']:.2f}")
                cols[2].metric("Energy", f"{features['energy']:.4f}")
        
        # =========================================
        # STAGE 6: Purity Classification
        # =========================================
        with st.expander("⚖️ Stage 6: Purity Classification"):
            purity_result = models['purity'].classify(features, hallmark)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Estimated Purity", purity_result['purity_band'])
            col2.metric("Confidence", f"{purity_result['confidence']:.1%}")
            col3.metric("Method", purity_result['method'].replace('_', ' ').title())
            
            st.info(f"**Reasoning:** {purity_result['reasoning']}")
        
        # =========================================
        # STAGE 7: Audio Analysis (if provided)
        # =========================================
        audio_result = None
        if uploaded_audio:
            with st.expander("🎵 Stage 7: Audio Tap Test Analysis", expanded=True):
                audio_data, sr = librosa.load(uploaded_audio, sr=44100)
                
                # Show waveform
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=audio_data[:min(len(audio_data), 220500)],  # First 5 sec
                    line=dict(color='#D4AF37', width=1)
                ))
                fig.update_layout(
                    title="Audio Waveform",
                    height=200,
                    plot_bgcolor='#1E3A5F',
                    paper_bgcolor='#1E3A5F',
                    font=dict(color='#FFFFFF')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Denoise
                cleaned = models['audio_denoiser'].clean_pipeline(audio_data)
                final_audio = cleaned['wavelet_denoised']
                
                # Detect taps
                taps = models['audio_features'].detect_taps(final_audio)
                st.info(f"🎯 Detected {len(taps)} tap events")
                
                # Extract features (use first tap)
                if len(taps) > 0:
                    audio_features = models['audio_features'].extract_features(taps[0])
                    audio_result = models['audio_features'].classify_audio(audio_features)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Pitch (Hz)", f"{audio_features['f0_mean']:.0f}")
                    col2.metric("Decay rate", f"{audio_features['decay_rate']:.4f}")
                    col3.metric("Brightness", f"{audio_features['brightness']:.0f}")
                    
                    if audio_result['classification'] == 'gold-like':
                        st.success(f"✅ Audio profile: **gold-like** (confidence: {audio_result['confidence']:.1%})")
                    else:
                        st.warning(f"⚠️ Audio profile: **alloy-like** (confidence: {audio_result['confidence']:.1%})")
        
        # =========================================
        # STAGE 8: Multimodal Fusion
        # =========================================
        with st.expander("🧠 Stage 8: Multimodal Fusion", expanded=True):
            fusion_result = models['fusion'].fuse(
                vision_result=purity_result,
                audio_result=audio_result,
                hallmark_result=hallmark,
                quality_score=quality['quality_score']
            )
            
            # Show modality weights as pie chart
            weights = fusion_result['modality_weights']
            fig = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                marker_colors=['#D4AF37', '#F1C40F', '#FAF5E6'],
                hole=0.4
            )])
            fig.update_layout(
                title="Modality Weights (Quality-Adjusted)",
                height=300,
                plot_bgcolor='#1E3A5F',
                paper_bgcolor='#1E3A5F',
                font=dict(color='#FFFFFF')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # =========================================
        # STAGE 9: FINAL DECISION
        # =========================================
        st.markdown("---")
        st.markdown("# 🎯 Final Assessment")
        
        # Compute estimated weight (placeholder logic)
        estimated_weight_low = max(0, declared_weight * 0.7) if declared_weight > 0 else 5
        estimated_weight_high = declared_weight * 1.1 if declared_weight > 0 else 10
        
        decision = models['decision'].decide(
            fusion_result=fusion_result,
            declared_weight=declared_weight if declared_weight > 0 else None,
            estimated_weight=(estimated_weight_low + estimated_weight_high) / 2
        )
        
        # Display decision
        cols = st.columns([2, 1, 1, 1])
        
        with cols[0]:
            color = decision['color']
            decision_emoji = {'PRE_APPROVE': '✅', 'VERIFY': '🔍', 'REJECT': '❌'}[decision['decision']]
            st.markdown(f"""
            <div style="background-color: #{('1E5631' if color=='green' else 'B8860B' if color=='orange' else '8B0000')};
                        padding: 30px; border-radius: 15px; text-align: center;">
                <h1 style="color: white; margin: 0;">{decision_emoji} {decision['decision'].replace('_', ' ')}</h1>
                <p style="color: white; font-size: 18px; margin-top: 10px;">{decision['message']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            st.metric("Estimated Purity", fusion_result['final_purity'])
        with cols[2]:
            st.metric("Estimated Weight", f"{estimated_weight_low:.1f}-{estimated_weight_high:.1f}g")
        with cols[3]:
            st.metric("Confidence", f"{decision['confidence']:.1%}")
        
        # Risk indicator gauge
        risk_fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = decision['risk_score'] * 100,
            title = {'text': "Risk Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgray"},
                'steps' : [
                    {'range': [0, 25], 'color': "#1E5631"},
                    {'range': [25, 60], 'color': "#B8860B"},
                    {'range': [60, 100], 'color': "#8B0000"}],
                'threshold' : {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': decision['risk_score'] * 100}
            }
        ))
        risk_fig.update_layout(
            height=300,
            paper_bgcolor='#1E3A5F',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(risk_fig, use_container_width=True)
        
        # Loan estimation (if pre-approved)
        if decision['decision'] == 'PRE_APPROVE':
            st.markdown("### 💰 Indicative Loan Offer")
            # Approximate gold price
            gold_price_per_gram = 7500  # ₹/gram (approximate, would use API in production)
            loan_to_value = 0.75  # 75% LTV
            avg_weight = (estimated_weight_low + estimated_weight_high) / 2
            estimated_value = avg_weight * gold_price_per_gram
            loan_amount_low = estimated_weight_low * gold_price_per_gram * loan_to_value
            loan_amount_high = estimated_weight_high * gold_price_per_gram * loan_to_value
            
            cols = st.columns(3)
            cols[0].metric("Gold Value (Est)", f"₹{estimated_value:,.0f}")
            cols[1].metric("Loan Amount (75% LTV)", f"₹{loan_amount_low:,.0f} - ₹{loan_amount_high:,.0f}")
            cols[2].metric("Time to Disbursement", "< 24 hours")
        
        # Show fraud signals if any
        if decision.get('fraud_signals'):
            st.error(f"⚠️ Fraud signals detected: {', '.join(decision['fraud_signals'])}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <p style='text-align: center; color: #94A3B8;'>
        <em>This is an early-stage assessment. Final loan decisions require human verification for items above ₹1L.</em>
        </p>
        """, unsafe_allow_html=True)

elif analyze_btn:
    st.error("⚠️ Please upload at least a jewelry image to proceed.")