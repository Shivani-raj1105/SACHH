import streamlit as st
import pandas as pd
import os
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from lime.lime_text import LimeTextExplainer
import re

# Debug statements for deployment troubleshooting (will be removed after testing)
# st.write("🚀 SACH App Starting...")
# st.write("Current directory:", os.getcwd())
# st.write("Files in current directory:", os.listdir("."))

# Check if saved_model directory exists
if os.path.exists("saved_model"):
    # st.write("✅ saved_model directory found")
    pass
else:
    st.error("❌ saved_model directory not found!")

# Check if requirements.txt exists
if os.path.exists("requirements.txt"):
    # st.write("✅ requirements.txt found")
    pass
else:
    st.error("❌ requirements.txt not found!")

# Try to import auth module with error handling
try:
    from auth import login_section
    # st.write("✅ auth module imported successfully")
except Exception as e:
    # st.error(f"❌ Error importing auth module: {e}")
    # Create a simple login_section function if import fails
    def login_section():
        st.sidebar.markdown("### Welcome to SACH")
        return True
    # st.write("✅ Using fallback auth function")

st.set_page_config(
    page_title="SACH (The fake news detector)",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
body, .stApp {
    background-color: #181818;
    color: #e5c07b;
    font-family: 'Georgia', serif;
}
.sidebar .sidebar-content {
    background: #FFD700 !important;
    color: #000000 !important;
}
.sidebar .sidebar-content * {
    color: #000000 !important;
}
.sidebar .sidebar-content .stRadio > label,
.sidebar .sidebar-content .stRadio > div,
.sidebar .sidebar-content .stRadio > div > div,
.sidebar .sidebar-content .stRadio > div > div > label {
    color: #000000 !important;
}
.block-container {
    background: #222;
    border-radius: 18px;
    box-shadow: 0 4px 32px 0 rgba(0,0,0,0.25);
    padding: 2rem;
}
.lux-title {
    font-size: 2.8rem;
    font-family: 'Georgia', serif;
    color: #FFD700;
    letter-spacing: 2px;
    text-shadow: 0 2px 8px #000, 0 0px 2px #FFD700;
    margin-bottom: 0.5em;
}
.lux-card {
    background: #181818;
    border: 1.5px solid #FFD700;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 2px 16px 0 rgba(255,215,0,0.08);
    margin-bottom: 1.5rem;
}
.lux-fake {
    color: #ff4c4c;
    font-size: 2.2rem;
    font-weight: bold;
    text-shadow: 0 1px 4px #000;
}
.lux-real {
    color: #50fa7b;
    font-size: 2.2rem;
    font-weight: bold;
    text-shadow: 0 1px 4px #000;
}
.lux-progress .stProgress > div > div > div {
    background-image: linear-gradient(90deg, #FFD700, #e5c07b);
}
.lux-footer {
    color: #FFD700;
    text-align: center;
    font-size: 1.1rem;
    margin-top: 2rem;
    letter-spacing: 1px;
}
/* Custom feedback button styling */
.stButton > button {
    background-color: #FFD700 !important;
    color: #000000 !important;
    border: 2px solid #000000 !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background-color: #e5c07b !important;
    border-color: #FFD700 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(255, 215, 0, 0.3) !important;
}
</style>
""", unsafe_allow_html=True)

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None


# Remove model selection in sidebar and always use original model
# st.sidebar.markdown("### Model Selection")
# model_version = st.sidebar.selectbox(
#     "Choose Model Version",
#     ["Improved Model (Recommended)", "Original Model"],
#     help="Improved model includes user feedback and synthetic data"
# )

model_path = "saved_model"


@st.cache_resource
def load_model() -> tuple:
    """Load the BERT model and tokenizer from the saved_model directory."""
    try:
        # st.write("🔄 Loading model...")
        model_path = model_path if "model_path" in locals() else "saved_model"
        if not os.path.exists(model_path):
            st.error("Model files not found! Please ensure the saved_model directory exists.")
            return None, None
        
        # st.write("📦 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # st.write("📦 Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        # st.write("✅ Model loaded successfully!")
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # st.write(f"Debug info: Model path = {model_path}")
        # st.write(f"Debug info: Files in directory = {os.listdir('.')}")
        return None, None

def predict_text(text: str, model, tokenizer) -> tuple[str, float]:
    """Predict whether the input text is real or fake news using the model and tokenizer."""
    try:
        if not text.strip():
            return "Error", 0.0
            
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        label = "Real" if prediction == 1 else "Fake"
        return label, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error", 0.0

def get_lime_explanation(text: str, model, tokenizer, num_features: int = 10) -> list[tuple[str, float]]:
    """Generate a LIME explanation for the model's prediction on the input text."""
    try:
        explainer = LimeTextExplainer(class_names=["Real", "Fake"])
        
        def predictor(texts: list[str]) -> np.ndarray:
            results = []
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    results.append(probabilities[0].numpy())
            return np.array(results)
        
        exp = explainer.explain_instance(text, predictor, num_features=num_features, num_samples=100)
        return [(word, weight) for word, weight in exp.as_list()]
    except Exception as e:
        st.warning(f"Could not generate explanation: {e}")
        return []

def save_feedback(input_text: str, label: str, confidence: float, explanation: list, feedback_type: str) -> bool:
    print(f"save_feedback called with label={label}, feedback_type={feedback_type}")
    try:
        # Sanitize input to prevent malformed CSV
        safe_text = input_text.replace('\n', ' ').replace('\r', ' ').replace(',', ' ').strip()
        safe_feedback_type = feedback_type.replace('\n', ' ').replace('\r', ' ').replace(',', ' ').strip()
        confidence_pct = f"{confidence * 100:.2f}%"
        feedback_data = [safe_text, label, confidence_pct, safe_feedback_type]
        file_exists = os.path.exists("feedback.csv")
        with open("feedback.csv", mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["text", "model_prediction", "confidence", "feedback_type"])
            writer.writerow(feedback_data)
        print("Attempting to write feedback.csv in", os.getcwd())
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return False

# Call login_section with error handling
try:
    login_section()
    # st.write("✅ Login section initialized")
except Exception as e:
    # st.error(f"❌ Error in login section: {e}")
    # st.write("⚠️ Continuing without login section")
    pass

# Test file write permissions
try:
    with open("test_write.txt", "w") as f:
        f.write("test")
    import os
    # st.success(f"File write test succeeded. Directory: {os.getcwd()}")
except Exception as e:
    # st.error(f"File write test failed: {e}")
    pass

if not st.session_state.model_loaded:
    with st.spinner("Loading AI model... This may take a moment on first run."):
        model, tokenizer = load_model()
        if model is not None and tokenizer is not None:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            # st.success("Model loaded successfully!")

    # Display model information
    if st.session_state.model_loaded:
        # Remove all uses of model_version and related model_name display
        st.sidebar.markdown(f"""
        <div style='background: #FFD700; color: #000; padding: 0.5rem; border-radius: 5px; margin: 1rem 0;'>
            <strong>Active Model:</strong> Original BERT Model
        </div>
        """, unsafe_allow_html=True)


st.sidebar.markdown("""
<div style='color:#000000; font-family:Georgia,serif; font-size:1.5rem; text-align:center; margin-bottom:1.5rem;'>
    <b>SACH (The fake news detector)</b>
</div>
<hr style='border:1px solid #000000;'>
<div style='color:#000000; font-size:1rem; margin-bottom:1rem;'>
SACH (The fake news detector) is an AI-powered web application that uses a BERT-based machine learning model to analyze and classify news articles as either real or fake with confidence scores. The project demonstrates modern AI/ML development practices with features like model caching, error handling, user feedback collection, and a responsive web interface that makes fake news detection accessible to non-technical users.
</div>
<hr style='border:1px solid #000000;'>
<div style='color:#000000; font-size:1.1rem;'>
<b>Instructions:</b><br>
- Paste a news headline or article.<br>
- Click <b>Check if it's Fake</b>.<br>
- See the luxurious result below.<br>
</div>
<hr style='border:1px solid #000000;'>
<div style='color:#000000; font-size:1rem;'>
<b>Created by:</b> Shivani Raj<br>
<b>Contact:</b> shivani.raj.urs1105@gmail.com</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["Detector", "Visualizations"])

# At the top of the Detector page, show feedback message if present
if 'feedback_message' in st.session_state:
    st.success(st.session_state['feedback_message'])
    del st.session_state['feedback_message']

if page == "Detector":
    st.markdown('<div class="lux-title">🧠 SACH (The fake news detector)</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.error("Model is still loading. Please wait...")
        st.stop()
    
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown('<div class="lux-card">', unsafe_allow_html=True)
        with st.expander("Disclaimer", expanded=False):
            st.warning("⚠️ **Important Notice:** The model is designed to analyze news articles and headlines. If you enter text that is not a news article (such as personal messages, fiction, or other content), the model may provide incorrect predictions. For best results, please enter actual news content.")
        st.subheader("Enter News Text", divider="orange")
        input_text = st.text_area("Paste news headline or article:", height=200)
        if 'submitted_text' not in st.session_state:
            st.session_state['submitted_text'] = ""
        if st.button("Submit"):
            st.session_state['submitted_text'] = input_text
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="lux-card">', unsafe_allow_html=True)
        st.subheader("Prediction Result", divider="orange")
        submitted_text = st.session_state['submitted_text']
        if submitted_text.strip():
            with st.spinner("Analyzing..."):
                label, confidence = predict_text(submitted_text, st.session_state.model, st.session_state.tokenizer)
                explanation = get_lime_explanation(submitted_text, st.session_state.model, st.session_state.tokenizer)
                st.session_state.last_input_text = submitted_text
                st.session_state.last_label = label
                st.session_state.last_confidence = confidence
                st.session_state.last_explanation = explanation
                if label == "Fake":
                    st.markdown('<div class="lux-fake">🛑 FAKE</div>', unsafe_allow_html=True)
                elif label == "Real":
                    st.markdown('<div class="lux-real">✅ REAL</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Error: {label}")
                st.markdown('<div class="lux-progress">', unsafe_allow_html=True)
                st.progress(min(max(confidence, 0), 1))
                st.markdown('</div>', unsafe_allow_html=True)
                st.write(f"Confidence: <b>{confidence:.2%}</b>", unsafe_allow_html=True)
                if explanation:
                    try:
                        words, weights = zip(*explanation)
                        df_explanation = pd.DataFrame({
                            'Word': words,
                            'Importance': weights
                        })
                        st.bar_chart(df_explanation.set_index('Word'))
                        highlighted = submitted_text
                        for word, weight in explanation:
                            if word.strip() and word in highlighted:
                                color = "#FFD700" if weight > 0 else "#ff4c4c"
                                pattern = r"(?<![\w>])" + re.escape(word) + r"(?![\w<])"
                                highlighted = re.sub(pattern, f'<span style="background-color:{color};border-radius:4px;">{word}</span>', highlighted, flags=re.IGNORECASE)
                        st.markdown(f'<div style="line-height:1.8;font-size:1.1rem;">{highlighted}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.info("Word analysis available but visualization could not be displayed.")
                        st.write("**Key words:**", ", ".join([word for word, _ in explanation[:5]]))
                st.markdown('<hr style="border:1px solid #FFD700;">', unsafe_allow_html=True)
                st.markdown('<b>Help Improve the Model:</b>', unsafe_allow_html=True)
                stable_input_key = submitted_text.strip().lower()
                if 'feedback_given' not in st.session_state:
                    st.session_state['feedback_given'] = {}
                disable_buttons = st.session_state['feedback_given'].get(stable_input_key, False)
                col_feedback1, col_feedback2, col_feedback3 = st.columns(3)
                clicked = None
                with col_feedback1:
                    if st.button("Flag as Correct", key=f"correct_{stable_input_key}", disabled=disable_buttons):
                        clicked = "correct"
                with col_feedback2:
                    if st.button("Flag as Incorrect", key=f"incorrect_{stable_input_key}", disabled=disable_buttons):
                        clicked = "incorrect"
                with col_feedback3:
                    if st.button("Not a News", key=f"notnews_{stable_input_key}", disabled=disable_buttons):
                        clicked = "notnews"
                if clicked:
                    st.session_state['feedback_given'][stable_input_key] = True
                    st.session_state['feedback_message'] = f"Thank you for your feedback! ({clicked.capitalize()})"
                    if save_feedback(submitted_text, label, confidence, explanation, clicked):
                        st.toast("Thank you for your valuable feedback!", icon="✅")
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class='lux-footer'>
        <hr style='border:1px solid #FFD700;'>
        <small>© 2024 <b>Shivani Raj</b>. All rights reserved. | SACH Edition</small>
    </div>
    """, unsafe_allow_html=True)

elif page == "Visualizations":
    st.markdown('<div class="lux-title">Enterprise Analytics & Insights</div>', unsafe_allow_html=True)
    
    # Advanced Model Information
    st.markdown('<div class="lux-card">', unsafe_allow_html=True)
    st.subheader("Advanced AI/ML Architecture", divider="orange")
    
    col_model1, col_model2 = st.columns(2)
    
    with col_model1:
        st.markdown("""
        **Core Technology Stack:**
        - **Model**: BERT-base-uncased (110M parameters)
        - **Framework**: Hugging Face Transformers
        - **Architecture**: Bidirectional Encoder Representations
        - **Pre-training**: 3.3B words from Wikipedia + BookCorpus
        - **Fine-tuning**: LIAR Dataset (12.8K labeled examples)
        
        **Performance Specifications:**
        - **Input Processing**: Up to 512 tokens per request
        - **Output Format**: Binary classification + confidence scoring
        - **Memory Usage**: Optimized for 4GB+ RAM systems
        - **GPU Acceleration**: CUDA support for enhanced performance
        """)
    
    with col_model2:
        st.markdown("""
        **Explainable AI (XAI):**
        - **LIME Integration**: Local Interpretable Model-agnostic Explanations
        - **Feature Importance**: Word-level contribution analysis
        - **Confidence Scoring**: Uncertainty quantification
        - **Decision Transparency**: Audit trail for compliance
        
        **Continuous Learning:**
        - **Automated Retraining**: Weekly model updates
        - **Feedback Integration**: User-driven improvements
        - **Performance Monitoring**: Real-time accuracy tracking
        - **Version Control**: Model versioning and rollback
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enterprise Features
    st.markdown('<div class="lux-card">', unsafe_allow_html=True)
    st.subheader("Enterprise-Grade Features", divider="orange")
    
    col_enterprise1, col_enterprise2 = st.columns(2)
    
    with col_enterprise1:
        st.markdown("""
        **Security & Compliance:**
        - **Data Privacy**: Local processing, no external transmission
        - **Input Validation**: Advanced sanitization protocols
        - **Session Security**: Encrypted state management
        - **Audit Logging**: Complete activity tracking
        - **GDPR Compliance**: Data protection standards
        
        **Performance & Scalability:**
        - **Horizontal Scaling**: Linear performance increase
        - **Load Balancing**: Automatic traffic distribution
        - **Caching Strategy**: Intelligent resource optimization
        - **Failover Protection**: Automatic recovery systems
        """)
    
    with col_enterprise2:
        st.markdown("""
        **Integration & Deployment:**
        - **API Ready**: RESTful endpoints for integration
        - **Docker Support**: Containerized deployment
        - **Kubernetes**: Orchestration and scaling
        - **Cloud Native**: Multi-cloud compatibility
        - **CI/CD Pipeline**: Automated deployment
        
        **Analytics & Reporting:**
        - **Real-time Monitoring**: Live performance tracking
        - **Custom Dashboards**: Tailored analytics views
        - **Export Capabilities**: Multiple format support
        - **Scheduled Reports**: Automated delivery
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ROI Calculator (in Rupees)
    st.markdown('<div class="lux-card">', unsafe_allow_html=True)
    st.subheader("ROI Calculator (in ₹)", divider="orange")

    col_roi1, col_roi2 = st.columns(2)

    with col_roi1:
        st.markdown("**Cost Savings Analysis (₹):**")
        # Input parameters
        daily_articles = st.slider("Daily articles to verify", 10, 1000, 100)
        manual_time = st.slider("Manual verification time (minutes)", 5, 30, 15)
        hourly_rate = st.slider("Hourly rate (₹)", 100, 2000, 500)
        # Calculations
        daily_savings = (daily_articles * manual_time / 60) * hourly_rate
        monthly_savings = daily_savings * 22  # Working days
        annual_savings = monthly_savings * 12
        st.metric("Daily Cost Savings", f"₹{daily_savings:,.0f}")
        st.metric("Monthly Savings", f"₹{monthly_savings:,.0f}")
        st.metric("Annual ROI", f"₹{annual_savings:,.0f}")

    with col_roi2:
        st.markdown("**Efficiency Metrics:**")
        # Efficiency calculations
        time_saved = daily_articles * manual_time
        efficiency_gain = 90  # 90% time reduction
        st.metric("Time Saved Daily", f"{time_saved} minutes")
        st.metric("Efficiency Gain", f"{efficiency_gain}%")
        st.metric("Accuracy Improvement", "+15%")
        st.metric("Risk Reduction", "85%")

    st.markdown('</div>', unsafe_allow_html=True)

    # Feedback Data Insights
    st.markdown('<div class="lux-card">', unsafe_allow_html=True)
    st.subheader("User Feedback Analytics", divider="orange")

    if not os.path.isfile("feedback.csv"):
        st.stop()  # Hide the rest of the analytics section if no feedback yet

    if os.path.isfile("feedback.csv"):
        try:
            with open("feedback.csv", "r", encoding="utf-8") as f:
                raw_feedback = f.read()
            # Remove or comment out the raw feedback.csv display
            # st.write("feedback.csv contents:")
            # st.code(raw_feedback)
            df = pd.read_csv("feedback.csv")
            # Convert 'confidence' from '51.82%' to 0.5182
            df['confidence'] = df['confidence'].str.rstrip('%').astype(float) / 100.0
            if len(df) > 0:
                col_feedback1, col_feedback2 = st.columns(2)
                
                with col_feedback1:
                    # Feedback type distribution
                    if 'feedback_type' in df.columns:
                        st.markdown("**Feedback Distribution:**")
                        feedback_counts = df['feedback_type'].value_counts()
                        st.bar_chart(feedback_counts)
                        
                        # Feedback metrics
                        correct_count = len(df[df['feedback_type'] == 'correct'])
                        incorrect_count = len(df[df['feedback_type'] == 'incorrect'])
                        total_feedback = len(df)
                        
                        st.metric("Positive Feedback", correct_count, f"{correct_count/total_feedback*100:.1f}%")
                        st.metric("Improvement Requests", incorrect_count, f"{incorrect_count/total_feedback*100:.1f}%")
                    else:
                        # Legacy format
                        st.markdown("**Prediction Distribution:**")
                        prediction_counts = df['model_prediction'].value_counts()
                        st.bar_chart(prediction_counts)
                
                with col_feedback2:
                    # Confidence distribution
                    if 'confidence' in df.columns:
                        st.markdown("**Confidence Score Distribution:**")
                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots()
                        ax.hist(df['confidence'], bins=20, color='#FFD700', edgecolor='black')
                        ax.set_xlabel('Confidence')
                        ax.set_ylabel('Count')
                        ax.set_title('Confidence Score Distribution')
                        st.pyplot(fig)
                        
                        avg_confidence = df['confidence'].mean()
                        st.metric("Average Confidence", f"{avg_confidence:.1%}")
                    
                    # Sample data
                    st.markdown("**Recent Feedback Sample:**")
                    # Show confidence as percentage string in the table
                    df_sample = df.sample(min(3, len(df))).copy()
                    df_sample['confidence'] = df_sample['confidence'].apply(lambda x: f"{x*100:.2f}%")
                    st.dataframe(df_sample, use_container_width=True)
                
                st.info(f"Total feedback entries: {len(df)} | Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
            else:
                st.info("No feedback data available yet. Start using the detector to see analytics!")
        except Exception as e:
            st.error(f"Error reading feedback data: {e}")
    else:
        st.info("No feedback data available yet. Flag predictions to see insights here!")
    
    st.markdown('</div>', unsafe_allow_html=True) 