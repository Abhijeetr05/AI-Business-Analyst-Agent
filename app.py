"""
AI Business Analyst Agent
Main Streamlit Application Entry Point
"""

import streamlit as st
import pandas as pd
import os
import tempfile

from agents.analyst_agent import BusinessAnalystAgent
from utils.data_processor import DataProcessor
from utils.dashboard_generator import DashboardGenerator
from utils.report_generator import ReportGenerator

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Business Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #e94560, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        color: #a8b2d8;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .kpi-card h3 {
        font-size: 0.85rem;
        font-weight: 500;
        opacity: 0.9;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .kpi-card h2 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0 0;
    }

    .insight-card {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }

    .rec-card {
        background: #f0fff4;
        border-left: 4px solid #38a169;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }

    .chat-message-user {
        background: #e8f4fd;
        padding: 0.8rem 1rem;
        border-radius: 12px 12px 4px 12px;
        margin: 0.5rem 0;
        margin-left: 2rem;
        color: #1a1a2e;
    }

    .chat-message-ai {
        background: #f0f4ff;
        padding: 0.8rem 1rem;
        border-radius: 12px 12px 12px 4px;
        margin: 0.5rem 0;
        margin-right: 2rem;
        border-left: 3px solid #667eea;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
        margin-bottom: 1.5rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f0f4ff;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }

    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }

    .sidebar-info {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ─────────────────────────────────────────────
def init_session_state():
    defaults = {
        "df": None,
        "file_name": None,
        "kpis": None,
        "insights": None,
        "recommendations": None,
        "chat_history": [],
        "agent": None,
        "analysis_done": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─── Header ───────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>📊 AI Business Analyst Agent</h1>
        <p>Upload your CSV dataset and get intelligent dashboards, insights, and recommendations instantly</p>
    </div>
    """, unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        st.markdown("### 🤖 AI Model (Ollama)")
        model_choice = st.selectbox(
            "Select LLM Model",
            ["qwen3", "llama3", "gemma"],
            index=0,
            help="Make sure Ollama is running locally with the selected model pulled."
        )

        st.markdown("---")
        st.markdown("### 📁 Dataset Info")
        if st.session_state.df is not None:
            df = st.session_state.df
            st.success(f"✅ **{st.session_state.file_name}**")
            st.metric("Rows", f"{len(df):,}")
            st.metric("Columns", len(df.columns))
            st.metric("Missing Values", int(df.isnull().sum().sum()))
        else:
            st.info("No dataset loaded yet.")

        st.markdown("---")
        st.markdown("""
        <div class="sidebar-info">
            <strong>How to use:</strong><br>
            1. Upload a CSV file<br>
            2. Click Analyze<br>
            3. Explore dashboard tabs<br>
            4. Ask questions in Chat<br>
            5. Download PDF report
        </div>
        """, unsafe_allow_html=True)

        return model_choice


# ─── Upload Section ───────────────────────────────────────────────────────────
def render_upload_section(model_choice):
    st.markdown('<p class="section-header">📂 Upload Dataset</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload your business dataset in CSV format",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 Analyze Dataset", use_container_width=True)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name
            st.session_state.analysis_done = False

            with st.expander("👁️ Preview Dataset", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return

    if analyze_btn:
        if st.session_state.df is None:
            st.warning("⚠️ Please upload a CSV file first.")
        else:
            run_analysis(model_choice)


# ─── Run Full Analysis ─────────────────────────────────────────────────────────
def run_analysis(model_choice):
    df = st.session_state.df

    with st.spinner("🔍 Analyzing your dataset with AI..."):
        try:
            # Initialize components
            processor = DataProcessor(df)
            processed_df = processor.process()
            kpis = processor.generate_kpis()

            agent = BusinessAnalystAgent(model=model_choice)
            insights = agent.generate_insights(processed_df, kpis)
            recommendations = agent.generate_recommendations(processed_df, kpis, insights)

            st.session_state.kpis = kpis
            st.session_state.insights = insights
            st.session_state.recommendations = recommendations
            st.session_state.agent = agent
            st.session_state.analysis_done = True

            st.success("✅ Analysis complete! Explore the tabs below.")

        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            st.info("💡 Make sure Ollama is running: `ollama serve` and the model is pulled.")


# ─── KPI Dashboard ────────────────────────────────────────────────────────────
def render_kpi_dashboard():
    kpis = st.session_state.kpis
    if not kpis:
        return

    st.markdown('<p class="section-header">📈 Key Performance Indicators</p>', unsafe_allow_html=True)

    cols = st.columns(min(len(kpis), 4))
    for i, (label, value) in enumerate(list(kpis.items())[:8]):
        col_idx = i % 4
        with cols[col_idx]:
            st.markdown(f"""
            <div class="kpi-card">
                <h3>{label}</h3>
                <h2>{value}</h2>
            </div>
            <br>
            """, unsafe_allow_html=True)


# ─── Visualizations Tab ───────────────────────────────────────────────────────
def render_visualizations():
    df = st.session_state.df
    if df is None:
        return

    st.markdown('<p class="section-header">📊 Visual Analytics</p>', unsafe_allow_html=True)

    gen = DashboardGenerator(df)
    charts = gen.generate_all_charts()

    if not charts:
        st.info("No suitable columns found for automatic chart generation.")
        return

    for title, fig in charts:
        st.subheader(title)
        st.plotly_chart(fig, use_container_width=True)


# ─── Insights Tab ─────────────────────────────────────────────────────────────
def render_insights():
    insights = st.session_state.insights
    if not insights:
        st.info("Run the analysis to generate insights.")
        return

    st.markdown('<p class="section-header">💡 AI-Generated Business Insights</p>', unsafe_allow_html=True)

    for i, insight in enumerate(insights, 1):
        st.markdown(f"""
        <div class="insight-card">
            <strong>💡 Insight {i}</strong><br>{insight}
        </div>
        """, unsafe_allow_html=True)


# ─── Recommendations Tab ──────────────────────────────────────────────────────
def render_recommendations():
    recs = st.session_state.recommendations
    if not recs:
        st.info("Run the analysis to generate recommendations.")
        return

    st.markdown('<p class="section-header">🎯 Strategic Recommendations</p>', unsafe_allow_html=True)

    for i, rec in enumerate(recs, 1):
        st.markdown(f"""
        <div class="rec-card">
            <strong>✅ Recommendation {i}</strong><br>{rec}
        </div>
        """, unsafe_allow_html=True)


# ─── Chat Interface ───────────────────────────────────────────────────────────
def render_chat():
    st.markdown('<p class="section-header">💬 Ask Your Data</p>', unsafe_allow_html=True)

    if st.session_state.agent is None or not st.session_state.analysis_done:
        st.info("📊 Please upload a dataset and run the analysis first to start chatting.")
        return

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # Suggested questions
    st.markdown("**💭 Suggested Questions:**")
    suggestions = [
        "Which product generated the highest revenue?",
        "What are the monthly sales trends?",
        "Which region performed best?",
        "What is the profit margin by category?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(suggestions):
        if cols[i % 2].button(q, key=f"suggest_{i}"):
            handle_chat_question(q)
            st.rerun()

    # User input
    user_q = st.chat_input("Ask a question about your data...")
    if user_q:
        handle_chat_question(user_q)
        st.rerun()


def handle_chat_question(question: str):
    st.session_state.chat_history.append({"role": "user", "content": question})
    agent = st.session_state.agent
    answer = agent.answer_question(
        question,
        st.session_state.df,
        st.session_state.kpis,
        st.session_state.insights,
    )
    st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ─── PDF Report Tab ───────────────────────────────────────────────────────────
def render_report():
    st.markdown('<p class="section-header">📄 Business Report</p>', unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.info("Run the analysis first to generate a report.")
        return

    st.write("Click the button below to generate and download your professional PDF business report.")

    if st.button("📥 Generate & Download PDF Report", use_container_width=True):
        with st.spinner("📝 Generating PDF report..."):
            try:
                gen = ReportGenerator(
                    df=st.session_state.df,
                    file_name=st.session_state.file_name,
                    kpis=st.session_state.kpis,
                    insights=st.session_state.insights,
                    recommendations=st.session_state.recommendations,
                )
                pdf_bytes = gen.generate()

                st.download_button(
                    label="💾 Download Report PDF",
                    data=pdf_bytes,
                    file_name="business_analysis_report.pdf",
                    mime="application/pdf",
                )
                st.success("✅ Report generated successfully!")
            except Exception as e:
                st.error(f"❌ Report generation failed: {e}")


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    init_session_state()
    render_header()
    model_choice = render_sidebar()
    render_upload_section(model_choice)

    if st.session_state.df is not None:
        st.markdown("---")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 KPI Dashboard",
            "📊 Visualizations",
            "💡 Insights",
            "🎯 Recommendations",
            "💬 Chat",
        ])

        with tab1:
            render_kpi_dashboard()

        with tab2:
            render_visualizations()

        with tab3:
            render_insights()

        with tab4:
            render_recommendations()

        with tab5:
            render_chat()

        if st.session_state.analysis_done:
            st.markdown("---")
            render_report()


if __name__ == "__main__":
    main()