"""
eval/eval_app.py — Streamlit UI for running the evaluation suite.
"""

import streamlit as st
from pathlib import Path
import sys
import os
import time
import traceback

# Add project root to path to allow imports from agent, tools etc.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# The eval suite needs to import from agent, which needs env vars
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

# Now we can import the suite
from eval import eval_suite

st.set_page_config(page_title="Lexi Agent Evaluation", layout="wide")

st.title("⚖️ Lexi Agent Evaluation Runner")
st.markdown("This app runs the automated evaluation suite (`eval/eval_suite.py`) against the agent and displays the results.")

# Check for API keys
provider = os.getenv("LLM_PROVIDER", "groq").lower()
api_key_name = f"{provider.upper()}_API_KEY"
api_key = os.getenv(api_key_name)

if not api_key:
    st.error(f"**Missing API Key.** The evaluation suite requires an LLM-as-judge. Please set `{api_key_name}` in your `.env` file.")
else:
    st.success(f"Found `{api_key_name}`. Ready to run evaluation using `{provider}`.")

st.divider()

if st.button("🚀 Run Full Evaluation Suite", use_container_width=True, type="primary"):

    status_text = st.empty()
    
    try:
        status_text.info("Running evaluation... This may take a few minutes depending on the LLM provider and rate limits. Please wait.")
        
        with st.spinner("Evaluating agent on all queries..."):
            summary = eval_suite.run_evaluation()
        
        status_text.success("Evaluation finished successfully!")
        st.balloons()
        
        st.header("📊 Evaluation Report")
        if eval_suite.REPORT_PATH.exists():
            with open(eval_suite.REPORT_PATH, "r", encoding="utf-8") as f:
                report_content = f.read()
            st.markdown(report_content, unsafe_allow_html=True)
        else:
            st.error("Report file was not generated.")

    except Exception as e:
        st.error(f"An error occurred during evaluation: {e}")
        st.code(traceback.format_exc())

st.info("After running the evaluation, the results will be saved to `eval/results_report.md` and displayed below if the run is triggered.")