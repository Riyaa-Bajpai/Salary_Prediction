import streamlit as st
from predict import show_predict_page
from insights import show_insights_page

# Page setup
st.set_page_config(page_title="Salary Prediction App", layout="wide")

with st.sidebar.expander("🚀 Navigation", expanded=True):
    page = st.radio("Go to", ("🔮 Predict", "📊 Insights"))
# Route pages
if page == "🔮 Predict":
    show_predict_page()
else:
    show_insights_page()
