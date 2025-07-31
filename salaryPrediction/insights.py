
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_insights_page():
    # Page config
    st.set_page_config(page_title="Insights", layout="wide")
    st.title("📊 Salary Insights & Trends")

    # Load data
    df = pd.read_csv('cleaned_data.csv')
    feature_df = pd.read_csv('feature_imp.csv')
    df['CountryName'] = df['CountryName'].replace({"United Kingdom of Great Britain and Northern Ireland": "United Kingdom"})

    # Top KPIs
    st.markdown("### 📌 Key Highlights")
    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Entries", f"{len(df):,}")
    col2.metric("💰 Avg Salary", f"${df['Salary'].mean():,.0f}")
    col3.metric("🌎 Countries", df['CountryName'].nunique())
       
    # Sidebar filters
    st.sidebar.header("🔍 Filter Insights")
    selected_country = st.sidebar.selectbox("🌍 Country", ['All'] + sorted(df['CountryName'].unique()))
    selected_ed = st.sidebar.selectbox("🎓 Education", ['All'] + sorted(df['EducationLevel'].unique()))

    # Apply filters
    filtered_df = df.copy()
    if selected_country != 'All':
        filtered_df = filtered_df[filtered_df['CountryName'] == selected_country]
    if selected_ed != 'All':
        filtered_df = filtered_df[filtered_df['EducationLevel'] == selected_ed]

    # Section 1: Salary Distribution
    st.markdown("### 💵 Salary Distribution by Country")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    sns.violinplot(data=filtered_df, x='CountryName', y='Salary', ax=ax1, palette="muted")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    st.pyplot(fig1)

    # Section 2: Salary vs Experience
    st.markdown("### 📈 Salary vs Years of Experience")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=filtered_df, x='YearsCodePro', y='Salary', hue='EducationLevel', alpha=0.7, ax=ax2)
    ax2.set_xlabel("Years of Professional Coding")
    ax2.set_ylabel("Salary (USD)")
    st.pyplot(fig2)

    # Section 3: Org Size Influence
    st.markdown("### 🏢 Organization Size and Remote Work")
    colA, colB = st.columns(2)

    with colA:
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=filtered_df, x='OrgSizeLabel', y='Salary', ax=ax3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
        st.pyplot(fig3)

    with colB:
        fig4, ax4 = plt.subplots()
        sns.boxplot(data=filtered_df, x='RemoteWorkLabel', y='Salary', ax=ax4, palette='coolwarm')
        st.pyplot(fig4)

    # Section 4: Feature Importance
    st.markdown("### 🧠 What Affects Salary the Most?")
    top_feats = feature_df[feature_df['Importance'] > 0.01].sort_values(by='Importance', ascending=True)

    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top_feats, x='Importance', y='Feature', ax=ax5, palette='viridis')
    st.pyplot(fig5)

    # End note
    st.markdown("---")
    st.caption("Data source: StackOverflow Developer Survey (cleaned and modeled)")
    st.caption("Developed by: Riya Bajpai")
    st.caption("IBM PBEL (Project-Based Experiential Learning) Internship – Salary Prediction (July 2025)")
