import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import openai

st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.title("📊 AI Data Analyst Dashboard")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Column Information")
    st.write(df.dtypes)

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    st.sidebar.title("Dashboard Builder")

    chart_type = st.sidebar.selectbox(
        "Select Chart",
        ["Bar Chart","Line Chart","Scatter Plot","Histogram"]
    )

    x_axis = st.sidebar.selectbox("X Axis", df.columns)
    y_axis = st.sidebar.selectbox("Y Axis", numeric_cols)

    if st.sidebar.button("Generate Chart"):

        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis)

        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis)

        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis)

        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_axis)

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Automatic Data Insights")

    st.write("Average Values")
    st.write(df[numeric_cols].mean())

    st.write("Maximum Values")
    st.write(df[numeric_cols].max())

    st.write("Minimum Values")
    st.write(df[numeric_cols].min())

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    st.pyplot(fig)

        st.write("AI Response:")
        st.write("This feature works after connecting OpenAI API.")
