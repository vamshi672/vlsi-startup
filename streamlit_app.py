import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="VLSI AI Tool", layout="wide")

st.title("🚀 VLSI AI Optimization Tool")

st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose Feature", ["Home", "Upload Design", "Results"])

# HOME
if option == "Home":
    st.header("Welcome 👋")
    st.write("This tool helps in VLSI optimization")

# UPLOAD
elif option == "Upload Design":
    st.header("Upload Your Design")
    file = st.file_uploader("Upload CSV file")

    if file:
        df = pd.read_csv(file)
        st.write("Preview of data:")
        st.dataframe(df)

# RESULTS
elif option == "Results":
    st.header("Results Dashboard")

    # Dummy data (for demo)
    data = pd.DataFrame({
        "Metric": ["Power", "Performance", "Area"],
        "Value": [10, 20, 15]
    })

    fig = px.bar(data, x="Metric", y="Value", title="PPA Metrics")
    st.plotly_chart(fig)
