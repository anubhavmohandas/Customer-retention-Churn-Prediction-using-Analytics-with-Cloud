import streamlit as st
import pandas as pd

st.title("Page 1: Data Upload")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.session_state["data"] = df

    st.success("File uploaded successfully!")
    st.write("### Preview of Data")
    st.dataframe(df.head())
