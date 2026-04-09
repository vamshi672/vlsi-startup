import streamlit as st

st.title("VLSI Startup App 🚀")

st.write("✅ App is working!")

name = st.text_input("Enter your name")

if name:
    st.write(f"Hello {name}")
