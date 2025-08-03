import streamlit as st

def app(data=None):
    st.title("Free Fall Simulator")
    st.markdown("Coming soon: vertical motion under gravity.")
    if data:
        st.write("AI Input:", data)
