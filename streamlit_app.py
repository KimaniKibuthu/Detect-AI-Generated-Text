import streamlit as st
import requests


def main():
    st.set_page_config(
        page_title="AI Text Detector",
        page_icon="🕵🏽",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    st.title("AI Text Detector 🕵🏽")
    st.caption("🔍 An app that detects whether text is AI generated or not")
    input_type = st.radio("Select input type", ("Text", "File"))

    if input_type == "Text":
        text = st.text_area("Enter text to predict")
        if st.button("Predict"):
            response = requests.post(
                "http://localhost:8000/predict", json={"text": text}
            )
            if response.status_code == 200:
                prediction = response.json()
                st.write(f"Prediction: {prediction['predictions_class']}")
            else:
                st.write(f"Error: {response.text}")
    else:
        uploaded_file = st.file_uploader("Upload a file", type=".txt")
        if uploaded_file is not None:
            text = uploaded_file.getvalue().decode("utf-8")
        if st.button("Predict"):
            response = requests.post(
                "http://localhost:8000/predict", json={"text": text}
            )
            if response.status_code == 200:
                prediction = response.json()
                st.write(f"Prediction: {prediction['predictions_class']}")
            else:
                st.write(f"Error: {response.text}")


if __name__ == "__main__":
    main()
