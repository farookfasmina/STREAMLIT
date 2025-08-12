import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Titanic Classifier", layout="wide")

@st.cache_data
def load_data(path="data/train.csv"):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path="model.pkl"):
    return joblib.load(path)

df = load_data()
model = None
try:
    model = load_model()
except Exception as e:
    st.error("Model failed to load: " + str(e))

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Visualisations", "Predict", "Model Performance", "About"])

if page == "Overview":
    st.header("Dataset overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())
    st.markdown("### Column info")
    st.write(df.dtypes)

    # interactive filter example
    col = st.selectbox("Filter by column", df.columns)
    if df[col].dtype == 'object':
        val = st.selectbox("Value", df[col].dropna().unique())
        st.dataframe(df[df[col] == val].head())

elif page == "Visualisations":
    st.header("Visualisations")
    st.subheader("Survival count")
    fig = px.histogram(df, x='Survived', title="Survived distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Age vs Fare (colored by Survived)")
    fig2 = px.scatter(df, x='Age', y='Fare', color='Survived', hover_data=['Name'])
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation heatmap")
    corr = df.select_dtypes(include=np.number).corr()
    fig3, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig3)

elif page == "Predict":
    st.header("Make a prediction")
    # example inputs (match training features)
    pclass = st.selectbox("Pclass", [1,2,3])
    sex = st.selectbox("Sex", ["male","female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
    sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=32.0)
    embarked = st.selectbox("Embarked", ["S","C","Q"])

    if st.button("Predict"):
        if model is None:
            st.error("Model not loaded.")
        else:
            X_new = pd.DataFrame([{
                'Pclass': pclass, 'Sex': sex, 'Age': age,
                'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked
            }])
            try:
                prob = model.predict_proba(X_new)[0]
                pred = model.predict(X_new)[0]
                st.write("Predicted class:", int(pred))
                st.write("Probabilities:", prob)
            except Exception as e:
                st.error("Prediction error: " + str(e))

elif page == "Model Performance":
    st.header("Model performance (test set)")
    st.markdown("Load your evaluation metrics from notebook or recompute here.")
    # Optionally, load saved metrics or recompute from a test CSV
    st.info("Show confusion matrix, accuracy, classification report here.")

else:
    st.header("About")
    st.write("Project: Machine Learning Model Deployment with Streamlit")
    st.write("Author: FASMINA")

import os
import joblib
import requests

MODEL_URL = "https://raw.githubusercontent.com/yourusername/your-repo-name/main/model.pkl"
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

model = joblib.load(MODEL_PATH)
