import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import requests

st.set_page_config(page_title="Titanic Classifier", layout="wide")

# Cache dataset
@st.cache_data
def load_data(path="data/train.csv"):
    return pd.read_csv(path)

# Cache model
@st.cache_resource
def load_model(path="model.pkl"):
    return joblib.load(path)

# Load
df = load_data()
model = None
try:
    model = load_model()
except Exception as e:
    st.error("Model failed to load: " + str(e))

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", 
    ["Overview", "Visualisations", "Predict", "Model Performance", "About"]
)

# ---------------------------
# OVERVIEW
# ---------------------------
if page == "Overview":
    st.header("Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())
    st.markdown("### Column Info")
    st.write(df.dtypes)

    col = st.selectbox("Filter by column", df.columns)
    if df[col].dtype == 'object':
        val = st.selectbox("Value", df[col].dropna().unique())
        st.dataframe(df[df[col] == val].head())

# ---------------------------
# VISUALISATIONS
# ---------------------------
elif page == "Visualisations":
    st.header("Visualisations")
    
    st.subheader("Survival Count")
    fig = px.histogram(df, x='Survived', title="Survived Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Age vs Fare (Colored by Survival)")
    fig2 = px.scatter(df, x='Age', y='Fare', color='Survived', hover_data=['Name'])
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=np.number).corr()
    fig3, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig3)

# ---------------------------
# PREDICT
# ---------------------------
elif page == "Predict":
    st.header("Make a Prediction")
    
    pclass = st.selectbox("Pclass", [1,2,3], help="Passenger Class")
    sex = st.selectbox("Sex", ["male","female"], help="Passenger Gender")
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
    sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0, help="Siblings/Spouses aboard")
    parch = st.number_input("Parch", min_value=0, max_value=10, value=0, help="Parents/Children aboard")
    fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=32.0)
    embarked = st.selectbox("Embarked", ["S","C","Q"], help="Port of Embarkation")

    if st.button("Predict"):
        if model is None:
            st.error("Model not loaded.")
        else:
            X_new = pd.DataFrame([{
                'Pclass': pclass, 'Sex': sex, 'Age': age,
                'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked
            }])
            with st.spinner("Predicting..."):
                try:
                    prob = model.predict_proba(X_new)[0]
                    pred = model.predict(X_new)[0]
                    st.write("Predicted Class:", int(pred))
                    st.write("Probabilities:", prob)
                except Exception as e:
                    st.error("Prediction error: " + str(e))

# ---------------------------
# MODEL PERFORMANCE

elif page == "Model Performance":
    st.header("Model Performance (Test Set)")
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    test_path = "data/test.csv"
    test_df = None

    # Try loading from disk
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
    else:
        st.warning("Test dataset not found. Please upload 'data/test.csv'.")
        uploaded_file = st.file_uploader("Upload test.csv", type="csv")
        if uploaded_file is not None:
            test_df = pd.read_csv(uploaded_file)
    
    if test_df is not None:
        if 'Survived' not in test_df.columns:
            st.error("Uploaded test set must contain a 'Survived' column.")
        elif model:
            X_test = test_df[features]
            y_test = test_df['Survived']
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {acc:.2f}")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.text(classification_report(y_test, y_pred))
        else:
            st.error("Model not loaded.")

# ---------------------------
# ABOUT
# ---------------------------
else:
    st.header("About")
    st.write("Project: Machine Learning Model Deployment with Streamlit")
    st.write("Author: Fasmina")


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
