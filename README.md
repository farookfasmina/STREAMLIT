# Titanic Survival Prediction Web App

## Project Overview
This project predicts passenger survival on the Titanic using a machine learning model.
It includes an interactive Streamlit web app where users can explore the dataset, visualize insights, make predictions, and view model performance metrics.

---

## Repository Contentsa
- `app.py`: Streamlit web app source code
- `model.pkl`: Trained ML pipeline (preprocessing + model)
- `data/`: Dataset files (train.csv, test.csv)
- `notebooks/model_training.ipynb`: Model training notebook
- `requirements.txt`: Python dependencies
- `README.md`: Project documentation

---

## Dataset
- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- Target: Survived (0 = No, 1 = Yes)

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/farookfasmina0614@gmail.come/STREAMLIT.git
cd STREAMLIT
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run locally
```bash
streamlit run app.py
```

---

## Features
- **Overview**: Dataset preview, shape, columns, and filter options
- **Visualisations**: Interactive charts (Survival count, Age vs Fare, Correlation heatmap)
- **Predict**: User inputs for passenger features and survival prediction
- **Model Performance**: Accuracy, confusion matrix, and classification report
- **About**: Project and author info

---

## Model Details
- Preprocessing: Handles missing values, scales numerics, encodes categoricals
- Algorithms tested: Logistic Regression, Random Forest
- Best model chosen based on cross-validation accuracy

---

## Deployment
The app can be deployed on [Streamlit Cloud](https://app-2ng3qir2rpb6s3d6leasm7.streamlit.app/) by connecting this repo.

---

## Author
- Your Name:Fasmina
- Email: farookfasmina0614@gmail.com
- GitHub: [farookfasmina0614@gmail.com](https://github.com/farookfasmina0614@gmail.com)
