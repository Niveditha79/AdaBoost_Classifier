# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AdaBoost Customer Churn Dashboard",
    page_icon="🚀",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#ECFEFF,#EEF2FF);
}

.main-title {
    font-size: 48px;
    font-weight: bold;
    color: #0F172A;
    text-align: center;
}

.sub-title {
    font-size: 30px;
    color: #1E3A8A;
    font-weight: bold;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    text-align: center;
}

.metric-text {
    font-size: 28px;
    font-weight: bold;
    color: #0F172A;
}

section[data-testid="stSidebar"] {
    background: #E0F2FE;
}

.stButton>button {
    background: linear-gradient(90deg,#2563EB,#7C3AED);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    border: none;
    font-size: 18px;
}

.stButton>button:hover {
    background: linear-gradient(90deg,#1D4ED8,#6D28D9);
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL & SCALER
# =========================================================

model = pickle.load(
    open("models/adaboost_model.pkl", "rb")
)

scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    "data/Churn_Modelling.csv"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("AdaBoost Parameters")

test_size = st.sidebar.slider(
    "Test Size",
    0.1,
    0.5,
    0.2
)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<p class="main-title">'
    'AdaBoost Customer Churn Prediction Dashboard'
    '</p>',
    unsafe_allow_html=True
)

st.write(
    "This dashboard predicts customer churn "
    "using AdaBoost Classification."
)

# =========================================================
# DATASET PREVIEW
# =========================================================

st.markdown(
    '<p class="sub-title">'
    'Dataset Preview'
    '</p>',
    unsafe_allow_html=True
)

st.dataframe(
    df.head(),
    use_container_width=True
)

# =========================================================
# INFO CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(f"""
    <div class="card">
        <h3>Total Customers</h3>
        <p class="metric-text">{df.shape[0]}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class="card">
        <h3>Total Features</h3>
        <p class="metric-text">{df.shape[1]}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown(f"""
    <div class="card">
        <h3>Missing Values</h3>
        <p class="metric-text">
        {df.isnull().sum().sum()}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col4:

    churn_rate = df["Exited"].mean() * 100

    st.markdown(f"""
    <div class="card">
        <h3>Churn Rate</h3>
        <p class="metric-text">
        {churn_rate:.2f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# DATA TYPES & MISSING VALUES
# =========================================================

col5, col6 = st.columns(2)

with col5:

    st.markdown(
        '<p class="sub-title">'
        'Data Types'
        '</p>',
        unsafe_allow_html=True
    )

    st.dataframe(
        pd.DataFrame(
            df.dtypes,
            columns=["Data Type"]
        ),
        use_container_width=True
    )

with col6:

    st.markdown(
        '<p class="sub-title">'
        'Missing Values'
        '</p>',
        unsafe_allow_html=True
    )

    st.dataframe(
        pd.DataFrame(
            df.isnull().sum(),
            columns=["Missing Count"]
        ),
        use_container_width=True
    )

# =========================================================
# STATISTICAL SUMMARY
# =========================================================

st.markdown(
    '<p class="sub-title">'
    'Statistical Summary'
    '</p>',
    unsafe_allow_html=True
)

st.dataframe(
    df.describe(),
    use_container_width=True
)

# =========================================================
# CHURN DISTRIBUTION
# =========================================================

st.markdown(
    '<p class="sub-title">'
    'Churn Distribution'
    '</p>',
    unsafe_allow_html=True
)

fig1, ax1 = plt.subplots(figsize=(5,3))

sns.countplot(
    x=df["Exited"],
    palette="coolwarm",
    ax=ax1
)

ax1.set_xticklabels(
    ["Retained", "Exited"]
)

st.pyplot(fig1)

# =========================================================
# CORRELATION HEATMAP
# =========================================================

temp_df = df.copy()

temp_df = temp_df.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1
)

temp_df = pd.get_dummies(
    temp_df,
    columns=["Geography", "Gender"],
    drop_first=True
)

st.markdown(
    '<p class="sub-title">'
    'Correlation Heatmap'
    '</p>',
    unsafe_allow_html=True
)

fig2, ax2 = plt.subplots(figsize=(10,5))

sns.heatmap(
    temp_df.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax2
)

st.pyplot(fig2)

# =========================================================
# FEATURE DISTRIBUTION
# =========================================================

st.markdown(
    '<p class="sub-title">'
    'Feature Distribution'
    '</p>',
    unsafe_allow_html=True
)

feature = st.selectbox(
    "Select Feature",
    [
        "CreditScore",
        "Age",
        "Balance",
        "EstimatedSalary"
    ]
)

fig3, ax3 = plt.subplots(figsize=(6,3))

sns.histplot(
    df[feature],
    kde=True,
    color="purple",
    ax=ax3
)

st.pyplot(fig3)

# =========================================================
# PREPROCESSING
# =========================================================

data = df.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1
)

data = pd.get_dummies(
    data,
    columns=["Geography", "Gender"],
    drop_first=True
)

for col in data.columns:

    if data[col].dtype == "bool":

        data[col] = data[col].astype(int)

X = data.drop("Exited", axis=1)

y = data["Exited"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

# =========================================================
# FEATURE SCALING
# =========================================================

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

# =========================================================
# PREDICTIONS
# =========================================================

pred = model.predict(
    X_test_scaled
)

# =========================================================
# MODEL EVALUATION
# =========================================================

accuracy = accuracy_score(
    y_test,
    pred
)

precision = precision_score(
    y_test,
    pred
)

recall = recall_score(
    y_test,
    pred
)

f1 = f1_score(
    y_test,
    pred
)

st.markdown(
    '<p class="sub-title">'
    'Model Evaluation'
    '</p>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

with m2:
    st.metric(
        "Precision",
        f"{precision:.4f}"
    )

with m3:
    st.metric(
        "Recall",
        f"{recall:.4f}"
    )

with m4:
    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

# =========================================================
# CONFUSION MATRIX
# =========================================================

st.markdown(
    '<p class="sub-title">'
    'Confusion Matrix'
    '</p>',
    unsafe_allow_html=True
)

fig4, ax4 = plt.subplots(figsize=(5,3))

cm = confusion_matrix(
    y_test,
    pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Retained", "Exited"]
)

disp.plot(
    ax=ax4,
    cmap=plt.cm.coolwarm
)

st.pyplot(fig4)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

st.markdown(
    '<p class="sub-title">'
    'Classification Report'
    '</p>',
    unsafe_allow_html=True
)

report = classification_report(
    y_test,
    pred,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)

# =========================================================
# PREDICTION SECTION
# =========================================================

st.markdown(
    '<p class="sub-title">'
    'Customer Churn Prediction'
    '</p>',
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

with c1:

    credit_score = st.slider(
        "Credit Score",
        300,
        850,
        650
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    age = st.slider(
        "Age",
        18,
        100,
        35
    )

    tenure = st.slider(
        "Tenure",
        0,
        10,
        5
    )

with c2:

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0
    )

    num_products = st.slider(
        "Products",
        1,
        4,
        1
    )

    has_card = st.selectbox(
        "Has Credit Card",
        ["No", "Yes"]
    )

    is_active = st.selectbox(
        "Active Member",
        ["No", "Yes"]
    )

    salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=100000.0
    )

# =========================================================
# INPUT PROCESSING
# =========================================================

has_card_val = 1 if has_card == "Yes" else 0

is_active_val = 1 if is_active == "Yes" else 0

gender_male = 1 if gender == "Male" else 0

geography_germany = 0
geography_spain = 0

if geography == "Germany":
    geography_germany = 1

elif geography == "Spain":
    geography_spain = 1

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("Predict Churn"):

    input_vector = np.array([[
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        has_card_val,
        is_active_val,
        salary,
        geography_germany,
        geography_spain,
        gender_male
    ]])

    scaled_vector = scaler.transform(
        input_vector
    )

    prediction = model.predict(
        scaled_vector
    )[0]

    probability = model.predict_proba(
        scaled_vector
    )[0][1] * 100

    if prediction == 1:

        st.error(
            f"Customer may leave the bank. "
            f"Churn Probability: {probability:.2f}%"
        )

    else:

        st.success(
            f"Customer likely to stay. "
            f"Churn Probability: {probability:.2f}%"
        )