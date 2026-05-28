# =========================================================
# ADABOOST CLASSIFIER EVALUATION
# =========================================================

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

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
# PREPROCESSING
# =========================================================

df = df.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1
)

df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True
)

for col in df.columns:

    if df[col].dtype == "bool":

        df[col] = df[col].astype(int)

# =========================================================
# FEATURES & TARGET
# =========================================================

X = df.drop("Exited", axis=1)

y = df["Exited"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# FEATURE SCALING
# =========================================================

X_test_scaled = scaler.transform(
    X_test
)

# =========================================================
# PREDICTIONS
# =========================================================

pred = model.predict(
    X_test_scaled
)

# =========================================================
# METRICS
# =========================================================

print(
    "Accuracy Score:",
    accuracy_score(y_test, pred)
)

print(
    "\nClassification Report:\n",
    classification_report(y_test, pred)
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Retained", "Exited"]
)

disp.plot(cmap=plt.cm.coolwarm)

plt.title("Confusion Matrix")

plt.show()