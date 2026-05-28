# =========================================================
# ADABOOST CLASSIFIER TRAINING SCRIPT
# =========================================================

import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier

# =========================================================
# CREATE MODELS FOLDER
# =========================================================

os.makedirs("models", exist_ok=True)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    "data/Churn_Modelling.csv"
)

# =========================================================
# DROP UNUSED COLUMNS
# =========================================================

df = df.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1
)

# =========================================================
# ENCODING
# =========================================================

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# =========================================================
# MODEL
# =========================================================

model = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

# =========================================================
# TRAIN MODEL
# =========================================================

model.fit(
    X_train_scaled,
    y_train
)

# =========================================================
# SAVE MODEL & SCALER
# =========================================================

pickle.dump(
    model,
    open(
        "models/adaboost_model.pkl",
        "wb"
    )
)

pickle.dump(
    scaler,
    open(
        "models/scaler.pkl",
        "wb"
    )
)

print("AdaBoost Model Saved Successfully!")