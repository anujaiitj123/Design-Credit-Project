import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def make_unique(cols):
    seen = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
    return new_cols


def main():
    file_path = r"C:\Users\ROG STRIX\OneDrive\Desktop\LaserWelding dataset.xlsx"  

    df_raw = pd.read_excel(file_path, header=1)

    # Copy and set proper header from the first row
    df = df_raw.copy()
    df.columns = df.iloc[0]
    df = df.drop(index=0).reset_index(drop=True)

    # Ensure column names are unique
    df.columns = make_unique(df.columns)

    target_col = "Tensile strength (Welds)"

    # Replace '-' with NaN in the target and convert to numeric
    df[target_col] = df[target_col].replace("-", np.nan)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Drop rows where target is missing
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # Columns that are just IDs or units 
    id_col = "S. No."
    unit_cols = [
        "Units",      # thickness unit
        "Units_1",    # another thickness unit
        "Unit ",      # microhardness unit
        "Unit",       # another microhardness unit
        "Unit _1",    # power unit
        "Unit_1",     # another power unit
        "Unit_2",     # tensile strength unit
        "Units_2",    # resistance unit
        "Speed units" # e.g. m/s
    ]

    # Numeric features we want to use
    numeric_cols = [
        "Mat1_thickness",
        "Mat2_thickness",
        "Mat1_microhardness",
        "Mat2_microhardness",
        "Core Beam Power",
        "Ring Beam Power",
        "Laser Speed",
        "Weld Configuration",
        "Electrical resistance/conductivity",
        "Mat1",
        "Mat2",
    ]

    # Categorical features we want to one-hot encode
    categorical_cols = [
        "Mat1_name",
        "Mat2_name",
        "Laser type",
        "Laser mode",
    ]

    # Drop ID and unit columns (if they exist)
    df_model = df.drop(columns=[id_col] + unit_cols, errors="ignore")

    # Keep only numeric columns that actually exist
    numeric_cols_existing = [c for c in numeric_cols if c in df_model.columns]
    for col in numeric_cols_existing:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    # Keep only categorical columns that actually exist
    categorical_cols_existing = [c for c in categorical_cols if c in df_model.columns]

    # Target vector
    y = df_model[target_col].values.astype(float)

    # Separate numeric and categorical features
    X_num = df_model[numeric_cols_existing].copy()
    X_cat = df_model[categorical_cols_existing].astype("category")

    # One-hot encode categorical features
    X_cat_encoded = pd.get_dummies(X_cat, drop_first=True)

    # Concatenate numeric + encoded categorical features
    X = pd.concat(
        [X_num.reset_index(drop=True), X_cat_encoded.reset_index(drop=True)],
        axis=1,
    )

    # Fill any remaining NaNs (e.g., from Mat1/Mat2 missing values)
    X = X.fillna(0)

    print("Final feature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5


    print("\nRandom Forest Regression Results:")
    print(f"R^2   : {r2:.4f}")
    print(f"MAE   : {mae:.4f}")
    print(f"RMSE  : {rmse:.4f}")


   #Feature Importance (Top 15)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    top_features = X.columns[indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(top_features)), top_importances)
    plt.xticks(range(len(top_features)), top_features, rotation=90)
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importances (Top 15)")
    plt.tight_layout()
    plt.show()

    #Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.xlabel("Actual Tensile Strength")
    plt.ylabel("Predicted Tensile Strength")
    plt.title("Actual vs Predicted Tensile Strength")
    plt.tight_layout()
    plt.show()

    #Residuals vs Predicted
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted Tensile Strength")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
