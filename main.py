import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Sklearn preprocessing & selection
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ML Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Ignore warnings for cleaner console output
warnings.filterwarnings('ignore')

def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Removes outliers using the Interquartile Range (IQR) method and imputes with the mean."""
    df_cleaned = df.copy()
    for col in columns:
        q1, q3 = np.percentile(df_cleaned[col].dropna(), [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        # Replace outliers with NaN, then fill with column mean
        df_cleaned[col] = np.where((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound), np.nan, df_cleaned[col])
        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
    return df_cleaned

def main():
    print("🚀 Starting ML Pipeline...")
    
    # 1. Load Data
    # Note: Use relative paths in GitHub repos, not hardcoded 'C:/...' paths
    data_path = "student_data_processed.csv" 
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Error: Dataset '{data_path}' not found. Please ensure it is in the correct directory.")
        return

    # 2. Preprocessing & Outlier Removal
    print("🧹 Cleaning data and handling outliers...")
    feature_cols = [col for col in df.columns if col not in ['Unnamed: 0', 'studentID', 'averagedCorrectness', 'averagedTimespent']]
    df = remove_outliers_iqr(df, feature_cols)

    # Define X (Features) and Y (Target)
    X = df[feature_cols]
    y = df['averagedCorrectness']

    # 3. Train/Test Split (CRITICAL: Must happen BEFORE scaling to prevent data leakage)
    print("🪓 Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Transform only!

    # 5. Feature Engineering: K-Means Clustering (adding cluster as a feature)
    print("🧬 Engineering features using K-Means Clustering...")
    kmeans = KMeans(n_clusters=2, random_state=42)
    
    # Fit on train, predict on train and test
    train_clusters = kmeans.fit_predict(X_train_scaled).reshape(-1, 1)
    test_clusters = kmeans.predict(X_test_scaled).reshape(-1, 1)
    
    X_train_final = np.hstack((X_train_scaled, train_clusters))
    X_test_final = np.hstack((X_test_scaled, test_clusters))

    # 6. Model Dictionary (Clean way to evaluate multiple models)
    print("🤖 Training and evaluating models...")
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(subsample=0.9, n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42),
        "LightGBM": LGBMRegressor(num_leaves=35, learning_rate=0.05, n_estimators=500, max_depth=7, random_state=42)
    }

    # 7. Training Loop & Results Leaderboard
    results = []
    for name, model in models.items():
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append({"Model": name, "R2 Score": round(r2, 4), "RMSE": round(rmse, 4)})

    # Print a beautiful leaderboard
    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    
    print("\n🏆 Model Performance Leaderboard:")
    print("-" * 50)
    print(results_df.to_string(index=False))
    print("-" * 50)
    print("✅ Pipeline execution complete.")

if __name__ == "__main__":
    main()
