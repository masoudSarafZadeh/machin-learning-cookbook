import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Sklearn preprocessing & selection
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ML Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Removes outliers using the Interquartile Range (IQR) method and imputes with the mean."""
    df_cleaned = df.copy()
    for col in columns:
        q1, q3 = np.percentile(df_cleaned[col].dropna(), [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        df_cleaned[col] = np.where((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound), np.nan, df_cleaned[col])
        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
    return df_cleaned


def run_hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """
    Executes Randomized Search to optimize RandomForest 
    hyperparameters across search space.
    """
    print("\n Phase 2: Running RandomizedSearchCV for RandomForest Optimization...")
    
    param_distributions = {
        'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
        'max_features': ['sqrt', 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42), 
        param_distributions=param_distributions, 
        n_iter=20,
        scoring="r2", 
        cv=5, 
        n_jobs=-1, 
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    print("\n Optimization Results:")
    print(f"  • Best Parameters: {random_search.best_params_}")
    print(f"  • Best Cross-Validation Score (R²): {random_search.best_score_:.4f}")
    
    # Evaluate optimized model on the test set
    y_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    print(f"  • Final Test Set Performance (R²): {test_r2:.4f}")


def main():
    print(" Starting Production ML Pipeline...")
    
    # 1. Load Data
    data_path = "student_data_processed.csv" 
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f" Error: Dataset '{data_path}' not found.")
        return

    # 2. Preprocessing & Outlier Handling
    feature_cols = [col for col in df.columns if col not in ['Unnamed: 0', 'studentID', 'averagedCorrectness', 'averagedTimespent']]
    df = remove_outliers_iqr(df, feature_cols)

    X = df[feature_cols]
    y = df['averagedCorrectness']

    # 3. Train/Test Split (Prevents data leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Feature Engineering (K-Means Clustering)
    kmeans = KMeans(n_clusters=2, random_state=42)
    train_clusters = kmeans.fit_predict(X_train_scaled).reshape(-1, 1)
    test_clusters = kmeans.predict(X_test_scaled).reshape(-1, 1)
    
    X_train_final = np.hstack((X_train_scaled, train_clusters))
    X_test_final = np.hstack((X_test_scaled, test_clusters))

    # 6. Phase 1: Baseline Comparison
    print("\n Phase 1: Evaluating Baseline Models...")
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Random Forest (Default)": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(subsample=0.9, n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42),
        "LightGBM": LGBMRegressor(num_leaves=35, learning_rate=0.05, n_estimators=500, max_depth=7, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({"Model": name, "R2 Score": round(r2, 4), "RMSE": round(rmse, 4)})

    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    print("\n Baseline Model Leaderboard:")
    print(results_df.to_string(index=False))

    # 7. Phase 2: Hyperparameter Tuning Execution
    run_hyperparameter_tuning(X_train_final, y_train, X_test_final, y_test)
    
    print("\n Entire pipeline completed successfully.")

if __name__ == "__main__":
    main()
