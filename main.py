import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Sklearn Preprocessing & Unsupervised Learning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# Sklearn Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

# Advanced Ensembles
from xgboost import XGBRegressor
import lightgbm as lgb

warnings.filterwarnings('ignore')

def generate_eda_plots(df: pd.DataFrame, feature_cols: list, target_col: str):
    """Generates and saves professional EDA charts to disk."""
    print("\n Generating Exploratory Data Analysis (EDA) Visualizations...")
    os.makedirs("assets", exist_ok=True)
    
    # 1. Feature Correlation Heatmap
    plt.figure(figsize=(20, 10), dpi=150)
    corr_matrix = df[feature_cols + [target_col]].corr()
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", cbar=True)
    plt.title("Feature Correlation Matrix", fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig("assets/correlation_matrix.png")
    plt.close()
    
    # 2. Boxplot for outlier detection (First 10 features for clean visualization)
    plt.figure(figsize=(12, 8), dpi=150)
    sns.boxplot(data=df[feature_cols[:10]], orient="h", palette="Set2")
    plt.title("Distribution & Outlier Boundary Verification (Sample Features)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig("assets/outlier_analysis.png")
    plt.close()
    print("  • Saved assets to /assets directory.")


def clean_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
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


def engineer_features(X_train, X_test):
    """
    Applies strict scaling, PCA dimensionality reduction, and K-Means clustering.
    """
    # 1. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. PCA (Keeping 95% of variance)
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # 3. K-Means Clustering (K=2)
    kmeans = KMeans(n_clusters=2, random_state=42)
    train_clusters = kmeans.fit_predict(X_train_pca).reshape(-1, 1)
    test_clusters = kmeans.predict(X_test_pca).reshape(-1, 1)
    
    # Combine PCA features with the new cluster feature
    X_train_final = np.hstack((X_train_pca, train_clusters))
    X_test_final = np.hstack((X_test_pca, test_clusters))
    
    return X_train_final, X_test_final


def evaluate_baselines(X_train, y_train, X_test, y_test):
    """Evaluates a diverse suite of models to establish performance benchmarks."""
    print("\n Phase 1: Evaluating Baseline Models...")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "ElasticNet": ElasticNet(alpha=0.1),
        "KNN Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Linear SVR": LinearSVR(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(subsample=0.9, n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42),
        "LightGBM": lgb.LGBMRegressor(num_leaves=35, learning_rate=0.05, n_estimators=500, max_depth=7, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({"Model": name, "R2 Score": round(r2, 4), "RMSE": round(rmse, 4)})

    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    print("\n Baseline Model Leaderboard:")
    print(results_df.to_string(index=False))
    return results_df


def tune_lightgbm(X_train, y_train, X_test, y_test):
    """Executes RandomizedSearchCV to find optimal parameters for the strongest ensemble."""
    print("\n🔎 Phase 2: Running RandomizedSearchCV for LightGBM Optimization...")
    
    param_distributions = {
        'num_leaves': [20, 30, 40, 50],
        'max_depth': [7, 15, 25, 50],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 500, 800],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0.1, 0.5, 0.9],
        'reg_lambda': [0.1, 0.5, 0.9]
    }
    
    lgbm_random = RandomizedSearchCV(
        estimator=lgb.LGBMRegressor(random_state=42), 
        param_distributions=param_distributions, 
        n_iter=20,
        scoring="r2", 
        cv=5, 
        n_jobs=-1, 
        verbose=1,
        random_state=42
    )
    
    lgbm_random.fit(X_train, y_train)
    best_model = lgbm_random.best_estimator_
    
    print("\n Optimization Results:")
    print(f"  • Best Parameters: {lgbm_random.best_params_}")
    
    y_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    print(f"  • Final Tuned Test Set Performance (R²): {test_r2:.4f}")


def main():
    print(" Starting Production ML Pipeline...")
    
    # 1. Load Data
    data_path = "student_data_processed.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f" Error: Dataset '{data_path}' not found.")
        return

    # 2. Define Features & Clean Outliers
    drop_cols = ['Unnamed: 0', 'studentID', 'averagedCorrectness', 'averagedTimespent']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    df_clean = clean_outliers_iqr(df, feature_cols)
    
    # Optional: Generate EDA Plots
    generate_eda_plots(df_clean, feature_cols, 'averagedCorrectness')

    # 3. Train/Test Split
    X = df_clean[feature_cols]
    y = df_clean['averagedCorrectness']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 4. Feature Engineering Pipeline (PCA + K-Means)
    X_train_final, X_test_final = engineer_features(X_train, X_test)

    # 5. Baseline Evaluation
    evaluate_baselines(X_train_final, y_train, X_test_final, y_test)

    # 6. Hyperparameter Tuning
    tune_lightgbm(X_train_final, y_train, X_test_final, y_test)
    
    print("\n Pipeline completed successfully.")

if __name__ == "__main__":
    main()
