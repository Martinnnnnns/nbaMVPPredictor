import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """Load and prepare the data for machine learning"""
    print("Loading data...")
    stats = pd.read_csv("player_mvp_stats.csv", index_col=0)
    
    print(f"Dataset shape: {stats.shape}")
    print(f"Missing values:\n{pd.isnull(stats).sum()}")
    
    # Fill missing values with 0
    stats = stats.fillna(0)
    
    return stats

def define_predictors():
    """Define the predictor variables for the model"""
    predictors = [
        "Age", "G", "GS", "MP", "FG", "FGA", 'FG%', '3P', '3PA', '3P%', 
        '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 
        'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'W', 'L', 'W/L%',
        'GB', 'PS/G', 'PA/G', 'SRS'
    ]
    return predictors

def create_ratio_features(stats):
    """Create ratio features normalized by year"""
    print("Creating ratio features...")
    
    # Calculate ratios for key stats relative to yearly averages
    stat_ratios = stats[["PTS", "AST", "STL", "BLK", "3P", "Year"]].groupby("Year").apply(
        lambda x: x / x.mean()
    )
    
    stats[["PTS_R", "AST_R", "STL_R", "BLK_R", "3P_R"]] = stat_ratios[["PTS", "AST", "STL", "BLK", "3P"]]
    
    return stats

def find_ap(combination):
    """Calculate Average Precision for top 5 MVP candidates"""
    actual = combination.sort_values("Share", ascending=False).head(5)
    predicted = combination.sort_values("predictions", ascending=False)
    
    ps = []
    found = 0
    seen = 1
    
    for index, row in predicted.iterrows():
        if row["Player"] in actual["Player"].values:
            found += 1
            ps.append(found / seen)
        seen += 1

    return sum(ps) / len(ps) if ps else 0

def add_ranks(predictions):
    """Add ranking columns to predictions"""
    predictions = predictions.sort_values("predictions", ascending=False)
    predictions["Predicted_Rk"] = list(range(1, predictions.shape[0] + 1))
    predictions = predictions.sort_values("Share", ascending=False)
    predictions["Rk"] = list(range(1, predictions.shape[0] + 1))
    predictions["Diff"] = (predictions["Rk"] - predictions["Predicted_Rk"])
    return predictions

def backtest(stats, model, years, predictors, use_scaler=False):
    """Backtest the model across multiple years"""
    print(f"Backtesting model across {len(years)} years...")
    
    aps = []
    all_predictions = []
    sc = StandardScaler() if use_scaler else None
    
    for year in years:
        train = stats[stats["Year"] < year].copy()
        test = stats[stats["Year"] == year].copy()
        
        if use_scaler:
            sc.fit(train[predictors])
            train_scaled = train.copy()
            test_scaled = test.copy()
            train_scaled[predictors] = sc.transform(train[predictors])
            test_scaled[predictors] = sc.transform(test[predictors])
        else:
            train_scaled = train
            test_scaled = test
        
        model.fit(train_scaled[predictors], train_scaled["Share"])
        predictions = model.predict(test_scaled[predictors])
        predictions_df = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
        
        combination = pd.concat([test[["Player", "Share"]], predictions_df], axis=1)
        combination = add_ranks(combination)
        all_predictions.append(combination)
        aps.append(find_ap(combination))
    
    return sum(aps) / len(aps), aps, pd.concat(all_predictions)

def train_models(stats, predictors):
    """Train and evaluate different models"""
    print("Training models...")
    
    # Define years for backtesting (skip first 5 years for initial training)
    years = list(range(1991, 2022))
    backtest_years = years[5:]  # Start from 1996
    
    results = {}
    
    # Ridge Regression
    print("Training Ridge Regression...")
    reg = Ridge(alpha=0.1)
    mean_ap, aps, all_predictions = backtest(stats, reg, backtest_years, predictors)
    results['Ridge'] = {
        'mean_ap': mean_ap,
        'aps': aps,
        'predictions': all_predictions
    }
    print(f"Ridge Regression Average Precision: {mean_ap:.4f}")
    
    # Ridge with ratio features
    print("Training Ridge Regression with ratio features...")
    predictors_with_ratios = predictors + ["PTS_R", "AST_R", "STL_R", "BLK_R", "3P_R"]
    mean_ap_ratios, aps_ratios, all_predictions_ratios = backtest(
        stats, reg, backtest_years, predictors_with_ratios
    )
    results['Ridge_Ratios'] = {
        'mean_ap': mean_ap_ratios,
        'aps': aps_ratios,
        'predictions': all_predictions_ratios
    }
    print(f"Ridge with Ratios Average Precision: {mean_ap_ratios:.4f}")
    
    # Random Forest (using recent years only due to computational cost)
    print("Training Random Forest...")
    stats['NPos'] = stats['Pos'].astype('category').cat.codes
    stats['NTm'] = stats['Tm'].astype('category').cat.codes
    
    rf = RandomForestRegressor(n_estimators=50, random_state=1, min_samples_split=5)
    rf_years = years[28:]  # Use more recent years for RF
    mean_ap_rf, aps_rf, all_predictions_rf = backtest(
        stats, rf, rf_years, predictors_with_ratios + ["NPos", "NTm"]
    )
    results['RandomForest'] = {
        'mean_ap': mean_ap_rf,
        'aps': aps_rf,
        'predictions': all_predictions_rf
    }
    print(f"Random Forest Average Precision: {mean_ap_rf:.4f}")
    
    return results

def analyze_results(results, stats):
    """Analyze and visualize results"""
    print("\n=== MODEL COMPARISON ===")
    for model_name, result in results.items():
        print(f"{model_name}: {result['mean_ap']:.4f}")
    
    # Show feature importance for Ridge model
    print("\n=== RIDGE REGRESSION ANALYSIS ===")
    predictors = define_predictors()
    reg = Ridge(alpha=0.1)
    reg.fit(stats[predictors], stats["Share"])
    
    # Feature coefficients
    feature_importance = pd.DataFrame({
        'Feature': predictors,
        'Coefficient': reg.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Correlation analysis
    print(f"\nCorrelation with MVP Share:")
    correlations = stats.corr()["Share"].sort_values(ascending=False)
    print(correlations.head(10))
    
    return feature_importance, correlations

def predict_current_year(stats, predictors, year=2021):
    """Make predictions for a specific year"""
    print(f"\n=== PREDICTIONS FOR {year} ===")
    
    train = stats[stats["Year"] < year]
    test = stats[stats["Year"] == year]
    
    # Add ratio features
    predictors_with_ratios = predictors + ["PTS_R", "AST_R", "STL_R", "BLK_R", "3P_R"]
    
    reg = Ridge(alpha=0.1)
    reg.fit(train[predictors_with_ratios], train["Share"])
    
    predictions = reg.predict(test[predictors_with_ratios])
    predictions_df = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
    
    combination = pd.concat([test[["Player", "Share"]], predictions_df], axis=1)
    combination = add_ranks(combination)
    
    # Show top predictions vs actual
    print("Top 10 Predicted vs Actual:")
    top_predictions = combination.sort_values("predictions", ascending=False).head(10)
    print(top_predictions[["Player", "Share", "predictions", "Predicted_Rk", "Rk"]].to_string())
    
    # Calculate MSE
    mse = mean_squared_error(combination["Share"], combination["predictions"])
    print(f"\nMean Squared Error: {mse:.6f}")
    
    return combination

def main():
    """Main function"""
    # Load and prepare data
    stats = load_and_prepare_data()
    
    # Define predictors
    predictors = define_predictors()
    
    # Create ratio features
    stats = create_ratio_features(stats)
    
    # Train models
    results = train_models(stats, predictors)
    
    # Analyze results
    feature_importance, correlations = analyze_results(results, stats)
    
    # Make predictions for most recent year
    current_predictions = predict_current_year(stats, predictors, 2021)
    
    # Save results
    feature_importance.to_csv("feature_importance.csv", index=False)
    correlations.to_csv("correlations.csv")
    current_predictions.to_csv("2021_predictions.csv", index=False)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Results saved:")
    print("- feature_importance.csv: Feature importance from Ridge regression")
    print("- correlations.csv: Correlations with MVP share")
    print("- 2021_predictions.csv: Predictions for 2021 season")

if __name__ == "__main__":
    main()