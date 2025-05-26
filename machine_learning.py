import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the data for machine learning"""
    print("Loading player MVP stats data...")
    stats = pd.read_csv("player_mvp_stats.csv", index_col=0)
    
    print(f"Dataset shape: {stats.shape}")
    print(f"Years available: {stats['Year'].min()} to {stats['Year'].max()}")
    
    # Check for missing values
    missing_summary = pd.isnull(stats).sum()
    print(f"Columns with missing values: {missing_summary[missing_summary > 0].shape[0]}")
    
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
    ratio_stats = ["PTS", "AST", "STL", "BLK", "3P"]
    
    for stat in ratio_stats:
        yearly_means = stats.groupby("Year")[stat].mean()
        stats[f"{stat}_R"] = stats.apply(lambda row: row[stat] / yearly_means[row["Year"]] if yearly_means[row["Year"]] != 0 else 0, axis=1)
    
    print("Ratio features created successfully")
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
        
        if test.empty:
            continue
            
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
    
    return sum(aps) / len(aps), aps, pd.concat(all_predictions) if all_predictions else pd.DataFrame()

def train_models(stats, predictors):
    """Train and evaluate different models"""
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    # Get available years
    available_years = sorted(stats['Year'].unique())
    print(f"Available years: {available_years}")
    
    # Define years for backtesting (skip first 5 years for initial training)
    backtest_years = [year for year in available_years if year >= 1996]
    print(f"Backtesting years: {backtest_years}")
    
    results = {}
    
    # Ridge Regression
    print("\n1. Training Ridge Regression...")
    reg = Ridge(alpha=0.1)
    mean_ap, aps, all_predictions = backtest(stats, reg, backtest_years, predictors)
    results['Ridge'] = {
        'mean_ap': mean_ap,
        'aps': aps,
        'predictions': all_predictions
    }
    print(f"Ridge Regression Average Precision: {mean_ap:.4f}")
    
    # Ridge with ratio features
    print("\n2. Training Ridge Regression with ratio features...")
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
    
    # Random Forest (using recent years only)
    print("\n3. Training Random Forest...")
    
    # Create categorical encodings
    stats['NPos'] = stats['Pos'].astype('category').cat.codes
    stats['NTm'] = stats['Tm'].astype('category').cat.codes
    
    rf = RandomForestRegressor(n_estimators=50, random_state=1, min_samples_split=5)
    rf_years = [year for year in available_years if year >= 2010]  # Use recent years for RF
    
    if len(rf_years) > 0:
        print(f"Random Forest years: {rf_years}")
        mean_ap_rf, aps_rf, all_predictions_rf = backtest(
            stats, rf, rf_years, predictors_with_ratios + ["NPos", "NTm"]
        )
        results['RandomForest'] = {
            'mean_ap': mean_ap_rf,
            'aps': aps_rf,
            'predictions': all_predictions_rf
        }
        print(f"Random Forest Average Precision: {mean_ap_rf:.4f}")
    else:
        print("Not enough recent years for Random Forest")
    
    return results

def analyze_results(results, stats):
    """Analyze and visualize results"""
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    for model_name, result in results.items():
        print(f"{model_name}: {result['mean_ap']:.4f}")
    
    # Show feature importance for Ridge model
    print("\n" + "="*50)
    print("RIDGE REGRESSION ANALYSIS")
    print("="*50)
    predictors = define_predictors()
    reg = Ridge(alpha=0.1)
    reg.fit(stats[predictors], stats["Share"])
    
    # Feature coefficients
    feature_importance = pd.DataFrame({
        'Feature': predictors,
        'Coefficient': reg.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Correlation analysis
    print(f"\nTop 10 Correlations with MVP Share:")
    correlations = stats.corr()["Share"].sort_values(ascending=False)
    print(correlations.head(10).to_string())
    
    return feature_importance, correlations

def predict_multiple_years(stats, predictors, years_to_predict):
    """Make predictions for multiple years"""
    print("\n" + "="*60)
    print(f"MAKING PREDICTIONS FOR YEARS: {years_to_predict}")
    print("="*60)
    
    # Add ratio features
    predictors_with_ratios = predictors + ["PTS_R", "AST_R", "STL_R", "BLK_R", "3P_R"]
    
    reg = Ridge(alpha=0.1)
    all_year_predictions = {}
    
    for year in years_to_predict:
        if year not in stats['Year'].values:
            print(f"\n❌ No data available for {year}")
            continue
            
        print(f"\n📊 Predictions for {year}")
        print("-" * 40)
        
        train = stats[stats["Year"] < year]
        test = stats[stats["Year"] == year]
        
        if train.empty:
            print(f"❌ No training data available for {year}")
            continue
        
        reg.fit(train[predictors_with_ratios], train["Share"])
        
        predictions = reg.predict(test[predictors_with_ratios])
        predictions_df = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
        
        combination = pd.concat([test[["Player", "Share"]], predictions_df], axis=1)
        combination = add_ranks(combination)
        
        # Show top predictions vs actual
        print("Top 10 Predicted vs Actual:")
        top_predictions = combination.sort_values("predictions", ascending=False).head(10)
        display_cols = ["Player", "Share", "predictions", "Predicted_Rk"]
        if "Rk" in top_predictions.columns:
            display_cols.append("Rk")
        
        # Format for better display
        display_df = top_predictions[display_cols].copy()
        display_df["Share"] = display_df["Share"].round(3)
        display_df["predictions"] = display_df["predictions"].round(3)
        print(display_df.to_string(index=False))
        
        # Calculate metrics
        mse = mean_squared_error(combination["Share"], combination["predictions"])
        print(f"\n📈 Mean Squared Error: {mse:.6f}")
        
        # Calculate AP if there are actual MVP votes
        if combination["Share"].sum() > 0:
            ap = find_ap(combination)
            print(f"🎯 Average Precision: {ap:.4f}")
        
        # Show actual MVP winner if available
        actual_winner = combination[combination["Share"] == combination["Share"].max()]
        if not actual_winner.empty and actual_winner["Share"].iloc[0] > 0:
            winner = actual_winner.iloc[0]
            print(f"🏆 Actual MVP: {winner['Player']} (Share: {winner['Share']:.3f})")
            print(f"🤖 Our prediction rank for MVP: #{int(winner['Predicted_Rk'])}")
        
        all_year_predictions[year] = combination
    
    return all_year_predictions

def main():
    """Main function"""
    print("="*70)
    print("NBA MVP PREDICTION MODEL - MACHINE LEARNING PIPELINE")
    print("="*70)
    
    # Load and prepare data
    stats = load_and_prepare_data()
    
    # Define predictors
    predictors = define_predictors()
    print(f"Using {len(predictors)} predictor variables")
    
    # Create ratio features
    stats = create_ratio_features(stats)
    
    # Train models
    results = train_models(stats, predictors)
    
    # Analyze results
    feature_importance, correlations = analyze_results(results, stats)
    
    # Make predictions for recent years (2021-2024)
    available_years = sorted(stats['Year'].unique())
    recent_years = [year for year in [2021, 2022, 2023, 2024] if year in available_years]
    
    if recent_years:
        predictions = predict_multiple_years(stats, predictors, recent_years)
        
        # Save results
        print("\n" + "="*50)
        print("SAVING RESULTS")
        print("="*50)
        
        feature_importance.to_csv("feature_importance.csv", index=False)
        correlations.to_csv("correlations.csv")
        
        # Save predictions for each year
        for year, pred_df in predictions.items():
            filename = f"predictions_{year}.csv"
            pred_df.to_csv(filename, index=False)
            print(f"Saved predictions for {year} to {filename}")
        
        print("\n✅ Analysis complete! Files saved:")
        print("   - feature_importance.csv: Feature importance from Ridge regression")
        print("   - correlations.csv: Correlations with MVP share")
        print("   - predictions_[year].csv: Predictions for each year")
    else:
        print("\n❌ No recent years (2021-2024) found in dataset")
        print("Available years:", available_years)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()