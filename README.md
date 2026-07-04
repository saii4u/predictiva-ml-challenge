Predictiva Machine Learning Challenge: Pairwise Preference Modeling

1. Problem Formulation
The objective of this task was to predict which of two trading agents (A or B) performed better given a shared market context (OHLCV) and their respective decision traces (Position Mask). Instead of feeding raw, noisy time-series sequences directly into a deep learning model, I reframed this pairwise ranking problem into a standard binary classification task using domain-specific feature engineering.

2. Feature Engineering Strategy
Financial time-series data is inherently noisy. To help the model focus on actual agent performance, I extracted aggregate summary statistics for each sequence:

Realized PnL: Calculated by multiplying the market's log returns by the agent's position mask.

Volatility: The standard deviation of the agent's returns, representing the risk taken.

Sharpe Ratio (Approximate): The risk-adjusted return (Mean PnL / Volatility).

Market Exposure: The percentage of time the agent was actively holding a position.

To capture the relative advantage between the two agents, I calculated the difference between their features (Feature_A - Feature_B). This "Difference Vector" elegantly feeds into a binary classifier, allowing the model to learn the specific thresholds of outperformance that lead to a win.

3. Modeling Approach
I utilized a Random Forest Classifier. Tree-based ensemble models are highly robust to overfitting, require minimal hyperparameter tuning, and excel at capturing non-linear relationships in tabular feature data. By training on the difference vectors, the Random Forest efficiently learned how the engineered performance metrics correlate with the hidden evaluation criterion.

Part 2: The Final Commented Code
This is the exact working code, but fully annotated for the evaluators.

"
Predictiva Machine Learning Challenge - Task 1
Author: Sai Krishna Anumula

This script processes pairwise financial time-series data to predict which
trading agent's strategy is preferred. It extracts key financial metrics 
from raw OHLCV and Position Mask data, calculates the relative differences, 
and trains a Random Forest model to classify the winning agent.
"

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

# Define column names as raw files lack headers
COLUMN_NAMES = ['Open', 'High', 'Low', 'Close', 'Volume', 'Mask']

def compute_features(file_path):
    """
    Reads a single agent's sequence file and extracts core performance metrics.
    
    Args:
        file_path (str): The path to the agent's data file.
        
    Returns:
        pd.Series: A vector of calculated financial features.
    """
    # Raw data is space-separated; using sep=r'\s+' to parse correctly
    df = pd.read_csv(file_path, header=None, names=COLUMN_NAMES, sep=r'\s+')
    
    # Calculate market returns using logarithmic returns for statistical stability
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    
    # Isolate the returns captured strictly when the agent is in the market
    df['agent_pnl'] = df['log_ret'] * df['Mask']
    
    # Calculate risk metric (standard deviation of captured returns)
    volatility = df['agent_pnl'].std()
    
    # Return aggregated features representing the agent's holistic performance
    return pd.Series({
        'total_pnl': df['agent_pnl'].sum(),
        'sharpe': df['agent_pnl'].mean() / (volatility + 1e-9), # 1e-9 prevents division by zero
        'exposure': df['Mask'].mean(),
        'volatility': volatility
    })

def build_dataset(metadata_df, data_folder):
    """
    Iterates through the metadata to build a dataset of pairwise feature differences.
    
    By subtracting Agent B's features from Agent A's features, we transform 
    the pairwise ranking problem into a standard tabular classification problem.
    """
    rows = []
    for _, row in metadata_df.iterrows():
        sample_id = row['id']
        
        # Construct paths for both agents in the pair
        path_a = f"{data_folder}/sample_{sample_id}_a.csv"
        path_b = f"{data_folder}/sample_{sample_id}_b.csv"
        
        # Extract individual performance profiles
        feat_a = compute_features(path_a)
        feat_b = compute_features(path_b)
        
        # Calculate the relative advantage (Difference Vector)
        diff_features = (feat_a - feat_b).to_dict()
        diff_features['id'] = sample_id
        
        # Append target label if this is training data
        if 'label' in row:
            diff_features['label'] = row['label']
            
        rows.append(diff_features)
        
    return pd.DataFrame(rows)

# ==========================================
# EXECUTION BLOCK
# ==========================================

print("Processing Training Data...")
train_metadata = pd.read_csv(r'C:\Users\91739\OneDrive\Desktop\predictiva\predictiva-machine-learning-challange-4\train.csv')
df_train = build_dataset(train_metadata, r'C:\Users\91739\OneDrive\Desktop\predictiva\predictiva-machine-learning-challange-4\train')
print("Training Data Ready! Shape:", df_train.shape)

print("\nProcessing Testing Data...")
test_metadata = pd.read_csv(r'C:\Users\91739\OneDrive\Desktop\predictiva\predictiva-machine-learning-challange-4\test.csv')
df_test = build_dataset(test_metadata, r'C:\Users\91739\OneDrive\Desktop\predictiva\predictiva-machine-learning-challange-4\test')
print("Testing Data Ready! Shape:", df_test.shape)

# ==========================================
# MACHINE LEARNING & SUBMISSION BLOCK
# ==========================================

print("\nTraining the Machine Learning Model...")

# Features derived from our difference vector engineering
features = ['total_pnl', 'sharpe', 'exposure', 'volatility']

X_train = df_train[features]
y_train = df_train['label']

# Initialize Random Forest (chosen for robustness against overfitting on tabular data)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model Training Complete!")

print("\nGenerating Predictions for Test Data...")
X_test = df_test[features]
test_predictions = model.predict(X_test)

# Format final submission file matching the required evaluation format
submission = pd.DataFrame({
    'id': df_test['id'],
    'label': test_predictions
})

# Export to CSV
submission_path = r'C:\Users\91739\OneDrive\Desktop\predictiva\predictiva-machine-learning-challange-4\submission.csv'
submission.to_csv(submission_path, index=False)

print(f"SUCCESS! Submission file saved to: {submission_path}")