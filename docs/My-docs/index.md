---
title: Introduction
sidebar_label: Introduction
---

# Introduction to Fraud Detection


**What is Fraud Detection?**

Fraud detection is the process of identifying fraudulent transactions in real-time before they cause financial damage. Think of it as a security guard for money - every time someone tries to use a credit card, make a transfer, or complete a payment, our system needs to decide in milliseconds: "Is this legitimate or fraudulent?"

**Why Machine Learning?**
```text
IF transaction_amount > $10,000 
AND country = "Nigeria" 
THEN block
```

**Problems with rule-based:**

Fraudsters learn the rules and bypass them

Thousands of manual rules to maintain

Can't adapt to new fraud patterns quickly

**ML-based systems learn patterns automatically:**

Detect complex relationships humans miss

Adapt to new fraud patterns

Reduce false positives (legitimate transactions wrongly blocked)

**Real-World Impact**

In 2023, payment fraud losses reached $38 billion globally. A good ML system can:

Block 80-90% of fraudulent transactions

Reduce false positives by 50-70%

Save millions in chargeback fees

Protect customer trust

This is exactly what we'll build — a full machine learning system that goes from raw transaction data → trained model → real-time fraud scoring.
in the image, the top part shows the old rule-based way (lots of manual work, easy to fool). The bottom part is the modern ML way (automatically learns patterns and keeps improving).

![Alt text](/img/z.png)

#### Understanding the problem

**The Dataset We'll Use**

Credit Card Fraud Detection Dataset (ULB, 2013)

284,807 transactions from European cardholders

492 frauds (only 0.172% of total)

31 columns: Time, V1-V28 (PCA transformed), Amount, Class

**The Challenge: Extreme Class Imbalance**

Genuine transactions: 284,315 (99.828%)
Fraudulent transactions: 492 (0.172%)

This imbalance is the fundamental challenge of fraud detection. If you predict "genuine" for every transaction, you're 99.8% accurate but completely useless - you'd miss all fraud!

**Real Constraints**

(1).**Speed:** Must score in `<100ms` (customer won't wait)

(2). **Interpretability:** Must explain WHY a transaction was flagged

(3). **Adaptability:** Fraud patterns change constantly

(4). **Regulatory:** Must comply with banking regulations

#### Complete walkthrough

**Environment setup**

```python
# Create isolated environment (prevents package conflicts)
conda create -n fraud_detection python=3.11 -y
conda activate fraud_detection

# Install essential libraries
pip install pandas numpy scikit-learn matplotlib seaborn
pip install lightgbm xgboost imbalanced-learn
pip install shap jupyter fastapi uvicorn joblib
```
**What each library does:**

**pandas/numpy:** Data manipulation (like Excel on steroids)

**scikit-learn:** Machine learning algorithms and tools

**matplotlib:** Create visualizations (understand your data)

**lightgbm**: State-of-the-art ML algorithms

**imbalanced-learn:** Handle the fraud/genuine imbalance

**shap:** Explain why model flagged transactions

**uvicorn:** Turn model into web service

**joblib:** load trained models

**Data loading and exploration**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('creditcard.csv')

# 1. BASIC INFO
print("Dataset Shape:", df.shape)  # (284807, 31)
print("\nColumns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# 2. TARGET DISTRIBUTION (The Imbalance)
print("\nClass Distribution:")
print(df['Class'].value_counts())
print(f"Fraud Rate: {df['Class'].mean()*100:.4f}%")

# Visualize imbalance
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar plot
sns.countplot(x='Class', data=df, ax=axes[0])
axes[0].set_title('Transaction Distribution')
axes[0].set_xticklabels(['Genuine', 'Fraud'])

# Pie chart
df['Class'].value_counts().plot.pie(
    autopct='%1.2f%%', 
    labels=['Genuine', 'Fraud'],
    ax=axes[1]
)
axes[1].set_title('Class Distribution')

plt.tight_layout()
plt.show()

# 3. STATISTICAL SUMMARY
print("\nAmount Statistics by Class:")
print(df.groupby('Class')['Amount'].describe())

# 4. TIME ANALYSIS
df['Hour'] = (df['Time'] / 3600) % 24  # Convert seconds to hour of day

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(data=df[df['Class']==0], x='Hour', bins=24, alpha=0.5, label='Genuine')
sns.histplot(data=df[df['Class']==1], x='Hour', bins=24, alpha=0.5, label='Fraud')
plt.legend()
plt.title('Transaction Time Distribution')

plt.subplot(1, 2, 2)
sns.boxplot(x='Class', y='Amount', data=df[df['Amount'] < 1000])  # Cap at 1000 for visibility
plt.xticks([0, 1], ['Genuine', 'Fraud'])
plt.title('Transaction Amount Distribution')

plt.tight_layout()
plt.show()

# 5. CORRELATION ANALYSIS
plt.figure(figsize=(16, 12))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, cmap='RdBu_r', center=0, 
            annot=False, fmt='.2f', square=True)
plt.title('Feature Correlations')
plt.show()

# Features most correlated with fraud
fraud_correlations = correlation_matrix['Class'].sort_values(ascending=False)
print("\nTop 10 Features Correlated with Fraud:")
print(fraud_correlations[1:11])  # Skip Class itself
```

**What we can learn from this:**

(a). Data is massively imbalanced (0.17% fraud)

(b). V1-V28 are anonymized (PCA transformed for privacy)

(c). Fraud transactions tend to be smaller amounts (hiding in plain sight)

(d). Fraud happens more at certain hours (when people sleep)

(e). Features have different scales (Amount vs PCA components)

**Data preprocessing**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop(['Class', 'Hour'], axis=1)  # Drop target and derived features
y = df['Class']

# FEATURE ENGINEERING
# 1. Scale Amount (important for some models)
scaler = StandardScaler()
X['Scaled_Amount'] = scaler.fit_transform(X[['Amount']])
X = X.drop('Amount', axis=1)  # Replace original with scaled

# 2. Create time-based features
X['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
X['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)

# 3. Add interaction features (domain knowledge)
X['Amount_V1_interaction'] = X['Scaled_Amount'] * X['V1']
X['Amount_V2_interaction'] = X['Scaled_Amount'] * X['V2']

# 4. Create velocity features (simulated - in real data you'd use historical)
# Here we simulate: transactions per hour for this card
X['Transaction_velocity'] = df.groupby('Time')['Time'].transform('count')

# TRAIN-TEST SPLIT (Critical: no data leakage!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # 20% for testing
    random_state=42,         # Reproducibility
    stratify=y,              # Maintain fraud ratio
    shuffle=True             # Randomize order
)

print(f"Train set: {X_train.shape}, Fraud rate: {y_train.mean():.4f}")
print(f"Test set: {X_test.shape}, Fraud rate: {y_test.mean():.4f}")
```
**Why preprocessing matters:**

**Scaling:** Prevents features with large values from dominating

**Time encoding:** Captures cyclical nature of time (23:00 and 01:00 are close)

**Velocity:** Fraud often involves many transactions in short time

**No leakage:** Test set must be completely unseen during training

 **Handling Imbalanced Data**

 ```python
 from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Approach 1: Class Weights (Recommended for trees)
# =================================================
# Penalize misclassifying fraud more heavily
model_rf_weighted = RandomForestClassifier(
    n_estimators=200,
    class_weight={0: 1, 1: 100},  # Fraud is 100x more important
    max_depth=10,                   # Prevent overfitting
    min_samples_split=5,            # Minimum samples to split node
    min_samples_leaf=2,             # Minimum samples in leaf
    random_state=42,
    n_jobs=-1                       # Use all CPU cores
)

# Approach 2: SMOTE (Synthetic Minority Oversampling)
# ====================================================
# Create synthetic fraud examples
smote = SMOTE(random_state=42, sampling_strategy=0.1)  # Make fraud 10% of training

# Create pipeline
pipeline_smote = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ))
])

# Approach 3: Balanced Random Forest (Built-in sampling)
# =======================================================
model_rf_balanced = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced_subsample',  # Balance each bootstrap sample
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Approach 4: LightGBM with scale_pos_weight (Best in 2026)
# ==========================================================
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
# scale_pos_weight ≈ 580 (genuine:fraqraud ratio)

model_lgb = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    scale_pos_weight=scale_pos_weight,  # Weight positive class
    subsample=0.8,                       # Use 80% of data per tree
    colsample_bytree=0.8,                 # Use 80% of features per tree
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Train all models to compare
models = {
    'Weighted RF': model_rf_weighted,
    'SMOTE Pipeline': pipeline_smote,
    'Balanced RF': model_rf_balanced,
    'LightGBM': model_lgb
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if name != 'SMOTE Pipeline':  # Pipeline needs fit, not separate
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train)  # Pipeline handles SMOTE internally
        pred = model.predict(X_test)
    
    results[name] = pred
    
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, pred, 
          target_names=['Genuine', 'Fraud']))
```
***Model evaluation and results**

Now let's evaluate our models properly. For imbalanced fraud detection, accuracy is misleading - we need precision, recall, and F1-score.

```python
from sklearn.metrics import (precision_recall_curve, average_precision_score, 
                            confusion_matrix, ConfusionMatrixDisplay, roc_auc_score)

def evaluate_models(models, X_test, y_test):
    """
    Comprehensive model evaluation with business-focused metrics
    """
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {name}")
        print('='*50)
        
        # Get predictions
        if name == 'SMOTE Pipeline':
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                     display_labels=['Genuine', 'Fraud'])
        disp.plot()
        plt.title(f'Confusion Matrix - {name}')
        plt.show()
        
        # 2. Key Metrics
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'model': name,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': 2 * (tp / (tp + fp) * tp / (tp + fn)) / 
                       ((tp / (tp + fp)) + (tp / (tp + fn))) if (tp + fp) > 0 and (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn),
            'false_negative_rate': fn / (fn + tp),
            'avg_precision': average_precision_score(y_test, y_proba),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # 3. Business Impact Calculations (assuming €100 avg fraud amount)
        avg_fraud_amount = 100
        total_frauds = len(y_test[y_test==1])
        
        metrics['business_impact'] = {
            'frauds_caught': tp,
            'frauds_missed': fn,
            'money_saved': tp * avg_fraud_amount,
            'money_lost': fn * avg_fraud_amount,
            'false_alarms': fp,
            'customers_inconvenienced': fp
        }
        
        results[name] = metrics
        
        # Print results
        print(f"\n📊 Performance Metrics for {name}:")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
        print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Avg Precision (PR-AUC): {metrics['avg_precision']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\n💰 Business Impact (test set, €100 avg fraud):")
        print(f"  Frauds caught: {metrics['business_impact']['frauds_caught']} (Saved: €{metrics['business_impact']['money_saved']:,.0f})")
        print(f"  Frauds missed: {metrics['business_impact']['frauds_missed']} (Lost: €{metrics['business_impact']['money_lost']:,.0f})")
        print(f"  False alarms:  {metrics['business_impact']['false_alarms']} (Customers inconvenienced)")
    
    return results

# Run evaluation
evaluation_results = evaluate_models(models, X_test, y_test)

# Compare models visually
def plot_model_comparison(results):
    """Create comparison bar charts for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = list(results.keys())
    
    # Precision comparison
    precisions = [results[m]['precision'] for m in models]
    axes[0, 0].bar(models, precisions, color='skyblue')
    axes[0, 0].set_title('Precision by Model', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_ylim([0, 1])
    for i, v in enumerate(precisions):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Recall comparison
    recalls = [results[m]['recall'] for m in models]
    axes[0, 1].bar(models, recalls, color='lightcoral')
    axes[0, 1].set_title('Recall by Model', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_ylim([0, 1])
    for i, v in enumerate(recalls):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # F1 comparison
    f1_scores = [results[m]['f1_score'] for m in models]
    axes[1, 0].bar(models, f1_scores, color='lightgreen')
    axes[1, 0].set_title('F1-Score by Model', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate(f1_scores):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # PR-AUC comparison (best metric for imbalanced data)
    pr_aucs = [results[m]['avg_precision'] for m in models]
    axes[1, 1].bar(models, pr_aucs, color='gold')
    axes[1, 1].set_title('PR-AUC by Model', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('PR-AUC')
    axes[1, 1].set_ylim([0, 1])
    for i, v in enumerate(pr_aucs):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.show()

plot_model_comparison(evaluation_results)

# Winner: LightGBM typically performs best for fraud detection
print("\n🏆 Recommended Model: LightGBM")
print("   - Best PR-AUC (handles imbalance naturally)")
print("   - Fastest predictions (<10ms)")
print("   - Built-in handling of missing values")
print("   - Native support for class weights")
```


```markdown
**Model Interpretability**

```python
import shap
import warnings
warnings.filterwarnings('ignore')

# Why SHAP? Banks need explanations!
# "Why did you block my transaction?" must be answerable

def explain_predictions(model, X_train, X_test, sample_size=100):
    """
    Create SHAP explanations for model predictions
    """
    # Create explainer (TreeExplainer for tree-based models)
    explainer = shap.TreeExplainer(model)
    
    # Get SHAP values for test sample
    X_test_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_test_sample)
    
    # For binary classification, shap_values is list of two arrays
    # We want the values for fraud class (class 1)
    if isinstance(shap_values, list):
        shap_values_fraud = shap_values[1]
    else:
        shap_values_fraud = shap_values
    
    # 1. Global feature importance
    plt.figure(figsize=(12, 6))
    
    # Summary plot (shows feature impact across all predictions)
    shap.summary_plot(
        shap_values_fraud, 
        X_test_sample,
        plot_type="dot",
        max_display=20,
        show=False
    )
    plt.title('SHAP Feature Importance (Global)')
    plt.tight_layout()
    plt.show()
    
    # 2. Bar plot of mean absolute SHAP values
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_fraud, 
        X_test_sample,
        plot_type="bar",
        max_display=20,
        show=False
    )
    plt.title('Mean |SHAP| Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # 3. Waterfall plot for a single high-fraud prediction
    high_fraud_idx = np.argsort(-model.predict_proba(X_test_sample)[:, 1])[0]
    
    plt.figure(figsize=(12, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_fraud[high_fraud_idx],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_test_sample.iloc[high_fraud_idx].values,
            feature_names=X_test_sample.columns.tolist()
        ),
        show=False,
        max_display=15
    )
    plt.title(f'Why Transaction #{high_fraud_idx} Was Flagged as Fraud')
    plt.tight_layout()
    plt.show()
    
    # 4. Dependence plots (how feature affects prediction)
    top_features = np.argsort(np.abs(shap_values_fraud).mean(0))[-3:]
    
    for feat_idx in top_features:
        feature = X_test_sample.columns[feat_idx]
        plt.figure(figsize=(10, 4))
        shap.dependence_plot(
            feat_idx, 
            shap_values_fraud, 
            X_test_sample,
            display_features=X_test_sample,
            show=False
        )
        plt.title(f'SHAP Dependence: {feature}')
        plt.tight_layout()
        plt.show()
    
    return explainer, shap_values_fraud

# Run explanations
explainer, shap_values = explain_predictions(
    model_lgb, 
    X_train, 
    X_test, 
    sample_size=1000
)

# Create a simple explanation function for API
def explain_transaction(transaction_dict, model, explainer, scaler):
    """
    Generate human-readable explanation for a transaction
    """
    # Convert to DataFrame and preprocess
    df_trans = pd.DataFrame([transaction_dict])
    
    # Apply same preprocessing as training
    df_trans['Scaled_Amount'] = scaler.transform(df_trans[['Amount']])
    df_trans = df_trans.drop('Amount', axis=1)
    
    # Add time features
    hour = (df_trans['Time'] / 3600) % 24
    df_trans['Hour_sin'] = np.sin(2 * np.pi * hour/24)
    df_trans['Hour_cos'] = np.cos(2 * np.pi * hour/24)
    
    # Get SHAP values for this transaction
    shap_val = explainer.shap_values(df_trans)
    if isinstance(shap_val, list):
        shap_val = shap_val[1]  # Fraud class
    
    # Get top 3 reasons
    feature_names = df_trans.columns
    shap_importance = np.abs(shap_val[0])
    top_indices = np.argsort(shap_importance)[-3:][::-1]
    
    reasons = []
    for idx in top_indices:
        feature = feature_names[idx]
        value = df_trans.iloc[0][feature]
        impact = shap_val[0][idx]
        
        if impact > 0:
            direction = "increases"
            reason = f"{feature} = {value:.4f} (strongly increases fraud risk)"
        else:
            direction = "decreases"
            reason = f"{feature} = {value:.4f} (decreases fraud risk)"
        
        reasons.append(reason)
    
    fraud_prob = model.predict_proba(df_trans)[0, 1]
    
    explanation = {
        'fraud_probability': float(fraud_prob),
        'risk_factors': reasons,
        'action': 'BLOCK' if fraud_prob > 0.8 else 'REVIEW' if fraud_prob > 0.3 else 'APPROVE'
    }
    
    return explanation

print("\nSample explanation:")
sample_transaction = {
    'Time': 3600 * 3,  # 3 AM
    'Amount': 500.00,
    'V1': -1.2,
    'V2': 0.5,
    # ... other V features would be here
}
for i in range(3, 29):
    sample_transaction[f'V{i}'] = 0

explanation = explain_transaction(sample_transaction, model_lgb, explainer, scaler)
for key, value in explanation.items():
    print(f"{key}: {value}")
```
**Model Serialization and Deployment Preparation**

```python
import joblib
import json
from datetime import datetime
import os

def save_model_artifacts(model, scaler, feature_names, metrics, path='models/'):
    """
    Save all necessary components for deployment
    """
    # Create directory if doesn't exist
    os.makedirs(path, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Save model
    model_path = f"{path}/fraud_model_{timestamp}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")
    
    # 2. Save scaler
    scaler_path = f"{path}/scaler_{timestamp}.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")
    
    # 3. Save feature names
    features_path = f"{path}/features_{timestamp}.json"
    with open(features_path, 'w') as f:
        json.dump(list(feature_names), f)
    print(f"Features saved: {features_path}")
    
    # 4. Save model metadata
    metadata = {
        'model_type': type(model).__name__,
        'training_date': timestamp,
        'features': list(feature_names),
        'metrics': metrics,
        'threshold': metrics.get('best_cost_threshold', 0.5),
        'n_features': len(feature_names)
    }
    
    metadata_path = f"{path}/metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")
    
    # 5. Save a sample configuration for API
    config = {
        'model_path': model_path,
        'scaler_path': scaler_path,
        'features_path': features_path,
        'thresholds': {
            'approve': 0.3,  # Below this, auto-approve
            'review': 0.8,   # Between 0.3-0.8, manual review
            'block': 0.8     # Above this, auto-block
        }
    }
    
    config_path = f"{path}/deploy_config_{timestamp}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Deployment config saved: {config_path}")
    
    return {
        'model_path': model_path,
        'scaler_path': scaler_path,
        'features_path': features_path,
        'metadata_path': metadata_path,
        'config_path': config_path
    }

# Prepare metrics dict
metrics = {
    'best_cost_threshold': float(eval_results['best_cost_threshold']),
    'pr_auc': float(eval_results['pr_auc']),
    'avg_precision': float(eval_results['avg_precision']),
    'roc_auc': float(eval_results['roc_auc']),
    'min_cost': float(eval_results['min_cost'])
}

# Save everything
saved_paths = save_model_artifacts(
    model_lgb, 
    scaler, 
    X_train.columns,
    metrics
)

# Save a lightweight version for fast inference
def save_lightweight_model(model, path='models/lightweight/'):
    """
    Save model in multiple formats for different deployment scenarios
    """
    os.makedirs(path, exist_ok=True)
    
    # 1. Joblib format (fastest for Python)
    joblib.dump(model, f"{path}/model.joblib")
    
    # 2. Try saving as ONNX for cross-platform deployment
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        # This is simplified - real ONNX conversion needs proper types
        initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
        onx = convert_sklearn(model, initial_types=initial_type)
        
        with open(f"{path}/model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        print("ONNX model saved")
    except:
        print("ONNX conversion skipped (install skl2onnx if needed)")
    
    # 3. Save model parameters as JSON (for inspection)
    if hasattr(model, 'get_params'):
        params = model.get_params()
        with open(f"{path}/model_params.json", 'w') as f:
            json.dump({k: str(v) for k, v in params.items()}, f, indent=2)

save_lightweight_model(model_lgb)
```
**Model monitoring and retraining strategy**

Fraud patterns evolve constantly. A model that works today may fail tomorrow. Here's a production monitoring system:

```python
def create_monitoring_dashboard():
    """
    Track model performance in production
    """
    monitoring_metrics = {
        'daily': {
            'transactions_processed': 0,
            'avg_response_time_ms': 0,
            'fraud_rate': 0.0,
            'block_rate': 0.0,
            'review_rate': 0.0
        },
        'drift_detection': {
            'feature_drift_scores': {},  # PSI for each feature
            'concept_drift': 0.0,        # Model performance degradation
            'last_retrain_date': None
        },
        'business_metrics': {
            'chargeback_rate': 0.0,
            'customer_complaints': 0,
            'false_positive_cost': 0.0
        }
    }
    return monitoring_metrics

### 2. Drift Detection System

import scipy.stats as stats
from datetime import datetime, timedelta

def calculate_psi(expected_distribution, actual_distribution, buckets=10):
    """
    Population Stability Index - measures feature drift
    PSI < 0.1: No drift
    PSI 0.1-0.2: Minor drift (monitor)
    PSI > 0.2: Major drift (retrain needed)
    """
    psi_value = 0
    for i in range(buckets):
        expected_perc = expected_distribution[i]
        actual_perc = actual_distribution[i]
        
        # Avoid division by zero
        if expected_perc == 0:
            expected_perc = 0.0001
        if actual_perc == 0:
            actual_perc = 0.0001
            
        psi_value += (actual_perc - expected_perc) * np.log(actual_perc / expected_perc)
    
    return psi_value

def monitor_feature_drift(reference_data, current_data, features):
    """
    Track drift for important features
    """
    drift_report = {}
    
    for feature in features:
        # Create distributions
        ref_hist, bins = np.histogram(reference_data[feature], bins=10)
        curr_hist, _ = np.histogram(current_data[feature], bins=bins)
        
        # Convert to percentages
        ref_pct = ref_hist / len(reference_data)
        curr_pct = curr_hist / len(current_data)
        
        # Calculate PSI
        psi = calculate_psi(ref_pct, curr_pct)
        
        drift_report[feature] = {
            'psi': psi,
            'drift_level': 'MAJOR' if psi > 0.2 else 'MINOR' if psi > 0.1 else 'NONE',
            'action': 'RETRAIN' if psi > 0.2 else 'MONITOR' if psi > 0.1 else 'OK'
        }
    
    return drift_report

### 3. Automated Retraining Pipeline

class FraudModelRetrainer:
    """
    Automated model retraining with validation
    """
    def __init__(self, model_path, performance_threshold=0.8):
        self.model_path = model_path
        self.performance_threshold = performance_threshold
        self.retrain_history = []
    
    def should_retrain(self, current_metrics, days_since_train):
        """
        Decision logic for when to retrain
        """
        reasons = []
        
        # Time-based retraining (monthly)
        if days_since_train > 30:
            reasons.append("Monthly scheduled retraining")
        
        # Performance degradation
        if current_metrics['avg_precision'] < self.performance_threshold:
            reasons.append(f"Performance dropped: {current_metrics['avg_precision']:.3f} < {self.performance_threshold}")
        
        # Feature drift
        if current_metrics.get('max_feature_drift', 0) > 0.2:
            reasons.append(f"Feature drift detected: PSI={current_metrics['max_feature_drift']:.2f}")
        
        # Business metric degradation
        if current_metrics.get('false_positive_rate', 0) > 0.05:
            reasons.append(f"False positives too high: {current_metrics['false_positive_rate']:.2%}")
        
        return {
            'retrain': len(reasons) > 0,
            'reasons': reasons
        }
    
    def retrain_pipeline(self, new_data, new_labels, feature_names):
        """
        Execute retraining with validation
        """
        print("🔄 Starting model retraining...")
        
        # Split new data
        X_new_train, X_new_val, y_new_train, y_new_val = train_test_split(
            new_data, new_labels, test_size=0.2, stratify=new_labels, random_state=42
        )
        
        # Train new model
        scale_pos_weight = len(y_new_train[y_new_train==0]) / len(y_new_train[y_new_train==1])
        
        new_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        
        new_model.fit(
            X_new_train, y_new_train,
            eval_set=[(X_new_val, y_new_val)],
            callbacks=[lgb.early_stopping(50)]
        )
        
        # Validate new model
        val_pred = new_model.predict(X_new_val)
        val_proba = new_model.predict_proba(X_new_val)[:, 1]
        
        new_performance = {
            'precision': precision_score(y_new_val, val_pred),
            'recall': recall_score(y_new_val, val_pred),
            'avg_precision': average_precision_score(y_new_val, val_proba)
        }
        
        # Compare with current model
        current_model = joblib.load(f"{self.model_path}/latest_model.pkl")
        current_pred = current_model.predict(X_new_val)
        current_performance = {
            'precision': precision_score(y_new_val, current_pred),
            'recall': recall_score(y_new_val, current_pred),
            'avg_precision': average_precision_score(y_new_val, current_model.predict_proba(X_new_val)[:, 1])
        }
        
        # Decide whether to deploy
        performance_improvement = new_performance['avg_precision'] - current_performance['avg_precision']
        
        retrain_record = {
            'timestamp': datetime.now(),
            'new_model_performance': new_performance,
            'current_model_performance': current_performance,
            'improvement': performance_improvement,
            'deployed': performance_improvement > 0.01  # Deploy if >1% improvement
        }
        
        if performance_improvement > 0.01:
            # Save new model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"{self.model_path}/fraud_model_{timestamp}.pkl"
            joblib.dump(new_model, model_path)
            
            # Update symlink or config to point to new model
            with open(f"{self.model_path}/latest_model.txt", 'w') as f:
                f.write(f"fraud_model_{timestamp}.pkl")
            
            print(f"✅ New model deployed! Improvement: {performance_improvement:.2%}")
        else:
            print(f"⏸️ Keeping current model. Improvement insufficient: {performance_improvement:.2%}")
        
        self.retrain_history.append(retrain_record)
        return retrain_record

### 4. Alerting System

def create_alerts_config():
    """
    Configure alerts for model issues
    """
    alerts = {
        'critical': {
            'response_time_above_100ms': 'slack #fraud-alerts-critical',
            'fraud_rate_drop_50%': 'pagerduty',
            'api_error_rate_above_5%': 'email cto@company.com'
        },
        'warning': {
            'feature_drift_above_0.15': 'slack #fraud-alerts',
            'false_positives_up_20%': 'email fraud-team@company.com',
            'model_accuracy_drop': 'jira ticket'
        },
        'info': {
            'daily_report': 'email dashboard@company.com',
            'weekly_retraining_summary': 'confluence page'
        }
    }
    return alerts

### 5. Production Monitoring Script

def production_monitoring_loop():
    """
    Continuous monitoring (run every hour)
    """
    while True:
        try:
            # 1. Collect recent predictions (last 24h)
            recent_data = get_recent_transactions(hours=24)
            reference_data = get_training_data()  # Original training set
            
            # 2. Check feature drift
            important_features = ['Scaled_Amount', 'V1', 'V2', 'V3', 'Hour_sin']
            drift_report = monitor_feature_drift(reference_data, recent_data, important_features)
            
            # 3. Check performance (if labels available)
            if has_labels(recent_data):
                performance = calculate_performance(recent_data)
                
                # 4. Decision: retrain?
                retrainer = FraudModelRetrainer(model_path='models/')
                decision = retrainer.should_retrain(
                    current_metrics=performance,
                    days_since_train=get_days_since_last_train()
                )
                
                if decision['retrain']:
                    print(f"🚨 Retraining triggered: {decision['reasons']}")
                    retrainer.retrain_pipeline(recent_data, recent_data['Class'], recent_data.columns)
            
            # 5. Send alerts if needed
            check_alerts(drift_report, performance)
            
            # 6. Sleep for 1 hour
            time.sleep(3600)
            
        except Exception as e:
            print(f"Monitoring error: {e}")
            time.sleep(300)  # Retry in 5 minutes
```


**Understanding key concepts**

**Understanding Class Imbalance**

**Why it's a problem:**

Most ML algorithms assume balanced classes. With 99.8% genuine transactions, a model can achieve 99.8% accuracy by simply predicting "genuine" for everything.

**solution**

1. **Classweights**

```python
# How class weights work
weights = {
    0: 1.0,  # Genuine: normal weight
    1: 100.0 # Fraud: 100x more important
}
# Model penalizes fraud misclassification 100x more
```
2. **Synthetic Minority Over-sampling**

```python
# SMOTE creates synthetic fraud examples by:
# 1. Pick a fraud transaction
# 2. Find its k nearest neighbors (also fraud)
# 3. Create new sample by interpolating between them
# 
# Example: If fraud A = [1,2,3] and fraud B = [2,3,4]
# New sample = [1.5, 2.5, 3.5]
```
3. **Undersampling**
   
```python
# Randomly sample genuine transactions to match fraud count
genuine_sample = genuine.sample(n=len(fraud), random_state=42)
balanced_data = pd.concat([genuine_sample, fraud])
# Problem: Discards 99.6% of genuine data!
```
**Evolution metrics**

**Why not accurate**

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)

With 99.8% genuine:
- Predict all genuine: 99.8% accuracy
- But catches 0% fraud!
```

***Precision**

```text
Precision = TP / (TP + FP)
Question: Of transactions flagged as fraud, how many were actually fraud?
Business impact: Low precision = annoying customers (false blocks)
```
**Recall:**

```text
Recall = TP / (TP + FN)
Question: Of actual fraud transactions, how many did we catch?
Business impact: Low recall = losing money (missed fraud)
```
**F1 Score**

```text
F1 = 2 * (Precision * Recall) / (Precision + Recall)
Harmonic mean of precision and recall
```

**PR-AUC (Precision-Recall Area Under Curve):**

Best metric for imbalanced data

Shows trade-off between precision and recall at all thresholds

Random model = PR-AUC = fraud rate (0.0017)

Perfect model = PR-AUC = 1.0

**Threshold Selection Strategies**

```python
def select_threshold_business_context(y_true, y_pred_proba, context):
    """
    Select threshold based on business context
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    if context == 'high_risk':
        # Banks losing lots of money → maximize recall
        target_recall = 0.95
        idx = np.argmin(np.abs(recall[:-1] - target_recall))
        return thresholds[idx]
    
    elif context == 'customer_sensitive':
        # Premium customers → maximize precision
        target_precision = 0.99
        idx = np.argmin(np.abs(precision[:-1] - target_precision))
        return thresholds[idx]
    
    elif context == 'balanced':
        # Standard scenario → maximize F1
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        return thresholds[np.argmax(f1_scores)]
```

**Feature Engineering**

**Time-based features**

```python
# Problem: Time is cyclical (23:00 and 01:00 are close numerically)
# Solution: Use sine/cosine transformation

def create_cyclical_features(hour):
    """
    Convert linear hour to cyclical features
    """
    radians = 2 * np.pi * hour / 24
    return np.sin(radians), np.cos(radians)

# Now 23:00 (sin= -0.5, cos=0.87) and 01:00 (sin=0.26, cos=0.97) are close!
```
***Velocity features**

```python
def create_velocity_features(transactions, time_window=3600):
    """
    Count transactions in last hour for each transaction
    """
    velocities = []
    for i, trans_time in enumerate(transactions['Time']):
        # Count transactions in last hour
        count = len(transactions[
            (transactions['Time'] > trans_time - time_window) & 
```


