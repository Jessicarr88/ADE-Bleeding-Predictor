import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc)

# Seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=500):
    """Generate realistic synthetic patient data for bleeding risk modeling"""

    data = {
        'encounter_id': [f'ENC{str(i).zfill(4)}' for i in range(n_samples)],
        'patient_id': [f'PT{np.random.randint(1000, 9999)}' for _ in range(n_samples)],
        'age': np.random.normal(72, 12, n_samples).astype(int).clip(18, 100),
        'sex': np.random.choice([0, 1], n_samples),  # 0=Male, 1=Female
        'htn': np.random.binomial(1, 0.65, n_samples),
        'hld': np.random.binomial(1, 0.50, n_samples),
        'ihd': np.random.binomial(1, 0.35, n_samples),
        'hf': np.random.binomial(1, 0.25, n_samples),
        'dm': np.random.binomial(1, 0.40, n_samples),
        'ckd': np.random.binomial(1, 0.30, n_samples),
        'copd': np.random.binomial(1, 0.20, n_samples),
        'afib': np.random.binomial(1, 0.25, n_samples),
        'dementia': np.random.binomial(1, 0.15, n_samples),
        'stroke': np.random.binomial(1, 0.20, n_samples),
        'baseline_anemia': np.random.binomial(1, 0.30, n_samples),
        'liver_disease': np.random.binomial(1, 0.10, n_samples),
        'active_cancer': np.random.binomial(1, 0.12, n_samples),
        'warfarin': np.random.binomial(1, 0.15, n_samples),
        'doac': np.random.binomial(1, 0.20, n_samples),
        'antiplatelet': np.random.binomial(1, 0.35, n_samples),
        'nsaid': np.random.binomial(1, 0.15, n_samples),
        'ssri_snri': np.random.binomial(1, 0.20, n_samples),
        'steroid': np.random.binomial(1, 0.10, n_samples),
        'heparin_lmwh': np.random.binomial(1, 0.20, n_samples),
        'num_home_meds': np.random.poisson(5, n_samples).clip(0, 30),
        'num_bleeding_risk_meds': np.random.poisson(2, n_samples).clip(0, 10),
        'systolic_bp': np.random.normal(138, 18, n_samples).astype(int).clip(60, 220),
        'heart_rate': np.random.normal(78, 15, n_samples).astype(int).clip(30, 200),
        'hemoglobin': np.random.normal(12.5, 1.8, n_samples).clip(5, 20),
        'platelets': np.random.normal(200, 50, n_samples).astype(int).clip(10, 500),
        'inr': np.random.exponential(1.0, n_samples) + 0.8,
        'creatinine': np.random.lognormal(0, 0.5, n_samples).clip(0.3, 10),
        'gfr': np.random.normal(60, 25, n_samples).astype(int).clip(5, 150),
        'ast': np.random.lognormal(3.3, 0.5, n_samples).astype(int),
        'alt': np.random.lognormal(3.1, 0.5, n_samples).astype(int),
        'recent_fall': np.random.binomial(1, 0.10, n_samples),
        'prior_bleed_12mo': np.random.binomial(1, 0.12, n_samples),
        'renal_dose_issue': np.random.binomial(1, 0.15, n_samples),
        'drug_interaction_flag': np.random.binomial(1, 0.20, n_samples),
    }

    df_syn = pd.DataFrame(data)

    # Generate target variable with realistic correlations
    bleeding_prob = (
        0.1 +  # baseline
        (df_syn['warfarin'] * 0.25) +
        (df_syn['inr'] > 2.5) * 0.20 +
        (df_syn['antiplatelet'] * 0.15) +
        (df_syn['hemoglobin'] < 10) * 0.15 +
        (df_syn['prior_bleed_12mo'] * 0.20) +
        (df_syn['num_bleeding_risk_meds'] > 2) * 0.10 +
        (df_syn['age'] > 75) * 0.10 +
        (df_syn['platelets'] < 100) * 0.15 +
        (df_syn['nsaid'] * 0.10) +
        (df_syn['heparin_lmwh'] * 0.15) +
        np.random.normal(0, 0.05, n_samples)  # noise
    ).clip(0.01, 0.85)

    df_syn['bleeding_event'] = (np.random.uniform(0, 1, n_samples) < bleeding_prob).astype(int)

    # Add dates
    dates = pd.date_range('2025-01-01', periods=n_samples, freq='D').to_numpy()
    np.random.shuffle(dates)
    df_syn['admission_date'] = dates

    return df_syn


# Load dataset
print("=" * 70)
print("STEP 1: LOADING AND INSPECTING DATA")
print("=" * 70)

# Try to load original dataset
try:
    df = pd.read_csv('/Users/jessicaramachandran/ade-bleeding-mvp/ade_bleeding_mvp_synthetic_500.csv')
    if len(df) < 50:
        print(f"⚠️  Original dataset has only {len(df)} rows. Generating synthetic expansion...")
        df = generate_synthetic_data(500)
except Exception as e:
    print(f"Could not load original dataset ({e}). Generating synthetic data...")
    df = generate_synthetic_data(500)

print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names ({len(df.columns)} total):")
print(df.columns.tolist())

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

# Examine target variable
print("\n" + "=" * 70)
print("TARGET VARIABLE: bleeding_event")
print("=" * 70)
print(df['bleeding_event'].value_counts())
print(f"Prevalence: {df['bleeding_event'].mean():.2%}")

# =============================
# STEP 2: IDENTIFY PREDICTORS
# =============================
print("\n" + "=" * 70)
print("STEP 2: IDENTIFYING POTENTIAL PREDICTORS")
print("=" * 70)

# Exclude non-predictor columns
exclude_cols = [
    'encounter_id', 'patient_id', 'admission_date',  # IDs and dates
    'bleeding_risk_pct',  # LEAK - synthetic score
    'bleeding_risk_category',  # LEAK - derived from synthetic score
    'bleeding_event',  # target
    'ade_type', 'culprit_medication', 'severity', 'preventability'  # outcomes/metadata
]

predictor_cols = [col for col in df.columns if col not in exclude_cols]
print(f"\nPredictors ({len(predictor_cols)}):")
print(predictor_cols)

# =============================
# STEP 3: DATA PREPROCESSING
# =============================
print("\n" + "=" * 70)
print("STEP 3: DATA PREPROCESSING & CLEANING")
print("=" * 70)

df_processed = df[predictor_cols + ['bleeding_event']].copy()

# Convert sex to numeric (Female=1, Male=0)
if 'sex' in df_processed.columns:
    df_processed['sex'] = (df_processed['sex'] == 'Female').astype(int)
    print("✓ Converted 'sex' to numeric (Female=1, Male=0)")

# Handle missing values
print(f"\nHandling missing values:")
numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_processed[col].isnull().sum() > 0:
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)
        print(f"  ✓ {col}: filled with median ({median_val})")

print(f"\nMissing values after cleaning: {df_processed.isnull().sum().sum()}")

# =============================
# STEP 4: TRAIN/TEST SPLIT
# =============================
print("\n" + "=" * 70)
print("STEP 4: TRAIN/TEST SPLIT")
print("=" * 70)

X = df_processed.drop('bleeding_event', axis=1)
y = df_processed['bleeding_event']

# Use stratified split if enough samples in each class, otherwise use random split
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    # If stratification fails (e.g., small dataset), use simple split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training set bleeding prevalence: {y_train.mean():.2%}")
print(f"Test set bleeding prevalence: {y_test.mean():.2%}")

# =============================
# STEP 5: TRAIN MODEL
# =============================
print("\n" + "=" * 70)
print("STEP 5: TRAINING RANDOM FOREST CLASSIFIER")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)
print("✓ Model trained successfully")

# =============================
# STEP 6: PREDICTIONS
# =============================
print("\n" + "=" * 70)
print("STEP 6: GENERATING PREDICTIONS")
print("=" * 70)

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print(f"✓ Predictions generated for {len(y_pred)} test samples")

# =============================
# STEP 7: EVALUATE MODEL
# =============================
print("\n" + "=" * 70)
print("STEP 7: MODEL EVALUATION METRICS")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nAccuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision:  {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:     {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score:   {f1:.4f}")
print(f"ROC-AUC:    {roc_auc:.4f}")

# Confusion Matrix
print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"  True Negatives:  {cm[0,0]}")
print(f"  False Positives: {cm[0,1]}")
print(f"  False Negatives: {cm[1,0]}")
print(f"  True Positives:  {cm[1,1]}")

# Classification Report
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Bleeding', 'Bleeding']))

# =============================
# STEP 8: FEATURE IMPORTANCE
# =============================
print("\n" + "=" * 70)
print("STEP 8: FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Predictors of Bleeding:")
print(feature_importance.head(15).to_string(index=False))

# Calculate cumulative importance
feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum()
n_features_90 = (feature_importance['cumulative_importance'] <= 0.9).sum() + 1
print(f"\nTop {n_features_90} features explain ~90% of predictions")

# =============================
# STEP 9: SAVE MODEL & DATA
# =============================
print("\n" + "=" * 70)
print("STEP 9: SAVING MODEL AND METADATA")
print("=" * 70)

# Save model
with open('/Users/jessicaramachandran/Downloads/Assignment 2.2-Bayesian Inference/ADE prototype/bleeding_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("✓ Model saved as: bleeding_model.pkl")

# Save feature names for app
feature_names = X.columns.tolist()
with open('/Users/jessicaramachandran/Downloads/Assignment 2.2-Bayesian Inference/ADE prototype/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✓ Feature names saved as: feature_names.pkl")

# Save metrics summary
metrics_summary = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'confusion_matrix': cm.tolist(),
    'feature_importance': feature_importance.to_dict('records'),
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'prevalence': float(y.mean())
}

with open('/Users/jessicaramachandran/Downloads/Assignment 2.2-Bayesian Inference/ADE prototype/model_metrics.pkl', 'wb') as f:
    pickle.dump(metrics_summary, f)
print("✓ Metrics summary saved as: model_metrics.pkl")

# =============================
# STEP 10: VISUALIZATIONS
# =============================
print("\n" + "=" * 70)
print("STEP 10: CREATING VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['No Bleeding', 'Bleeding'],
            yticklabels=['No Bleeding', 'Bleeding'])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')

# Feature Importance (Top 15)
top_15 = feature_importance.head(15)
axes[0, 1].barh(range(len(top_15)), top_15['importance'].values)
axes[0, 1].set_yticks(range(len(top_15)))
axes[0, 1].set_yticklabels(top_15['feature'].values)
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_title('Top 15 Feature Importances')
axes[0, 1].invert_yaxis()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1, 0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Metrics Bar Chart
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
metrics_values = [accuracy, precision, recall, f1, roc_auc]
axes[1, 1].bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Model Performance Metrics')
axes[1, 1].set_ylim([0, 1])
for i, v in enumerate(metrics_values):
    axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('/Users/jessicaramachandran/Downloads/Assignment 2.2-Bayesian Inference/ADE prototype/model_performance.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as: model_performance.png")

print("\n" + "=" * 70)
print("✓ MODEL TRAINING COMPLETE")
print("=" * 70)
print(f"\nFiles created:")
print(f"  1. bleeding_model.pkl (trained model)")
print(f"  2. feature_names.pkl (feature list)")
print(f"  3. model_metrics.pkl (metrics summary)")
print(f"  4. model_performance.png (visualizations)")
print(f"\nReady for deployment with Streamlit app!")
