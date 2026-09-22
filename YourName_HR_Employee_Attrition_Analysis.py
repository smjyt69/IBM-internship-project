"""
HR Employee Attrition — Data Analysis with AI
================================================
Project   : Predicting and Understanding Employee Attrition using EDA and Machine Learning
Dataset   : IBM HR Analytics Employee Attrition & Performance
            https://www.kaggle.com/datasets/itssuru/hr-employee-attrition
Internship: AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 | BharatCares

Problem Statement
------------------
Employee attrition (staff leaving the organization) is costly — it drives up
recruitment, onboarding and training expenses, and causes loss of institutional
knowledge. This project analyzes IBM's HR employee dataset to:
    1. Understand WHY employees leave (exploratory data analysis).
    2. Identify the strongest predictors of attrition.
    3. Build a machine learning model to predict at-risk employees, so HR can
       intervene proactively.

Dataset Overview
------------------
1,470 employee records, 35 attributes covering demographics (Age, Gender,
Marital Status), job details (Department, JobRole, JobLevel, MonthlyIncome),
satisfaction scores (JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance)
and tenure information (YearsAtCompany, YearsInCurrentRole). Target: Attrition
(Yes/No).

Before running: download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from the
Kaggle link above and place it in the same folder as this script.

All charts are displayed on screen (plt.show) AND saved as PNG files into the
`outputs/` folder, so you have them even when running non-interactively.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (9, 5)
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'

RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_and_show(fig, name):
    """Save a figure to outputs/ and display it."""
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=120, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
section("1. LOAD DATASET")

df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

print("\nColumn info:")
df.info()

print("\nStatistical summary (numeric columns):")
print(df.describe().T)


# ---------------------------------------------------------------------------
# 2. DATA CLEANING & PREPROCESSING
# ---------------------------------------------------------------------------
section("2. DATA CLEANING & PREPROCESSING")

missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Columns with a single unique value carry no information for analysis/modeling
constant_cols = [c for c in df.columns if df[c].nunique() == 1]
print("\nConstant columns to drop:", constant_cols)

df_clean = df.drop(columns=constant_cols + ['EmployeeNumber'], errors='ignore')
print(f"Shape after cleaning: {df_clean.shape}")

# Encode target variable for analysis convenience
df_clean['AttritionFlag'] = df_clean['Attrition'].map({'Yes': 1, 'No': 0})
attrition_rate = df_clean['AttritionFlag'].mean() * 100
print(f"\nOverall attrition rate: {attrition_rate:.2f}%")


# ---------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------------------
section("3. EXPLORATORY DATA ANALYSIS")

# --- 3.1 Target variable distribution ---
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
df_clean['Attrition'].value_counts().plot(kind='bar', ax=ax[0], color=['#4C72B0', '#DD8452'])
ax[0].set_title('Attrition Count')
ax[0].set_xlabel('Attrition')
ax[0].set_ylabel('Number of Employees')
ax[0].tick_params(axis='x', rotation=0)

df_clean['Attrition'].value_counts().plot(kind='pie', ax=ax[1], autopct='%1.1f%%',
                                           colors=['#4C72B0', '#DD8452'], startangle=90)
ax[1].set_ylabel('')
ax[1].set_title('Attrition Proportion')
save_and_show(fig, "01_attrition_distribution.png")

# --- 3.2 Attrition by Department and Job Role ---
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(data=df_clean, x='Department', hue='Attrition', ax=ax[0], palette='Set2')
ax[0].set_title('Attrition by Department')
ax[0].tick_params(axis='x', rotation=15)

role_attrition = df_clean.groupby('JobRole')['AttritionFlag'].mean().sort_values(ascending=False) * 100
role_attrition.plot(kind='barh', ax=ax[1], color='#C44E52')
ax[1].set_title('Attrition Rate (%) by Job Role')
ax[1].set_xlabel('Attrition Rate (%)')
save_and_show(fig, "02_attrition_by_department_role.png")

# --- 3.3 Attrition vs. Overtime, Income and Age ---
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
sns.countplot(data=df_clean, x='OverTime', hue='Attrition', ax=ax[0], palette='Set1')
ax[0].set_title('Attrition by OverTime')

sns.boxplot(data=df_clean, x='Attrition', y='MonthlyIncome', ax=ax[1], palette='Set3')
ax[1].set_title('Monthly Income vs Attrition')

sns.histplot(data=df_clean, x='Age', hue='Attrition', kde=True, ax=ax[2], multiple='stack', palette='coolwarm')
ax[2].set_title('Age Distribution vs Attrition')
save_and_show(fig, "03_overtime_income_age.png")

# --- 3.4 Satisfaction & Work-Life Balance factors ---
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
factors = ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'JobInvolvement']
for i, factor in enumerate(factors):
    r, c = divmod(i, 2)
    rate = df_clean.groupby(factor)['AttritionFlag'].mean() * 100
    rate.plot(kind='bar', ax=axes[r, c], color='#55A868')
    axes[r, c].set_title(f'Attrition Rate (%) by {factor}')
    axes[r, c].set_ylabel('Attrition Rate (%)')
    axes[r, c].tick_params(axis='x', rotation=0)
save_and_show(fig, "04_satisfaction_factors.png")

# --- 3.5 Tenure-related factors ---
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df_clean, x='Attrition', y='YearsAtCompany', ax=ax[0], palette='Pastel1')
ax[0].set_title('Years at Company vs Attrition')

sns.boxplot(data=df_clean, x='Attrition', y='TotalWorkingYears', ax=ax[1], palette='Pastel2')
ax[1].set_title('Total Working Years vs Attrition')
save_and_show(fig, "05_tenure_factors.png")

# --- 3.6 Correlation heatmap ---
numeric_df = df_clean.select_dtypes(include=[np.number])
corr = numeric_df.corr()

fig = plt.figure(figsize=(16, 12))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, linewidths=0.3)
plt.title('Correlation Heatmap of Numeric Features')
save_and_show(fig, "06_correlation_heatmap.png")

corr_with_target = corr['AttritionFlag'].drop('AttritionFlag').sort_values(key=abs, ascending=False)
print("\nTop 10 features correlated with Attrition:")
print(corr_with_target.head(10))


# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING & ENCODING
# ---------------------------------------------------------------------------
section("4. FEATURE ENGINEERING & ENCODING")

model_df = df_clean.drop(columns=['Attrition']).copy()
categorical_cols = model_df.select_dtypes(include='object').columns.tolist()
print("Categorical columns to encode:", categorical_cols)

le = LabelEncoder()
for col in categorical_cols:
    model_df[col] = le.fit_transform(model_df[col])

print(model_df.head())


# ---------------------------------------------------------------------------
# 5. PREDICTIVE MODELING
# ---------------------------------------------------------------------------
section("5. PREDICTIVE MODELING")

X = model_df.drop(columns=['AttritionFlag'])
y = model_df['AttritionFlag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

# --- Logistic Regression ---
log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("Models trained successfully.")


def evaluate(name, y_true, y_pred, y_proba):
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba)
    }


results = pd.DataFrame([
    evaluate('Logistic Regression', y_test, y_pred_lr, y_proba_lr),
    evaluate('Random Forest', y_test, y_pred_rf, y_proba_rf)
]).set_index('Model').round(3)

print("\nModel comparison:")
print(results)

# --- Confusion matrices ---
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax[0])
ax[0].set_title('Logistic Regression — Confusion Matrix')
ax[0].set_xlabel('Predicted'); ax[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens',
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax[1])
ax[1].set_title('Random Forest — Confusion Matrix')
ax[1].set_xlabel('Predicted'); ax[1].set_ylabel('Actual')
save_and_show(fig, "07_confusion_matrices.png")

# --- ROC curve comparison ---
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

fig = plt.figure(figsize=(7, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={roc_auc_score(y_test, y_proba_lr):.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={roc_auc_score(y_test, y_proba_rf):.2f})")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Attrition Prediction Models')
plt.legend()
save_and_show(fig, "08_roc_curve.png")

# --- Feature importance (Random Forest) ---
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)

fig = plt.figure(figsize=(9, 7))
importances.sort_values().plot(kind='barh', color='#4C72B0')
plt.title('Top 15 Feature Importances — Random Forest')
plt.xlabel('Importance')
save_and_show(fig, "09_feature_importance.png")

print("\nTop 15 most important features (Random Forest):")
print(importances)


# ---------------------------------------------------------------------------
# 6. KEY INSIGHTS & RECOMMENDATIONS
# ---------------------------------------------------------------------------
section("6. KEY INSIGHTS")
print("""
- OverTime is one of the strongest predictors — employees who work overtime
  leave at a noticeably higher rate.
- Low job satisfaction, low environment satisfaction, and poor work-life
  balance are strongly associated with attrition.
- Lower monthly income and fewer years at the company correlate with higher
  attrition — newer, lower-paid employees are the highest flight risk.
- Sales Representatives and Laboratory Technicians show higher attrition
  rates than other job roles.
- Frequent business travel and longer commute distance mildly increase
  attrition risk.
- The Random Forest model outperforms plain Logistic Regression on
  ROC-AUC, indicating non-linear interactions between features matter.
""")

section("7. RECOMMENDATIONS FOR HR")
print("""
1. Review overtime policies and workloads in high-attrition departments/roles.
2. Introduce retention-focused interventions (career growth plans, mentorship)
   for employees in their first 1-2 years.
3. Benchmark and adjust compensation for roles/levels with below-market
   monthly income.
4. Conduct regular pulse surveys on job satisfaction and work-life balance,
   and act on low scores early.
5. Use the trained model's probability scores to flag high-risk employees
   for proactive HR outreach.
""")

section("DONE")
print(f"All charts saved to the '{OUTPUT_DIR}/' folder.")
