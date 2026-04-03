import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
import warnings
from sklearn.metrics import roc_curve
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             precision_score, recall_score,
                             accuracy_score)
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp
import seaborn as sns
import os
os.makedirs("outputs", exist_ok=True)

# PHASE 2 — LOAD DATA

df = pd.read_csv("data/application_train.csv")

# PHASE 3 — CLEAN DATA

thresh = len(df) * 0.6
df = df.dropna(thresh=int(thresh), axis=1)

df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

# PHASE 3B — FEATURE ENGINEERING

df['CREDIT_INCOME_RATIO']   = df['AMT_CREDIT']  / (df['AMT_INCOME_TOTAL'] + 1)
df['ANNUITY_INCOME_RATIO']  = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
df['CREDIT_TERM']           = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1)
df['AGE_YEARS']             = df['DAYS_BIRTH'] / -365

# PHASE 4 — TRAIN / TEST SPLIT

TARGET = 'TARGET'
DROP   = ['SK_ID_CURR', TARGET]
X = df.drop(columns=DROP, errors='ignore')
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Class imbalance weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_weight = neg / pos

# PHASE 4B — TRAIN LIGHTGBM

model = lgb.LGBMClassifier(
    n_estimators     = 500,
    learning_rate    = 0.05,
    max_depth        = 6,
    num_leaves       = 31,
    scale_pos_weight = scale_weight,
    random_state     = 42,
    verbose          = -1
)

model.fit(
    X_train, y_train,
    eval_set  = [(X_test, y_test)],
    callbacks = [lgb.early_stopping(50, verbose=False)]
)

# PHASE 5 — EVALUATION

y_prob = model.predict_proba(X_test)[:, 1]

threshold = float(np.percentile(y_prob, 85))
y_pred    = (y_prob >= threshold).astype(int)

print(f"\nThreshold : {threshold:.4f}")
print(f"Prob range: min={y_prob.min():.4f} "
      f"max={y_prob.max():.4f} "
      f"mean={y_prob.mean():.4f}")

auc            = roc_auc_score(y_test, y_prob)
gini           = 2 * auc - 1
ks_stat, _     = ks_2samp(y_prob[y_test == 1],
                           y_prob[y_test == 0])

print("\n" + "=" * 45)
print(f"  AUC-ROC   : {auc:.4f}")
print(f"  Gini      : {gini:.4f}   ← finance standard")
print(f"  KS Stat   : {ks_stat:.4f}   ← finance standard")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
print("=" * 45)

# ── Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Repaid', 'Default'],
            yticklabels=['Repaid', 'Default'])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.show()

# PHASE 6 — SHAP EXPLAINABILITY

print("\nComputing SHAP values (takes 1-2 mins)...")

X_sample = X_test.sample(1000, random_state=42)

explainer = shap.TreeExplainer(
    model,
    feature_perturbation = 'interventional',
    model_output         = 'probability',
    data                 = X_sample
)

shap_values = explainer(X_sample)

# Global SHAP Summary Plot

plt.figure()
shap.summary_plot(shap_values, X_sample,
                  max_display=15, show=False)
plt.title("SHAP Global Feature Importance")
plt.tight_layout()
plt.savefig("outputs/shap_global.png", bbox_inches='tight')
plt.show()

# Global Bar Plot

plt.figure()
shap.summary_plot(shap_values, X_sample,
                  plot_type='bar',
                  max_display=15, show=False)
plt.title("Top 15 Features by Mean SHAP Value")
plt.tight_layout()
plt.savefig("outputs/shap_bar.png", bbox_inches='tight')
plt.show()

# PHASE 6B — PER CUSTOMER EXPLANATION + REASON CODES

reason_map = {
    'EXT_SOURCE_2'         : 'External credit score is low',
    'EXT_SOURCE_3'         : 'External credit score is low',
    'EXT_SOURCE_1'         : 'External credit score is low',
    'CREDIT_INCOME_RATIO'  : 'Loan amount is too high vs income',
    'ANNUITY_INCOME_RATIO' : 'Monthly repayment too high vs income',
    'DAYS_BIRTH'           : 'Customer age is a risk factor',
    'AGE_YEARS'            : 'Customer age is a risk factor',
    'DAYS_EMPLOYED'        : 'Employment duration is insufficient',
    'DAYS_EMPLOYED_PERCENT': 'Employment ratio is low',
    'AMT_CREDIT'           : 'Credit amount is too high',
    'AMT_INCOME_TOTAL'     : 'Income level is below threshold',
    'DAYS_ID_PUBLISH'      : 'Identity documents are outdated',
    'CREDIT_TERM'          : 'Loan term ratio is unfavourable',
    'CODE_GENDER'          : 'Demographic risk factor detected',
    'NAME_EDUCATION_TYPE'  : 'Education level is a risk factor',
    'DEF_60_CNT_SOCIAL_CIRCLE': 'High defaults in social circle',
}

def explain_customer(customer_index):
    customer = X_test.iloc[[customer_index]]
    prob     = model.predict_proba(customer)[0][1]
    decision = "REJECTED" if prob >= threshold else "APPROVED"

    print(f"\n{'='*52}")
    print(f"  Customer #{customer_index}")
    print(f"  Default Probability : {prob:.4f} "
          f"({prob*100:.2f} out of 100)")
    print(f"  Threshold           : {threshold:.4f}")
    print(f"  Decision            : {decision}")
    print(f"{'='*52}")

    # SHAP waterfall chart
    shap_exp = explainer(customer)
    shap.waterfall_plot(
        shap_exp[0],
        max_display = 10,
        show        = False
    )
    plt.title(f"Customer #{customer_index} — Why {decision}?")
    plt.tight_layout()
    plt.savefig(f"outputs/shap_customer_{customer_index}.png",
                bbox_inches='tight')
    plt.show()

    # Reason codes
    sv = shap_exp[0].values
    feature_impacts = pd.DataFrame({
        'Feature'    : X_test.columns,
        'SHAP_Value' : sv,
        'Feature_Val': customer.values[0]
    })

    if decision == "REJECTED":
        reasons = (feature_impacts
                   .sort_values('SHAP_Value', ascending=False)
                   .head(3))
        print("\n  REJECTION REASON CODES:")
        print("  " + "-" * 46)
        for i, row in enumerate(reasons.itertuples(), 1):
            label = reason_map.get(
                row.Feature,
                f"{row.Feature} is outside acceptable range"
            )
            print(f"  R{i}. {label}")
            print(f"       Feature : {row.Feature}")
            print(f"       Value   : {row.Feature_Val:.3f}")
            print(f"       Impact  : +{row.SHAP_Value:.5f} "
                  f"(increases default risk)")
    else:
        reasons = (feature_impacts
                   .sort_values('SHAP_Value', ascending=True)
                   .head(3))
        print("\n  APPROVAL REASON CODES:")
        print("  " + "-" * 46)
        for i, row in enumerate(reasons.itertuples(), 1):
            label = reason_map.get(
                row.Feature,
                f"{row.Feature} supports repayment"
            )
            print(f"  A{i}. {label}")
            print(f"       Feature : {row.Feature}")
            print(f"       Value   : {row.Feature_Val:.3f}")
            print(f"       Impact  : {row.SHAP_Value:.5f} "
                  f"(reduces default risk)")

    return decision, prob

# Run for 4 customers
print("\nExplaining individual customers...")
for idx in [0, 5, 10, 20]:
    explain_customer(idx)

# PHASE 6C — RISK SCORING & SEGMENTATION

all_probs   = model.predict_proba(X_test)[:, 1]
risk_scores = (all_probs * 100).round(1)

# Percentile-based thresholds so segments are always balanced
high_thresh   = float(np.percentile(all_probs, 85))  # top 15%
medium_thresh = float(np.percentile(all_probs, 50))  # middle 35%

def segment(score):
    p = score / 100
    if p >= high_thresh:     return "High Risk"
    elif p >= medium_thresh: return "Medium Risk"
    else:                    return "Low Risk"

results = pd.DataFrame({
    "Risk_Score"   : risk_scores,
    "Risk_Segment" : [segment(s) for s in risk_scores],
    "Decision"     : ["REJECTED" if s / 100 >= threshold
                      else "APPROVED" for s in risk_scores]
})

print("\n" + "=" * 45)
print("  Risk Segment Distribution:")
print(results['Risk_Segment'].value_counts().to_string())
print("\n  Decision Distribution:")
print(results['Decision'].value_counts().to_string())
print("=" * 45)

# Pie Chart
seg_counts = results['Risk_Segment'].value_counts()

color_map   = {
    "High Risk"   : "#e74c3c",
    "Medium Risk" : "#f39c12",
    "Low Risk"    : "#27ae60"
}
colors = [color_map[s] for s in seg_counts.index]

plt.figure(figsize=(6, 6))
plt.pie(seg_counts,
        labels     = seg_counts.index,
        autopct    = '%1.1f%%',
        colors     = colors,
        startangle = 90,
        wedgeprops = {'edgecolor': 'white', 'linewidth': 1.5})
plt.title("Customer Risk Segments", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/risk_segments.png")
plt.show()
print("Risk segment pie chart saved.")

# Bar Chart — Risk Score Distribution

plt.figure(figsize=(8, 4))
plt.hist(risk_scores, bins=50,
         color='steelblue', edgecolor='white')
plt.axvline(high_thresh * 100,
            color='red', linestyle='--',
            label=f'High Risk threshold ({high_thresh*100:.1f})')
plt.axvline(medium_thresh * 100,
            color='orange', linestyle='--',
            label=f'Medium Risk threshold ({medium_thresh*100:.1f})')
plt.title("Risk Score Distribution Across All Customers")
plt.xlabel("Risk Score (0-100)")
plt.ylabel("Number of Customers")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/risk_distribution.png")
plt.show()
print("Risk distribution chart saved.")

# ROC Curve

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--')  # random model line
plt.fill_between(fpr, tpr, alpha=0.3)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/roc_curve.png")
plt.show()
