# Fraud Detection in Job Offers (EMSCAD)

An end-to-end machine learning pipeline to detect **fraudulent job postings** using NLP feature engineering, hand-crafted fraud signals, and gradient boosting — with SMOTE oversampling, Bayesian hyperparameter tuning (Optuna), and isotonic probability calibration.

##  Dataset

[Real / Fake Job Posting Prediction (EMSCAD)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **17,880** job postings, ~**5% fraudulent** (~866 fake)
- Stratified 80/20 split — encoders fit on train only, test set used once for final evaluation

##  Pipeline

```
Raw Data → Stratified Split → Text Cleaning → Feature Engineering → Encoding → Modeling → Calibration
                ↑                                    ↑                              ↑
          No leakage                          30 binary fraud             SMOTE + Optuna +
          (split first)                       signal features             Isotonic calibration
```

### Feature Engineering (30 binary fraud signals)
| Group | Examples |
|-------|----------|
| **Textual (12)** | Spam words, money mentions, non-org email domains, consecutive punctuation, bold HTML |
| **Structural (13)** | Missing company profile, short description, no logo, no screening questions |
| **Contextual (5)** | Previously-seen-as-fraudulent company profile, US-based, low education requirement |

### Text Preprocessing
- **ML pipeline** (XGBoost): HTML stripping → lemmatization → stopword removal → TF-IDF ready
- **DL pipeline** (future): light cleaning preserving punctuation for transformer models

##  Models Compared

| Model | Threshold | Precision (Fraud) | Recall (Fraud) | F1 | ROC-AUC |
|-------|-----------|-------------------|----------------|------|---------|
| Naive Bayes | 0.44 | 0.671 | 0.613 | 0.641 | 0.974 |
| RIPPER (rule-based) | 0.03 | 0.273 | 0.971 | 0.426 | 0.963 |
| **XGBoost baseline** | 0.50 | 0.576 | **0.942** | 0.715 | **0.990** |
| **XGBoost (tuned threshold)** | 0.89 | 0.928 | 0.746 | **0.827** | **0.990** |
| XGBoost + SMOTE | 0.89 | 0.943 | 0.671 | 0.784 | 0.988 |
| **Tuned + Calibrated** | 0.89 | **0.982** | 0.624 | 0.763 | 0.988 |

> **Finding:** SMOTE + Optuna tuning + isotonic calibration dramatically improved precision from 0.576 → 0.982 (near-zero false alarms), at the cost of reduced recall (0.942 → 0.624). The validation-selected threshold (0.89) provides the best F1 balance at 0.827.

## Most Important Features (XGBoost)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `hasCompanyProfile` | 0.207 |
| 2 | `previouslySeenAsFraudulent` | 0.187 |
| 3 | `hasCompanyLogo` | 0.101 |
| 4 | `hasCompanyName` | 0.069 |
| 5 | `required_education_Bachelor's Degree` | 0.030 |

##  Tools & Libraries

```
Python · XGBoost · Scikit-learn · imbalanced-learn (SMOTE) · Optuna
NLTK · BeautifulSoup · wittgenstein (RIPPER) · Pandas · NumPy
Matplotlib · Seaborn · WordCloud
```

## Limitations
- Single dataset (EMSCAD) — no external validation
- Only ~173 fraud cases in the test set
- No deep learning models tested yet (DistilBERT pipeline prepared but not run)

##  Links
- [Kaggle Notebook](https://www.kaggle.com/code/mohamedelhani/fraud-detection-in-job-offers) — full interactive version with all outputs and visualisations

---

## ─── Below: LinkedIn copy-paste content ───

### LinkedIn Project Description (copy this)

```
Built an end-to-end ML pipeline to detect fraudulent job postings from 17,880+
listings (EMSCAD dataset, ~5% fraud rate).

Pipeline:
• Engineered 30 binary fraud-signal features (textual, structural, contextual)
• Compared XGBoost, Naive Bayes, and RIPPER (interpretable rule learner)
• Applied SMOTE oversampling + Bayesian hyperparameter tuning (Optuna)
• Calibrated probabilities with isotonic regression

Results (held-out test set):
• XGBoost ROC-AUC: 0.990
• Tuned + calibrated precision: 0.982 (near-zero false alarms)
• Best F1 score: 0.827 at optimised threshold

Tools: Python, XGBoost, Scikit-learn, SMOTE, Optuna, NLTK, BeautifulSoup
```

### LinkedIn Post (copy this)

```
🔍 Built a machine learning system to detect fake job postings!

With job scams on the rise, I built an end-to-end fraud detection pipeline
using the EMSCAD dataset (17,880 job postings, ~5% fraudulent).

What makes this project different:
🔹 30 hand-crafted binary fraud signals (spam words, missing company info,
   non-org email domains, suspiciously short descriptions)
🔹 Compared 3 model families: XGBoost, Naive Bayes, and RIPPER (rule-based)
🔹 SMOTE oversampling + Optuna Bayesian tuning + isotonic calibration
🔹 Leak-free pipeline: stratified split before ANY preprocessing

Results:
📊 ROC-AUC: 0.990
📊 Precision jumped from 0.58 → 0.98 after tuning (near-zero false alarms)
📊 Top feature? Simply having a company profile — fraudsters skip it 

Full notebook → [Kaggle link]
Code → [GitHub link]

#MachineLearning #NLP #FraudDetection #Python #XGBoost #DataScience
```

