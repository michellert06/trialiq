# Clinical Trial Predictor

## Clinical trials (drug/treatment tests) fail or get abandoned all the time — wrong patients, bad design, funding pulled, side effects, whatever. Pharma companies and investors spend a lot of money trying to predict which trials are likely to succeed or fail before they sink years into them

## This tool is a clinical trial risk and intelligence platform that predicts which trials are likely to fail and answering natural language questions about the trial using a RAG chatbot

---

# Phase 0: Setup
Created the initial repo with a python virtual environment. 

---

# Phase 1: Data Ingestion
- Wrote 'src/fetch_data.py' to fetch data from ClinicalTrials.gov using their public API and included 10 conditions: breast cancer, lung cancer, diabetes, alzheimer, heart failure, depression, rheumatoid arthritis, asthma,covid-19, obesity. This resulted in 5000 raw trials being stored as JSON.
- Wrote 'src/parse_data.py' to flatten the nested JSON into a tabular format saved as data/'processed_trials.csv'. This contains the following 9 columns: nct_id, title, status, phase, study_type, conditions, sponsor, dates.

---

# Phase 2: Data Engineering
- Installed and setup postgreSQL locally.
- Wrote 'src/load_to_db.py' to load the CSV into Postgres, deduplicating redundant trials, thus making it 4803 unique trials. 
- Created 2 queries: one to examine failure rate by trial phase and another to see the top trial sponsors and saved the queries to 'src/queries.sql'


# Phase 3: Machine Learning - Trial Failure Prediction
## Objective: Train a binary classification model to predict whether a given clinical trial will fail(terminated/withdrawn early), or be completed successfully.


This is a binary classification problem with class imbalance — only ~14.7% of trials in the dataset result in failure (TERMINATED or WITHDRAWN). Trials with unresolved statuses (RECRUITING, UNKNOWN, etc.) were excluded since they have no known outcome, leaving 3,103 trials with definitive results for training and evaluation. 
With 85% of trials completing successfully, a naive model that always predicts “completed” would score 85% accuracy while catching zero failures. ROC AUC measures how well the model separates the two classes across all possible decision thresholds — a much more honest metric for imbalanced problems. This is why ROC AUC was chosen over Accuracy

## Feature Engineering
The first model iteration used only 4 basic features (phase, study type, sponsor name, duration) and achieved a ROC AUC of ~0.59 — barely better than random guessing. This revealed the need for richer features. A second parsing pass extracted additional fields from the raw API JSON:

	•	enrollment_count — number of patients enrolled (larger trials carry more risk)
	•	sponsor_class — whether the sponsor is industry, NIH, or academic (funding stability signal)
	•	num_sites — number of trial locations (complexity signal)
	•	num_collaborators — number of partner organizations
	•	fda_regulated_drug — whether the intervention is FDA regulated
	•	accepts_healthy — whether healthy volunteers are accepted
	•	sex — eligibility sex restriction
	•	duration_days — derived from start and completion dates

## Models Trained
Two models were trained and compared using an 80/20 stratified train-test split:

Logistic Regression

	•	A linear baseline model with class_weight="balanced" to handle imbalance
	•	ROC AUC: 0.747
	•	High recall (75%) but low precision (25%) — catches most failures but with many false alarms

Random Forest (selected model)

	•	An ensemble of 200 decision trees with class_weight="balanced"
	•	ROC AUC: 0.854
	•	Precision: 60%, Recall: 66% on the failure class
	•	Overall accuracy: 89%
	•	Significantly outperformed logistic regression on all meaningful metrics


## Tools Used

	•	scikit-learn — model training, pipelines, preprocessing, evaluation
	•	imbalanced-learn — class imbalance handling
	•	joblib — model serialization
	•	SQLAlchemy / pandas — data loading and feature preparation
	•	OneHotEncoder + ColumnTransformer — categorical feature encoding inside a sklearn Pipeline

## Output
the trained Random Forest model is saved to models/trial_failure_model.pkl and will be served via FastAPI in Phase 6


# Phase 4: Deep - Learning using DistilliBERT transformer
## Objective: Add an additional layer of failure predictions using transformer that classifies risk from raw text rather than structured metadata. This tests whether natural language in trial titles and conditions carries any predictive value. 

## Approach
 The approach is instea of hand selecting features like enrollement count and sponsor class, we use a distilliBERT model to read the raw text of each trial's title and learns which language patternscorrelate with trial failure. The transformer converts this into contextual embeddings: numberical representations of these word relationships, which are then passed to a classification head the calculates failure probability. 

## Architecture
• Model: distilbert-base-uncased — a distilled version of BERT, 40% smaller with 97% of BERT’s performance
• Task: Binary sequence classification(success vs fail)
• Input: trial title + conditions, concatenated with a SEP token separator 
• Max sequence length: 128 tokens
• Optimizer: adamW with learning rate 2e-5
• Epochs: 3 epochs

## Class Imbalance
Unlike scikit learn, Pytorch transformers dont have a bult in class_weight parameter. Thus a weighted cross entropy loss was used and it penalized misclassified failure cases about 5 times more than completed cases.

## Findings
The distilliBERT ROC AUC was 0.662. while this was better than the initial random forest model, the finalized model still outperforms this (0.854 ROC AUC). This makes sense in the context of the problem. Text alone is a pretty weak signal to predict whether a clinical trial will be a failure. What actually helps predict failure is structured metadata about enrollment count, sponsor type, number of sites, funding source, etc.

## Tools Used
	•	PyTorch — model training, custom loss function, data loading
	•	Hugging Face Transformers — DistilBERT pretrained weights and tokenizer
	•	CrossEntropyLoss with class weights — imbalance handling
	•	scikit-learn — evaluation metrics (ROC AUC, classification report)
	•	SQLAlchemy / pandas — data loading from PostgreSQL
## Output
	•	Trained model saved to models/distilbert_trial/ for reuse in Phase 5
	•	Random Forest (models/trial_failure_model.pkl) confirmed as the stronger production model


# DEFINITIONS
- epoch: one complete pass of the entire training dataset through the neural network model
- class imbalance: refers to a situation where the number of examples in each class is unevenly distributed. 