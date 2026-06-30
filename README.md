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
- installed and setup postgreSQL locally.
- wrote 'src/load_to_db.py' to load the CSV into Postgres, deduplicating redundant trials, thus making it 4803 unique trials. 
- Created 2 queries: one to examine failure rate by trial phase and another to see the top trial sponsors and saved the queries to 'src/queries.sql'
