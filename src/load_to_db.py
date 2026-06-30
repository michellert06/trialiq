import pandas as pd
from db import SessionLocal, Trial, init_db

def load_csv_to_db(filepath="data/processed_trials.csv"):
    df = pd.read_csv(filepath)
    df = df.where(pd.notnull(df), None)  # convert NaN to None for SQL

    init_db()  # make sure table exists
    session = SessionLocal()

    count = 0
    for _, row in df.iterrows():
        trial = Trial(
            nct_id=row["nct_id"],
            title=row["title"],
            status=row["status"],
            phase=row["phase"],
            study_type=row["study_type"],
            conditions=row["conditions"],
            lead_sponsor=row["lead_sponsor"],
            start_date=row["start_date"],
            completion_date=row["completion_date"],
        )
        session.merge(trial)  # merge = insert or update, avoids duplicate key errors
        count += 1

    session.commit()
    session.close()
    print(f"Loaded {count} trials into the database.")

if __name__ == "__main__":
    load_csv_to_db()
