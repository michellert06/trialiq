import pandas as pd
from db import SessionLocal, Trial, init_db
import math

def clean(value):
    """Convert NaN/float NaN to None for SQL compatibility."""
    if value is None:
        return None
    try:
        if math.isnan(float(value)):
            return None
    except (ValueError, TypeError):
        pass
    return value


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
            enrollment_count=clean(row.get("enrollment_count")),
            sponsor_class=row.get("sponsor_class"),
            num_sites=clean(row.get("num_sites")),
            fda_regulated_drug=row.get("fda_regulated_drug"),
            minimum_age=row.get("minimum_age"),
            maximum_age=row.get("maximum_age"),
            accepts_healthy=row.get("accepts_healthy"),
            sex=row.get("sex"),
            num_collaborators=clean(row.get("num_collaborators")),
        )

        session.merge(trial)  # merge = insert or update, avoids duplicate key errors
        count += 1

    session.commit()
    session.close()
    print(f"Loaded {count} trials into the database.")

if __name__ == "__main__":
    load_csv_to_db()
