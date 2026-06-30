import json
import pandas as pd

def load_raw_data(filepath="data/raw/raw_trials.json"):
    with open(filepath, "r") as f:
        return json.load(f)

def parse_trials(data):
    """Flatten the nested JSON into a list of clean dicts."""
    rows = []
    for study in data.get("studies", []):
        protocol = study.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})
        design = protocol.get("designModule", {})
        conditions = protocol.get("conditionsModule", {})
        sponsor = protocol.get("sponsorCollaboratorsModule", {})

        row = {
            "nct_id": identification.get("nctId"),
            "title": identification.get("officialTitle") or identification.get("briefTitle"),
            "status": status.get("overallStatus"),
            "phase": ", ".join(design.get("phases", [])) if design.get("phases") else None,
            "study_type": design.get("studyType"),
            "conditions": ", ".join(conditions.get("conditions", [])) if conditions.get("conditions") else None,
            "lead_sponsor": sponsor.get("leadSponsor", {}).get("name"),
            "start_date": status.get("startDateStruct", {}).get("date"),
            "completion_date": status.get("completionDateStruct", {}).get("date"),
        }
        rows.append(row)

    return pd.DataFrame(rows)

if __name__ == "__main__":
    data = load_raw_data()
    df = parse_trials(data)
    print(df.head())
    print(f"\nTotal trials: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")

    df.to_csv("data/processed_trials.csv", index=False)
    print("\nSaved to data/processed_trials.csv")
