import json
import pandas as pd

def load_raw_data(filepath="data/raw/raw_trials.json"):
    with open(filepath, "r") as f:
        return json.load(f)

def parse_trials(data):
    rows = []
    for study in data.get("studies", []):
        protocol = study.get("protocolSection", {})

        identification = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})
        design = protocol.get("designModule", {})
        conditions = protocol.get("conditionsModule", {})
        sponsor = protocol.get("sponsorCollaboratorsModule", {})
        eligibility = protocol.get("eligibilityModule", {})
        locations = protocol.get("contactsLocationsModule", {})
        oversight = protocol.get("oversightModule", {})

        # --- original fields ---
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

            # --- new fields ---
            # How many patients enrolled
            "enrollment_count": design.get("enrollmentInfo", {}).get("count"),

            # Industry vs NIH vs academic — strong signal for funding stability
            "sponsor_class": sponsor.get("leadSponsor", {}).get("class"),

            # Number of trial sites — more sites = more complex trial
            "num_sites": len(locations.get("locations", [])),

            # Is the drug FDA regulated?
            "fda_regulated_drug": oversight.get("isFdaRegulatedDrug"),

            # Eligibility details
            "minimum_age": eligibility.get("minimumAge"),
            "maximum_age": eligibility.get("maximumAge"),
            "accepts_healthy": eligibility.get("healthyVolunteers"),
            "sex": eligibility.get("sex"),

            # Number of collaborators (more = more support/funding)
            "num_collaborators": len(sponsor.get("collaborators", [])),
        }
        rows.append(row)

    return pd.DataFrame(rows)

if __name__ == "__main__":
    data = load_raw_data()
    df = parse_trials(data)
    print(df.head())
    print(f"\nTotal trials: {len(df)}")
    print(f"\nNew columns: {list(df.columns)}")
    print(f"\nEnrollment count sample:\n{df['enrollment_count'].describe()}")
    print(f"\nSponsor class breakdown:\n{df['sponsor_class'].value_counts()}")

    df.to_csv("data/processed_trials.csv", index=False)
    print("\nSaved to data/processed_trials.csv")
