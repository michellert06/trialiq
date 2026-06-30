import requests
import json
import os
import time

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

CONDITIONS = [
    "breast cancer",
    "lung cancer",
    "diabetes",
    "alzheimer",
    "heart failure",
    "depression",
    "rheumatoid arthritis",
    "asthma",
    "covid-19",
    "obesity",
]

def fetch_trials_for_condition(condition, max_pages=5, page_size=100):
    """Fetch multiple pages of trials for one condition, following pagination."""
    all_studies = []
    next_page_token = None

    for _ in range(max_pages):
        params = {
            "query.cond": condition,
            "pageSize": page_size,
            "format": "json"
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        studies = data.get("studies", [])
        all_studies.extend(studies)

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.5)  # be polite to the API

    print(f"  {condition}: fetched {len(all_studies)} trials")
    return all_studies

def save_raw_data(data, filename="raw_trials.json"):
    os.makedirs("data/raw", exist_ok=True)
    filepath = os.path.join("data/raw", filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved data to {filepath}")

if __name__ == "__main__":
    all_studies = []
    for condition in CONDITIONS:
        studies = fetch_trials_for_condition(condition)
        all_studies.extend(studies)

    print(f"\nTotal trials fetched: {len(all_studies)}")
    save_raw_data({"studies": all_studies})
