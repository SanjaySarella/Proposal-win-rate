import requests
import pandas as pd
import time
import json

BASE_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

def pull_contracts(num_pages=10):
    all_records = []
    
    for page in range(1, num_pages + 1):
        print(f"Pulling page {page} of {num_pages}...")
        
        payload = {
            "subawards": False,
            "limit": 100,
            "page": page,
            "filters": {
                "award_type_codes": ["A", "B", "C", "D"],
                "time_period": [
                    {"start_date": "2022-01-01", "end_date": "2024-12-31"}
                ],
                "award_amounts": [
                    {"lower_bound": 100000, "upper_bound": 50000000}
                ]
            },
            "fields": [
                "Award ID",
                "Recipient Name",
                "Award Amount",
                "Total Outlays",
                "Description",
                "Contract Award Type",
                "Award Type",
                "Awarding Agency",
                "Awarding Sub Agency",
                "Start Date",
                "End Date",
                "recipient_id",
                "def_codes",
                "COVID-19 Obligations",
                "COVID-19 Outlays",
                "Infrastructure Obligations",
                "Infrastructure Outlays",
                "Funding Agency",
                "Place of Performance State Code",
                "Place of Performance Country Code",
                "Recipient UEI",
                "Recipient DUNS",
                "recipient_id"
            ],
            "sort": "Award Amount",
            "order": "desc"
        }
        
        try:
            response = requests.post(BASE_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            if not results:
                print(f"No more results at page {page}. Stopping.")
                break
                
            all_records.extend(results)
            print(f"  Got {len(results)} records. Total so far: {len(all_records)}")
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break
    
    df = pd.DataFrame(all_records)
    df.to_csv("data/raw_contracts.csv", index=False)
    print(f"\nDone. Total records pulled: {len(df)}")
    print(f"Saved to data/raw_contracts.csv")
    print(f"\nColumns: {list(df.columns)}")
    return df

if __name__ == "__main__":
    df = pull_contracts(num_pages=10)
    print(df.head())