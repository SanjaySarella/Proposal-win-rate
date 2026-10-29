-- Data-quality and snapshot-shape checks for the federal award case study.
SELECT COUNT(*) AS source_rows,
       COUNT(*) - COUNT(DISTINCT generated_internal_id) AS duplicate_internal_ids,
       SUM(CASE WHEN "Award Amount" IS NULL OR "Award Amount" < 0 THEN 1 ELSE 0 END) AS invalid_amount_rows,
       MIN("Award Amount") AS minimum_amount,
       MEDIAN("Award Amount") AS median_amount,
       MAX("Award Amount") AS maximum_amount
FROM read_csv_auto('data/processed/contracts_features.csv');

-- Agency reconciliation within the bounded snapshot.
SELECT "Awarding Agency", COUNT(*) AS award_records,
       SUM("Award Amount") AS recorded_value
FROM read_csv_auto('data/processed/contracts_features.csv')
GROUP BY "Awarding Agency"
ORDER BY recorded_value DESC;
