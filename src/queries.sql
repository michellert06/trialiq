-- Failure rate by trial phase
-- Shows TERMINATED + WITHDRAWN trials as % of total, grouped by phase
SELECT phase, COUNT(*) AS total,
       ROUND(100.0 * SUM(CASE WHEN status IN ('TERMINATED','WITHDRAWN') THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_failed
FROM trials
GROUP BY phase
ORDER BY total DESC;

-- Top 10 most active trial sponsors in the dataset
SELECT lead_sponsor, COUNT(*) AS total_trials
FROM trials
GROUP BY lead_sponsor
ORDER BY total_trials DESC
LIMIT 10;
