CREATE TEMP VIEW user_stats AS
SELECT user_id, COUNT(*) tx_count, AVG(amount) avg_amount, SUM(amount) total_amount FROM transactions GROUP BY user_id;
CREATE TEMP VIEW daily_user AS
SELECT user_id, date, COUNT(*) daily_tx, SUM(amount) daily_amount FROM transactions GROUP BY user_id,date;
SELECT t.tx_id,t.user_id,t.date,t.region,t.merchant,t.amount,us.tx_count,us.avg_amount,us.total_amount,COALESCE(du.daily_tx,0) daily_tx,COALESCE(du.daily_amount,0.0) daily_amount
FROM transactions t
LEFT JOIN user_stats us ON t.user_id=us.user_id
LEFT JOIN daily_user du ON t.user_id=du.user_id AND t.date=du.date;