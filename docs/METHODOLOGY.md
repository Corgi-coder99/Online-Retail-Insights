# Methodology

## 1. Definitions

| Term | Definition |
|---|---|
| Sale line | Product line on a non-cancellation invoice with price > 0 and quantity > 0 |
| Gross sales | Sum of quantity x price over sale lines (GBP) |
| Returns | Absolute value of cancellation lines (invoice number starts with `C`) for products |
| Net revenue | Gross sales minus returns, by invoice date |
| Customer | A customer id (anonymised in the source); lines without one are excluded from customer analysis only |
| Order | An invoice with at least one sale line |

## 2. Cleaning (see `outputs/cleaning_log.md` for counts)

1. Drop the duplicated 1-9 Dec 2010 window from the 2009-2010 sheet (identical multiset of rows in both sheets, verified).
2. Classify the rest into `sale`, `cancellation`, `non_product`, `zero_or_negative_price`, `adjustment`. Nothing is deleted from `clean_lines`.
3. Product descriptions: one description per stock code (most frequent), upper-cased, trimmed.
4. Outliers are flagged, not removed, except for the forecast series and product statistics where lines above GBP 50,000 in absolute value (one order and its cancellation) are excluded.

## 3. Customer segments (RFM)

Snapshot 10 Dec 2011. Recency, frequency and monetary value are scored 1-5 by quintile of the rank. Segments: Champions (R,F,M all >= 4), Loyal (F >= 4 and R >= 3), At risk (R <= 2 and F >= 3), New (R = 5, F = 1), Promising (R >= 4, F 2-3), Hibernating (R <= 2, F <= 2), Needs attention (rest). Thresholds are a judgement call and are written in the code.

## 4. Cohorts and LTV

Cohort = month of a customer's first sale invoice. Retention in month k = customers with a sale in month k / cohort size. 12-month spend = sum of revenue per customer over months 0-11 for cohorts Dec 2009 to Nov 2010.

## 5. Bundles

Among the 300 products on the most invoices: pairs bought together in at least 150 invoices; support, confidence and lift computed on all sale invoices. Reported pairs need 300+ joint orders.

## 6. Revenue forecast

- Series: weekly net revenue (Monday start), first and last partial weeks dropped, missing weeks filled with zero, 104 weeks.
- Models compared: last value, same week last year, same week last year with damped growth, regression on log revenue (trend, three Fourier pairs, year-end closure weeks), and the average of same-week-last-year and regression.
- Backtest: rolling origin from week 56, every 4 weeks, 13 weeks ahead (9 origins). Metric: weighted absolute percentage error (sum of absolute errors / sum of actuals).
- Choice rule: a complex model must beat same-week-last-year by 2 percentage points, otherwise the simple method is used.
- 80% range: 10th and 90th percentile of actual / forecast ratios from the backtest. Coverage on those same backtest weeks: 79% (it was calibrated on them, so this is a consistency check, not out-of-sample proof).

## 7. Customer model

- Question: will a known customer buy again in the next 180 days, and how much?
- Features at a cut-off date: recency, orders, tenure, spend to date, average order value, orders and spend in the last 90/180 days, return ratio, distinct products, average gap between orders, UK flag.
- Models: gradient boosting classifier for "buys again", gradient boosting regressor for spend given a purchase; expected spend = probability x spend.
- Validation without leakage: train on cut-offs in 2010 (targets known), test on 2011-06-09 (target window 2011-06-10 to 2011-12-09 fully observed).
- Result on the test cut-off: AUC 0.76 (baseline "recent orders then recency": 0.78), Spearman of spend 0.54 (baseline 0.56), top decile captured 62% of spend (baseline 61%), total over-predicted by 36%.
- Final prediction: model retrained on all cut-offs with known targets, applied to 10 Dec 2011. Only segment-level aggregates are reported.

## 8. Insight engine

Each insight is produced by a rule with a number behind it. It carries: headline, evidence values, a confidence level (high, medium, low) with the reason, a caveat, and a suggested action. Confidence is high when the figure is a direct count over all data, medium when it depends on a modelling choice or has measurable noise, low when it rests on a few accounts or on a prediction that did not clearly beat a simple baseline. Statements never claim causation.

## 9. Known limitations

- Two years of history; one Christmas season per year.
- No external data (marketing, prices, calendar events, weather).
- Returns are not linked to original sales, so product return rates are approximate.
- Missing customer ids (13% of revenue) are not recoverable.
