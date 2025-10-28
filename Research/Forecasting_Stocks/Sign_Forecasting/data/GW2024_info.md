| # | Name | Description | Frequency | SampleBeg | SampleEnd | Notes |
|---:|---|---|---|---:|---:|---|
| 1 | date |  |  |  |  |  |
| 2 | price | price (index value) | Monthly | 187101 | 202412 |  |
| 3 | d12 | 12-month dividends | Monthly | 187101 | 202412 |  |
| 4 | e12 | 12-month earnings | Monthly | 187101 | 202412 |  |
| 5 | ret | return w/ dividends (CRSP calc) | Monthly | 192601 | 202412 | CRSP's calculation of S&P500 return (incl. dividends) |
| 6 | retx | return w/o dividends (CRSP calc) | Monthly | 192601 | 202412 | CRSP's calculation of S&P500 return (excl. dividends) |
| 7 | AAA | AAA bond yield | Monthly | 191901 | 202412 |  |
| 8 | BAA | BAA bond yield | Monthly | 187101 | 202412 |  |
| 9 | lty | long govt yield | Monthly | 187101 | 202412 |  |
| 10 | ltr | long govt return | Monthly | 192601 | 202412 |  |
| 11 | corpr | corporate bond return | Monthly | 192601 | 202412 |  |
| 12 | tbl | t-bill | Monthly | 192001 | 202412 |  |
| 13 | Rfree | riskfree return | Monthly | 187102 | 202412 |  |
| 14 | d/p | dividend price ratio | Monthly | 187101 | 202412 | = d12/price |
| 15 | d/y | dividend yield | Monthly | 187101 | 202412 | = d12/lag price |
| 16 | e/p | earnings price ratio | Monthly | 187101 | 202412 | = e12/price |
| 17 | d/e | dividend payout | Monthly | 187101 | 202412 | = d12/e12 |
| 18 | b/m | b/m | Monthly | 192103 | 202412 |  |
| 19 | tms | term spread | Monthly | 187101 | 202412 | = lty - tbl |
| 20 | dfy | default yield spread | Monthly | 187101 | 202412 | = BAA - AAA |
| 21 | dfr | default return spread | Monthly | 187101 | 202412 | = corpr - ltr |
| 22 | infl | inflation | Monthly | 191302 | 202412 |  |
| 23 | eqis | pct equity issuance | Annual | 1927 | 2024 |  |
| 24 | ntis | net equity issuance | Monthly | 192612 | 202412 |  |
| 25 | svar | $\sigma^2$ | Monthly | 188502 | 202412 |  |
| 26 | cay | cnsm, wlth, incm | Quarterly | 19521 | 20244 | Needs recomputation every period. Only full-sample version here. |
| 27 | i/k | invstmt/capital | Quarterly | 19471 | 20244 |  |
| 28 | csp | cross-sectional premium | Monthly | 193705 | 200212 |  |
| 29 | pce | consumption/trend | Quarterly | 19534 | 20244 | Needs recomputation every period. Only full-sample version here. |
| 30 | vp | variance premium | Monthly | 199001 | 202112 |  |
| 31 | impvar | implied $\sigma^2$ | Monthly | 199601 | 202308 |  |
| 32 | vrp | $\sigma^2$ risk premium | Monthly | 199001 | 202312 |  |
| 33 | govik | public sector investmt | Quarterly | 19471 | 20244 |  |
| 34 | lzrt | 9 illiq measures | Monthly | 192601 | 202412 |  |
| 35 | skew | skewness | Semiannual | 19512 | 20192 |  |
| 36 | crdstd | credit standards | Quarterly | 19902 | 20244 |  |
| 37 | ogap | prdctn-output gap | Monthly | 192601 | 202412 | Needs recomputation every period. Only full-sample version here. |
| 38 | wtexas | oil price changes | Monthly | 192601 | 202412 |  |
| 39 | accrul | accruals | Annual | 1965 | 2024 |  |
| 40 | cfacc | accruals (CFO) | Annual | 1965 | 2024 |  |
| 41 | sntm | distilled sentiment | Monthly | 196507 | 202312 | PLS variable. Needs to be computed very period for the forecasting target, and for the frequency of forecast for OOS. Only the full-sample version for predicting monthly log excess returns here. |
| 42 | ndrbl | new order-ship durables | Monthly | 195802 | 202412 |  |
| 43 | skvw | avg stock skewness | Monthly | 192607 | 202412 |  |
| 44 | tail | x-sect tail risk | Monthly | 192607 | 202412 |  |
| 45 | fbm | b/m x-sect factor | Monthly | 192606 | 202411 | PLS variable. Needs to be computed very period for the forecasting target, and for the frequency of forecast for OOS. Only the full-sample version for predicting monthly log excess returns here. |
| 46 | dtoy | to Dow 52-week high | Monthly | 192601 | 202412 |  |
| 47 | dtoat | to Dow all-time high | Monthly | 192601 | 202412 |  |
| 48 | ygap | stock-bond yield gap | Monthly | 195304 | 202412 |  |
| 49 | rdsp | stock return dispersion | Monthly | 192609 | 202412 |  |
| 50 | rsvix | scaled risk-neutral vix | Monthly | 199601 | 202308 | Comes in three varieties fore predicting monthly, quarterly, and annual forecasts. Only the version for predicting at monthly frequency here. |
| 51 | gpce | yearend econ growth | Annual | 1947 | 2024 |  |
| 52 | gip | yearend econ growth | Annual | 1926 | 2024 |  |
| 53 | tchi | 14 technical indicators | Monthly | 195101 | 202412 | Needs recomputation every period. Only full-sample version here. |
| 54 | house | housing/consumption | Annual | 1929 | 2024 |  |
| 55 | avgcor | acvg corr stock returns | Monthly | 192603 | 202412 |  |
| 56 | shtint | short interest | Monthly | 197301 | 202412 | Needs recomputation every period. Only full-sample version here. |
| 57 | disag | analyst disagreement | Monthly | 198112 | 202412 |  |