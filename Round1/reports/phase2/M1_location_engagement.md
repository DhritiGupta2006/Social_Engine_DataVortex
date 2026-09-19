# M1 — Which Locations Generate the Most Engagement

**Query:** `sql/M1_location_engagement.sql`  
**Database:** `data/datavortex.db`  
**Rows returned:** 33 (one per distinct location)

## Results

| location | post_count | total_engagement | avg_engagement_per_post |
| --- | --- | --- | --- |
| Los Angeles, USA | 459 | 1,691,398.00 | 3,685.00 |
| Munich, Germany | 452 | 1,654,881.00 | 3,661.20 |
| Shanghai, China | 451 | 1,623,667.00 | 3,600.10 |
| Barcelona, Spain | 439 | 1,620,828.00 | 3,692.10 |
| Dubai, UAE | 421 | 1,532,010.00 | 3,639.00 |
| Melbourne, Australia | 421 | 1,528,838.00 | 3,631.40 |
| Houston, USA | 422 | 1,501,845.00 | 3,558.90 |
| Rio de Janeiro, Brazil | 414 | 1,494,576.00 | 3,610.10 |
| Osaka, Japan | 398 | 1,472,669.00 | 3,700.20 |
| Mumbai, India | 413 | 1,466,527.00 | 3,550.90 |
| Milan, Italy | 403 | 1,437,389.00 | 3,566.70 |
| London, UK | 386 | 1,428,971.00 | 3,702.00 |
| Chicago, USA | 392 | 1,401,187.00 | 3,574.50 |
| New York, USA | 372 | 1,396,613.00 | 3,754.30 |
| Johannesburg, South Africa | 383 | 1,389,593.00 | 3,628.20 |
| Paris, France | 379 | 1,376,281.00 | 3,631.30 |
| Beijing, China | 373 | 1,371,525.00 | 3,677.00 |
| São Paulo, Brazil | 376 | 1,351,220.00 | 3,593.70 |
| Berlin, Germany | 355 | 1,310,485.00 | 3,691.50 |
| Singapore | 358 | 1,298,653.00 | 3,627.50 |
| Tokyo, Japan | 354 | 1,281,361.00 | 3,619.70 |
| Toronto, Canada | 350 | 1,259,966.00 | 3,599.90 |
| Manchester, UK | 331 | 1,197,263.00 | 3,617.10 |
| Rome, Italy | 338 | 1,187,932.00 | 3,514.60 |
| Vancouver, Canada | 331 | 1,181,335.00 | 3,569.00 |
| Mexico City, Mexico | 320 | 1,173,552.00 | 3,667.40 |
| Cairo, Egypt | 322 | 1,166,086.00 | 3,621.40 |
| Delhi, India | 297 | 1,094,894.00 | 3,686.50 |
| Madrid, Spain | 290 | 1,050,420.00 | 3,622.10 |
| Lyon, France | 293 | 1,007,149.00 | 3,437.40 |
| Seoul, South Korea | 276 | 1,001,751.00 | 3,629.50 |
| Lagos, Nigeria | 223 | 802,697.00 | 3,599.50 |
| Sydney, Australia | 208 | 766,727.00 | 3,686.20 |

## Interpretation

- 33 distinct locations found across 12,000 posts.
- `total_engagement` uses `COALESCE(likes, 0)` so missing likes are treated as 0 (conservative lower bound — not imputed).
- Top 3 locations by total engagement:
  1. **Los Angeles, USA** — 1,691,398 total engagement across 459 posts (avg 3,685.0/post)
  1. **Munich, Germany** — 1,654,881 total engagement across 452 posts (avg 3,661.2/post)
  1. **Shanghai, China** — 1,623,667 total engagement across 451 posts (avg 3,600.1/post)
- Locations with fewer posts naturally accumulate less total engagement, so `avg_engagement_per_post` is the fairer cross-location comparator.
