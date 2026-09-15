# H2 — Rank Users Within Their Location

**Query:** `sql/H2_rank_users_by_location.sql`  
**Database:** `data/datavortex.db`  
**Rows returned:** 99 across 33 locations

## Notes

- `RANK()` is used (not `ROW_NUMBER()`), so tied users at rank 3 both appear.
- Users with zero posts are excluded by the INNER JOIN in the CTE — none are expected given 100% `user_id` resolution in Phase 1 validation.
- Locations with more than 3 rows due to rank-3 ties: **0**.

## Results

| location | location_rank | user_id | follower_count | post_count | total_engagement |
| --- | --- | --- | --- | --- | --- |
| Barcelona, Spain | 1 | user_nfo3ih5u | 40,429 | 19 | 71,529.00 |
| Barcelona, Spain | 2 | user_5s9ifp0y | 28,412 | 11 | 56,538.00 |
| Barcelona, Spain | 3 | user_24wzfb8b | 38,633 | 14 | 53,039.00 |
| Beijing, China | 1 | user_cdzp4vm2 | 11,187 | 11 | 48,019.00 |
| Beijing, China | 2 | user_ttlouvlq | 19,373 | 12 | 47,244.00 |
| Beijing, China | 3 | user_gre0h06x | 23,303 | 11 | 45,311.00 |
| Berlin, Germany | 1 | user_6e99lerx | 25,366 | 14 | 54,263.00 |
| Berlin, Germany | 2 | user_m5z8a3sg | 34,800 | 13 | 53,951.00 |
| Berlin, Germany | 3 | user_rtp2dykx | 14,113 | 11 | 52,748.00 |
| Cairo, Egypt | 1 | user_cuzig14g | 8,070 | 15 | 52,541.00 |
| Cairo, Egypt | 2 | user_g37hrbp3 | 6,699 | 11 | 51,740.00 |
| Cairo, Egypt | 3 | user_ogtvuuki | 4,459 | 11 | 44,566.00 |
| Chicago, USA | 1 | user_ujllj1n7 | 49,855 | 16 | 64,082.00 |
| Chicago, USA | 2 | user_pr7bcf45 | 30,128 | 13 | 57,393.00 |
| Chicago, USA | 3 | user_u98jwp3f | 49,936 | 11 | 48,996.00 |
| Delhi, India | 1 | user_8776jund | 21,147 | 12 | 52,005.00 |
| Delhi, India | 2 | user_zq2nib62 | 5,153 | 12 | 51,134.00 |
| Delhi, India | 3 | user_sktr0cyy | 42,180 | 13 | 46,306.00 |
| Dubai, UAE | 1 | user_xetn4exw | 14,902 | 16 | 59,677.00 |
| Dubai, UAE | 2 | user_wyeb78qv | 24,634 | 16 | 55,214.00 |
| Dubai, UAE | 3 | user_rsrg2sgy | 32,791 | 12 | 54,505.00 |
| Houston, USA | 1 | user_68enpikx | 9,364 | 15 | 57,816.00 |
| Houston, USA | 2 | user_r7eg1rac | 2,052 | 12 | 54,552.00 |
| Houston, USA | 3 | user_44h4xl31 | 48,691 | 10 | 45,515.00 |
| Johannesburg, South Africa | 1 | user_9mtets0p | 8,262 | 14 | 62,414.00 |
| Johannesburg, South Africa | 2 | user_6wra58f7 | 4,953 | 14 | 54,891.00 |
| Johannesburg, South Africa | 3 | user_cefi5hpb | 15,895 | 11 | 50,813.00 |
| Lagos, Nigeria | 1 | user_slqlepta | 40,296 | 13 | 57,975.00 |
| Lagos, Nigeria | 2 | user_jnknxe13 | 35,496 | 13 | 44,618.00 |
| Lagos, Nigeria | 3 | user_tvrrizk4 | 24,598 | 13 | 43,468.00 |
| London, UK | 1 | user_634lriof | 12,327 | 12 | 60,436.00 |
| London, UK | 2 | user_94hehp26 | 33,258 | 13 | 59,706.00 |
| London, UK | 3 | user_tyvzn6ay | 46,141 | 15 | 53,346.00 |
| Los Angeles, USA | 1 | user_8wk0j5ae | 36,943 | 18 | 59,623.00 |
| Los Angeles, USA | 2 | user_3zzvv8vp | 42,908 | 13 | 59,221.00 |
| Los Angeles, USA | 3 | user_kxujr0gz | 34,963 | 11 | 52,299.00 |
| Lyon, France | 1 | user_fgjkkrie | 2,211 | 14 | 62,690.00 |
| Lyon, France | 2 | user_2lslaaod | 7,105 | 10 | 41,992.00 |
| Lyon, France | 3 | user_d3ymednx | 17,904 | 10 | 37,730.00 |
| Madrid, Spain | 1 | user_edi9lmxp | 22,782 | 13 | 48,850.00 |
| Madrid, Spain | 2 | user_v4objqex | 16,772 | 12 | 48,218.00 |
| Madrid, Spain | 3 | user_w4p8zi5g | 7,103 | 13 | 46,956.00 |
| Manchester, UK | 1 | user_oknxkzax | 34,257 | 15 | 61,070.00 |
| Manchester, UK | 2 | user_q3vs0uj7 | 32,714 | 18 | 58,438.00 |
| Manchester, UK | 3 | user_qmta1cuk | 44,187 | 12 | 46,834.00 |
| Melbourne, Australia | 1 | user_qdztdeac | 14,266 | 14 | 60,231.00 |
| Melbourne, Australia | 2 | user_d6tjszz8 | 37,186 | 14 | 58,020.00 |
| Melbourne, Australia | 3 | user_mtuzxdlb | 49,341 | 14 | 53,746.00 |
| Mexico City, Mexico | 1 | user_s87enj6b | 43,476 | 14 | 57,872.00 |
| Mexico City, Mexico | 2 | user_716l1rz5 | 41,236 | 13 | 55,724.00 |
| Mexico City, Mexico | 3 | user_cyqqg8x5 | 44,679 | 13 | 49,795.00 |
| Milan, Italy | 1 | user_xa1s1sd5 | 37,758 | 12 | 51,002.00 |
| Milan, Italy | 2 | user_rr1uzkql | 1,069 | 14 | 49,565.00 |
| Milan, Italy | 3 | user_t59clgrj | 43,441 | 12 | 45,341.00 |
| Mumbai, India | 1 | user_2ytzut7i | 17,064 | 14 | 64,710.00 |
| Mumbai, India | 2 | user_gq0rypar | 17,672 | 15 | 63,267.00 |
| Mumbai, India | 3 | user_y9bjiho1 | 24,858 | 13 | 61,956.00 |
| Munich, Germany | 1 | user_al9p1gnu | 49,268 | 13 | 54,267.00 |
| Munich, Germany | 2 | user_g9g7onlw | 7,981 | 13 | 52,499.00 |
| Munich, Germany | 3 | user_58rc29a2 | 29,379 | 12 | 46,004.00 |
| New York, USA | 1 | user_wr2785sn | 12,713 | 12 | 54,539.00 |
| New York, USA | 2 | user_tlwy34pg | 24,453 | 12 | 49,574.00 |
| New York, USA | 3 | user_kk1qdsq3 | 32,189 | 13 | 49,231.00 |
| Osaka, Japan | 1 | user_yr80oaxg | 30,039 | 12 | 50,865.00 |
| Osaka, Japan | 2 | user_5r58l1kr | 23,520 | 11 | 48,738.00 |
| Osaka, Japan | 3 | user_c66o14p3 | 24,996 | 10 | 47,727.00 |
| Paris, France | 1 | user_kf84zwv2 | 49,575 | 13 | 53,942.00 |
| Paris, France | 2 | user_45yefmlp | 37,589 | 12 | 51,355.00 |
| Paris, France | 3 | user_zozhwbis | 4,580 | 11 | 43,697.00 |
| Rio de Janeiro, Brazil | 1 | user_hdas0iau | 1,620 | 16 | 59,073.00 |
| Rio de Janeiro, Brazil | 2 | user_aroc3ncd | 10,428 | 12 | 54,206.00 |
| Rio de Janeiro, Brazil | 3 | user_31qdm3s9 | 33,635 | 13 | 53,535.00 |
| Rome, Italy | 1 | user_uerv85na | 1,824 | 16 | 64,753.00 |
| Rome, Italy | 2 | user_0aacyvtz | 1,452 | 11 | 45,616.00 |
| Rome, Italy | 3 | user_vknr0v08 | 15,568 | 13 | 44,430.00 |
| Seoul, South Korea | 1 | user_ceim4vux | 40,811 | 12 | 45,157.00 |
| Seoul, South Korea | 2 | user_tmjtxubu | 48,104 | 10 | 44,602.00 |
| Seoul, South Korea | 3 | user_ivpoy0x1 | 28,950 | 11 | 43,343.00 |
| Shanghai, China | 1 | user_zyptgxi8 | 47,455 | 12 | 53,471.00 |
| Shanghai, China | 2 | user_cntp6452 | 11,329 | 15 | 50,052.00 |
| Shanghai, China | 3 | user_qq52pl85 | 23,934 | 12 | 47,164.00 |
| Singapore | 1 | user_14ibkds2 | 7,361 | 13 | 57,044.00 |
| Singapore | 2 | user_ggnjpujh | 35,532 | 14 | 48,536.00 |
| Singapore | 3 | user_v469etxv | 23,469 | 12 | 46,567.00 |
| Sydney, Australia | 1 | user_bnswpe65 | 20,002 | 15 | 56,938.00 |
| Sydney, Australia | 2 | user_h4lueh1i | 17,902 | 9 | 43,948.00 |
| Sydney, Australia | 3 | user_kiebkz7i | 34,969 | 11 | 41,329.00 |
| São Paulo, Brazil | 1 | user_0snf4coc | 42,927 | 14 | 54,457.00 |
| São Paulo, Brazil | 2 | user_8bcte4x2 | 46,936 | 12 | 51,993.00 |
| São Paulo, Brazil | 3 | user_e9paascb | 19,695 | 14 | 48,463.00 |
| Tokyo, Japan | 1 | user_d4eat3v3 | 49,914 | 15 | 64,208.00 |
| Tokyo, Japan | 2 | user_csluibwk | 19,091 | 12 | 49,748.00 |
| Tokyo, Japan | 3 | user_kbdvf8d6 | 4,645 | 10 | 48,490.00 |
| Toronto, Canada | 1 | user_0irp4abu | 44,206 | 17 | 66,967.00 |
| Toronto, Canada | 2 | user_cpxksz4r | 49,247 | 13 | 51,429.00 |
| Toronto, Canada | 3 | user_ukkkyp0o | 28,045 | 11 | 45,589.00 |
| Vancouver, Canada | 1 | user_zqv2vrf5 | 13,531 | 22 | 83,224.00 |
| Vancouver, Canada | 2 | user_i1e1kek5 | 46,765 | 16 | 62,400.00 |
| Vancouver, Canada | 3 | user_lk8wtuqr | 32,276 | 14 | 51,577.00 |

## Interpretation

- Rankings are based on `total_engagement = SUM(COALESCE(likes, 0) + shares + comments)` per user within each location.
- `follower_count` is shown for reference; as established in Phase 1 Insight #1, it carries no engagement signal (r = −0.011).
- The top-ranked user per location can be used to seed a 'local influencer' feature for a rebuilt recommendation engine.
