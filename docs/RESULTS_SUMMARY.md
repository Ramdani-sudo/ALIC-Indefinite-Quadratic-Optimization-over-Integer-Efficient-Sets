# Main experimental findings

- 1440 paired instances, arranged in 48 configurations with 30 replications each.
- ALIC: 1438 optimal solves (99.86%); Prerna--Sharma reconstruction: 1421 (98.68%).
- 1421 instances were solved to optimality by both methods, and all 1421 optimal objective values agree.
- Decision vectors agree on 1419/1421 common optimal solves; two pairs have different optimal decisions with the same objective value.
- ALIC solved 17 instances that the reference did not certify; there is no instance solved only by the reference. Both methods timed out on 2 instances.
- On common optimal solves, the reference method is faster on the typical instance: median 0.168 s versus 0.433 s for ALIC. The paired Wilcoxon test gives p = 4.39e-78 and rank-biserial = -0.573.
- ALIC has the better heavy-tail behavior: mean runtime 1.314 s versus 2.969 s, maximum common-solve runtime 99.493 s versus 283.294 s, and PAR-2 mean 2.587 s versus 10.846 s.
- ALIC uses fewer explicit solver calls on every common optimal instance: median 2 versus 25; mean 2.12 versus 44.97. The median reference/ALIC call ratio is 12.5.
- Across 48 configuration-level runtime comparisons after Holm correction, 31 significantly favor Prerna--Sharma, 4 significantly favor ALIC, and 13 are not significant.
- The 4 significant ALIC runtime wins are all at n=50, m=50, with median speedups from 6.52 to 7.03.
- Bayesian signed-rank with a ±10% ROPE gives posterior probabilities 0.786 (reference better), 0.043 (practically equivalent), and 0.171 (ALIC better) on common optimal runtimes.
