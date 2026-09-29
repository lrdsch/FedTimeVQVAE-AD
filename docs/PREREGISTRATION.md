# Pre-registration — confirmation sample (50 UCR series)

**Written 2026-08-09, ~03:00 UTC, BEFORE producing any number on the drawn series.**
This is a faithful English rendering of the dated document that was frozen at that time; the
original (Italian) is the one that was written first, and nothing below has been added after the
fact. The final section distinguishes this historical plan from the submitted reporting scope.

## The draw (closed, non-renegotiable)

- Population: the **180 series** of the `ucr_split_w2p` build (W = 2 × period).
- Excluded: the **10 development series** (cohort `ucr2p_10`) — 001, 011, 014, 043, 082, 083,
  086, 170, 222, 229.
- Sample: **50 series** drawn with `random.Random(20260809).sample(...)` from the 170 candidates,
  then permuted with the same RNG to fix the **order of execution**.
- The ordered list is at the bottom; the queue executes in that order, so whatever prefix has
  completed by the freeze is a valid random sample by construction.
- **Minimum n: 30 complete series** (all three arms). **Freeze: the morning of 2026-08-18.**
  A series with an incomplete arm at the freeze is discarded entirely, from every arm.

## Arms (identical to the development tags; cohort `c50`, fingerprint 5080389790ba83f3, seed 0)

- `c50_central` — arm `centralized`, extra `--window-normalization zscore`
- `c50_local` — arm `local`, extra `--window-normalization zscore`
- `c50_a2` — arm `federated_enc_fedavg`, extra `--window-normalization zscore --fed-enc-cb
  suffstat --fed-enc-prior partial --fed-enc-bn shared`. The mobile-dictionary variant is
  **excluded**: it failed its development gate on `ucr_043` (0.298 against 0.744).
- Protocol: `converged`, S1 = S2 = 300 rounds (ceiling), patience 6, 10 local epochs, batch 64,
  fp16, metric tolerance 64. Truncated cells are not reportable.
- One post-hoc column at zero cost: **local-fused**, the mean of the five `local` clients'
  z-normalized score profiles. The fusion rule is declared *here*: mean of z-scores; the trimmed
  mean, the median and the rank mean are reported only as sensitivity.

## Analysis (declared now)

- **Primary**: `paper_top1_acc@64` per series; paired comparisons between arms by **sign test**
  across series; **ties count as ties**.
- Secondary: AUPRC (paired Δ, mean + median + win count), AUROC (idem), VUS-PR **per series
  only**, never averaged across series.
- Unit of analysis: the **series** (cluster). The five clients are not replicates.
- Contrasts: A2 vs local · centralized vs local · A2 vs centralized · local-fused vs local and
  vs A2.

## Hardware

All cells on one host (2 × Quadro RTX 8000, sm_75, fp16), so the arms are paired by host and by
architecture.

## What actually happened

- 48 of the 50 series were analyzed. `ucr_190` and `ucr_240` were discarded from **all** arms by
  the rule above: the federated arm aborted on non-finite entries in the aggregated codebook.
  Recovering both as wins would move the primary contrast against local training from p = 0.33 to
  p = 0.17.
- **Departure 1**: `c50_tau64` (the step-counted stage-2 rhythm) was added to the sample after
  this document was written. It is not a pre-registered arm and the paper reports it as a null.
- **Final reporting scope**: the submitted paper uses the preregistered paired sign
  test, with ties excluded, and reports localization accuracy and AUPRC. It does not
  report a Wilcoxon comparison, AUROC, range-aware scores or confirmation-set score fusion.
- **Stopping disclosure**: the paper retains local and centralized baseline runs that
  reach their step ceilings. Its federated truncation flag identifies stages whose
  best round is the last one. The original blanket statement above should not be
  read as an assertion that every reported baseline reached patience.


## The ordered list of 50 series

```
 1. ucr_163   11. ucr_080   21. ucr_057   31. ucr_245   41. ucr_088
 2. ucr_055   12. ucr_207   22. ucr_117   32. ucr_190   42. ucr_060
 3. ucr_045   13. ucr_092   23. ucr_095   33. ucr_240   43. ucr_192
 4. ucr_091   14. ucr_246   24. ucr_113   34. ucr_093   44. ucr_211
 5. ucr_172   15. ucr_164   25. ucr_184   35. ucr_059   45. ucr_064
 6. ucr_244   16. ucr_194   26. ucr_203   36. ucr_150   46. ucr_147
 7. ucr_044   17. ucr_018   27. ucr_250   37. ucr_185   47. ucr_002
 8. ucr_187   18. ucr_166   28. ucr_202   38. ucr_210   48. ucr_106
 9. ucr_100   19. ucr_107   29. ucr_224   39. ucr_174   49. ucr_073
10. ucr_151   20. ucr_223   30. ucr_012   40. ucr_010   50. ucr_165
```

Series 32 and 33 in that order are the two that were discarded.
