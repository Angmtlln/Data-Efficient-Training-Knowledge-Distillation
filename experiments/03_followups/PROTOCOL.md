# Follow-up series, 18 November 2026

Frozen before new GPU outcomes. 30 runs: 10 methods × seeds 202/303/404. Same ModernBERT config, source splits and 6534 row budget as old series. No teacher. New results exploratory: old test has already informed design. No test labels used in selection scripts. No changes to old outputs.

## Forgetting (15 runs)

Fixed OOF score from original scoring seed17; 932 nonzero rows. New selection/training seeds, not new scoring proxies. Common filler of 5602 rows, plus a block of 932 rows. Each target row has a separate class×length-bin matched zero-score replacement. Arms: full-pool random, matched-zero control, 327 targets (~5%), 653 targets (~10%), all 932 (~14.3%). Replacements are nested. Matched control explicitly excludes all nonzero rows; do not call it ordinary random. Full-pool random included separately.

Primary contrast: forget_all versus forget_matched_zero; secondary doses and full-pool random. Counts, length/class diagnostics in reports/selection_audit.csv. This isolates a targeted block, not full-population rankings.

## Gradient/skill repair (15 runs)

Keep original MiniLM proxy and references. CPU 2×2 probe (bias on/off × class-centering on/off) showed that bias removal and centering alone fail to remove class-specific difficulty extremes. Therefore no full-top-k GPU sweep of these variants.

All five arms share exact quotas over label × word-length-bin × proxy correctness × within-label/correctness loss quintile. Same ~75% background per seed; remaining ~25% slots selected within strata. Arms:
- skill_matched_random_v2: matched random control;
- grad_raw_mix25: original gradient, now quotas + mixture;
- grad_nobias_mix25: remove bias term, otherwise same;
- grad_randomref_nobias_mix25: no bias, random matched reference;
- neg_embedding_mix25: soft negation targeting, same controls.

Thus every arm has identical class-specific proxy error fractions (20.66% for label0, 14.32% for label1), instead of the old 0/100% asymmetry. This is a quota-based correction, NOT proof the scores now measure useful information. Quotas preserve counts in coarse bins, not exact continuous loss or length.

Primary contrasts: no-bias vs raw (effect of bias after controlling mixture/quotas); no-bias target vs random-reference (reference specificity); each targeted arm vs common random (incremental value). Overall accuracy/macro F1 plus negation/contrast lexical slices. No independent behavioral validation claimed.

Checkpoints every100 steps/epoch, two most recent persistent states plus best; local limit2. Approximately 13 GB free on VM before launch. Automatic off-VM backup copies metrics/predictions/configs/logs, not heavy weights. Local permanent archive will be updated on checks; temporary background staging is not the only copy of old results.
