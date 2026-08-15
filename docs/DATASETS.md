# Dataset registry

Prompt content is not bundled. The registry stores source IDs, immutable revisions, licenses,
roles, and—in the case of official raw files—SHA-256 checksums.

## Direction cohorts

### `cyberseceval-mitre`

Meta PurpleLlama CyberSecEval MITRE benchmark, pinned to commit
`114d02388213ae791650b10fa61c8785661c2602`. The loader uses `base_prompt`, not the much longer
augmented mutation, to avoid teaching the direction an augmentation style. License: MIT.

Source: <https://github.com/meta-llama/PurpleLlama/tree/114d02388213ae791650b10fa61c8785661c2602/CybersecurityBenchmarks/datasets>

### `cyberseceval-frr`

Meta’s benign, security-looking false-refusal-rate prompts at the same pinned commit. License:
MIT. These prompts are not one-to-one semantic pairs with MITRE, which is why the loader length
matches the cohorts and records exact prompt hashes.

Paper: <https://arxiv.org/abs/2404.13161>

## Ranking, evaluation, and prompt-only sources

- `jbb-cyber-harmful` and `jbb-cyber-benign`: the `Malware/Hacking` category from pinned
  `JailbreakBench/JBB-Behaviors`; MIT. These two must be used together for direction ranking so a
  lower false-refusal score cannot hide lost harmful refusal behavior.
- `wmdp-cyber`: scored multiple-choice cyber knowledge preservation; MIT. Final evaluation only,
  never direction fitting or ranking.
- `ifeval`: pinned Apache-2.0 prompt-only source. The registry intentionally marks it unscored
  because this package does not retain/run IFEval’s constraint checker.
- `wildguard-test`: pinned prompt-only source under ODC-By plus gated AI2 responsible-use terms.
  The registry marks commercial use false and does not claim to score labels it does not retain.

The three registry flags have distinct meanings: `use_for_direction`, `use_for_ranking`, and
`use_for_eval`. Final holdouts cannot be used by `rank` through the CLI.

Recommended additional offline holdouts are HarmBench’s `cybercrime_intrusion` test set and
XSTest. They are documented but intentionally not auto-downloaded by this release.

## Data hygiene

- Split by stable source/group ID before tuning.
- Near-deduplicate across direction, validation, and holdout sources.
- Match domain and length; otherwise the dominant vector may represent formatting.
- Length-matched pairs are shuffled together before truncation so short bins are not favored.
- Never print sensitive prompt text in CI, telemetry, cache names, or reports.
- Preserve `source_id`, revision, row ID, license, and cohort hash in every artifact.
- Keep holdout sources outside direction selection and hyperparameter tuning.
