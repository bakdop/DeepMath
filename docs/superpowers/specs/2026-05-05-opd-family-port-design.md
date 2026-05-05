# OPD-family port: SRPO, KDRL-MASK, HDPO-cliff, RLSD

## Goal

Port four OPD-family advantage estimators from the reference implementation
(`reference_*.py` at the repo root) into DeepMath's `verl/` submodule, matching
`DESIGN.md` verbatim. The methods are:

| Method     | Enum / `algorithm.adv_estimator` |
|------------|----------------------------------|
| SRPO       | `grpo_srpo`                      |
| KDRL-MASK  | `grpo_opd_mask`                  |
| HDPO-cliff | `grpo_opd_cliff`                 |
| RLSD       | `grpo_rlsd`                      |

All four share rollout, reward, KL handling, and the actor update with the
existing `grpo_opd` path. They differ only in how `compute_advantage` builds
the advantage tensor.

## Non-goals

- No new functionality outside `compute_advantage` and the reward function.
- No FSDP worker changes — `ref_log_prob` and `old_log_probs` are already in
  `data.batch`.
- No port of the additional reference variants (`grpo_opd_flatten`,
  `grpo_opd_cov`, `grpo_opd_flatten_adv`, `seq_kl_grpo`).
- No automated tests. Verification is by visual diff against the reference.
- No changes to the existing `train_kdrl.sh` (additive `grpo_opd`); the new
  KDRL-MASK script lives at `train_kdrl_mask.sh`.

## Files touched

| File | Change |
|---|---|
| `verl/verl/trainer/ppo/core_algos.py` | Add 4 enum members to `AdvantageEstimator` |
| `verl/verl/trainer/ppo/ray_trainer.py` | Add 4 branches in `compute_advantage`; extend `compute_advantage` signature with `rlsd_eps_w`, `rlsd_lambda`; thread them through the call site; add `correctness_scores` inject/pop around the call; add the four enum members to the rollout-N gate |
| `verl/verl/trainer/config/ppo_trainer.yaml` | Add `rlsd_eps_w: 0.2`, `rlsd_lambda: 1.0` under `algorithm:` |
| `verl/verl/trainer/ppo/ray_trainer.py` (additional, in the same edit) | After the per-rollout `reward_fn(...)` call assembles `reward_tensor`, derive a per-sequence binary `correctness_score ∈ {0.0, 1.0}` from `reward_tensor.sum(-1) > 0` and write it to `new_batch.non_tensor_batch['correctness_scores']` as a numpy `object` array. Gated on `adv_estimator` being one of the OPD-family values. (No edit to `utils/reward_utils/reward_func.py` — the existing reward fn already returns the right per-rollout signal.) |
| `train_srpo.sh` (new) | SLURM entry point, `ADV_ESTIMATOR=grpo_srpo` |
| `train_kdrl_mask.sh` (new) | SLURM entry point, `ADV_ESTIMATOR=grpo_opd_mask` |
| `train_hdpo_cliff.sh` (new) | SLURM entry point, `ADV_ESTIMATOR=grpo_opd_cliff` |
| `train_rlsd.sh` (new) | SLURM entry point, `ADV_ESTIMATOR=grpo_rlsd`, `+algorithm.rlsd_eps_w=0.2`, `+algorithm.rlsd_lambda=1.0` |

## Advantage formulas

Shared inputs already available in `data.batch`:

```
token_level_rewards   # (bs, T)
response_mask         # (bs, T) attention mask sliced to response
student_logprob       # (bs, T) = data.batch['old_log_probs']      detached
teacher_logprob       # (bs, T) = data.batch['ref_log_prob']        detached
correctness_scores    # (bs,)   in [0, 1] — injected from non_tensor_batch
index                 # (bs,)   uid for GRPO grouping
```

Shared preamble:

```python
grpo_adv, _ = core_algos.compute_grpo_outcome_advantage(
    token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index)

opd_mask = response_mask.clone()
last_idx = (opd_mask.long().sum(dim=1) - 1).clamp(min=0)
opd_mask[torch.arange(bs, device=opd_mask.device), last_idx] = 0   # drop EOS

with torch.no_grad():
    opd_term = (teacher_logprob - student_logprob) * opd_mask
```

Per-method advantage:

| Method            | Final advantage Â_t |
|-------------------|----------------------|
| `grpo_srpo`       | `grpo_adv * 1[r_i=1] + opd_beta * opd_term * 1[r_i<1]` |
| `grpo_opd_mask`   | `grpo_adv + opd_beta * opd_term * 1[r_i<1]` |
| `grpo_opd_cliff`  | `grpo_adv + opd_beta * opd_term * 1[max_{j∈group(i)} r_j < 1]` |
| `grpo_rlsd`       | `grpo_adv * ((1−λ) + λ · clip(exp(sign(grpo_adv[:,0]) · δ), 1−ε_w, 1+ε_w)) * response_mask`, where `δ = teacher_logprob − student_logprob` (un-masked; RLSD reweights every response token, not just non-EOS) |

`returns` follows the same structure as the existing `grpo_opd` path:
`returns = grpo_returns` (i.e., the GRPO returns tensor) for the additive
methods; for `grpo_srpo` the same `grpo_returns` is used (matching
`reference_ray_trainer.py:709-742`); for `grpo_rlsd`, `returns =
advantages.clone()` per the reference.

Cliff masking: build `id2max` over `correctness_scores` keyed by `index`; for
each rollout, gate = 1 iff `id2max[index[i]] < 1.0`, else 0.

## `correctness_scores` plumbing

### Trainer-side derivation (`verl/verl/trainer/ppo/ray_trainer.py`)

The existing per-rollout reward fn already returns `score ∈ {-1.0, 1.0}` per
rollout. Right after the trainer assembles `reward_tensor` from
`reward_fn(...)` and assigns it to `new_batch.batch['token_level_scores']`
(currently around line 1245 in the training step, and the analogous block
~line 1347 inside `filter_groups`), derive a per-sequence
`correctness ∈ {0.0, 1.0}` and write it to
`new_batch.non_tensor_batch['correctness_scores']`:

```python
if self.config.algorithm.adv_estimator in (
        AdvantageEstimator.GRPO_OPD, AdvantageEstimator.GRPO_OPD_MASK,
        AdvantageEstimator.GRPO_OPD_CLIFF, AdvantageEstimator.GRPO_SRPO,
        AdvantageEstimator.GRPO_RLSD):
    seq_score = reward_tensor.sum(dim=-1)                   # (bs,)  ∈ {-1, +1}
    correctness = (seq_score > 0).to(torch.float32).cpu().numpy()
    new_batch.non_tensor_batch['correctness_scores'] = np.array(
        correctness.tolist(), dtype=object)
```

Guard: only emit for the OPD-family estimators. For other estimators behavior
is unchanged. No edit to `utils/reward_utils/reward_func.py` is needed — its
existing `score ∈ {-1.0, 1.0}` is enough.

### Driver (`verl/verl/trainer/ppo/ray_trainer.py`)

Around the `compute_advantage` call (currently ~line 1445):

```python
injected = False
if 'correctness_scores' in batch.non_tensor_batch:
    cs_np = batch.non_tensor_batch['correctness_scores']
    batch.batch['correctness_scores'] = torch.as_tensor(
        np.asarray(cs_np.tolist(), dtype=np.float32))
    injected = True

batch = compute_advantage(
    batch,
    adv_estimator=self.config.algorithm.adv_estimator,
    # gamma / lam / num_repeat forwarded from the existing call site, unchanged
    opd_beta=current_opd_beta,
    rlsd_eps_w=self.config.algorithm.get('rlsd_eps_w', 0.2),
    rlsd_lambda=self.config.algorithm.get('rlsd_lambda', 1.0),
)

if injected and 'correctness_scores' in batch.batch:
    batch.batch.pop('correctness_scores')
```

The pop is load-bearing: leaving `correctness_scores` in `batch.batch` across
the `update_actor` dispatch causes Ray/TensorDict to accumulate references and
the loop stalls around step 7-8 (DESIGN.md lines 46-49).

## Config keys

Added under `algorithm:` in `verl/verl/trainer/config/ppo_trainer.yaml`:

```yaml
rlsd_eps_w: 0.2     # per-token reweighting clip for grpo_rlsd
rlsd_lambda: 1.0    # mix factor between identity (0) and full reweighting (1)
```

Existing keys reused unchanged: `opd_beta`, `opd_beta_init`, `opd_beta_end`,
`opd_beta_delta`. The β schedule at `ray_trainer.py:1435-1442` already handles
all three additive methods (SRPO, KDRL-MASK, HDPO-cliff). RLSD does not
consume `opd_beta`.

## `compute_advantage` signature

```python
def compute_advantage(data, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1,
                      opd_beta=0.0, rlsd_eps_w=0.2, rlsd_lambda=1.0):
```

Backwards-compatible — old callers without the RLSD kwargs get the defaults.
Single call site is the trainer (~line 1445).

The estimator-list at `ray_trainer.py:1174-1179` (which routes
GRPO-style estimators away from the GAE/critic path) is extended with
`AdvantageEstimator.GRPO_SRPO`, `GRPO_OPD_MASK`, `GRPO_OPD_CLIFF`,
`GRPO_RLSD`.

## Dynamics logging

Each new branch attaches metrics to `data.meta_info['opd_dynamics']` (same
pattern as the existing `grpo_opd` branch). All metrics surface via the
existing trainer logger.

| Branch | Metrics |
|---|---|
| `grpo_srpo` | `_compute_opd_dynamics_metrics` over `effective_opd = opd_beta * opd_term * incorrect_mask`; plus `dynamics/srpo_rl_fire_rate` (frac with r=1), `dynamics/srpo_opd_fire_rate` (frac with r<1). No signal-covariance metrics. |
| `grpo_opd_mask` | `_compute_opd_dynamics_metrics` and `_compute_signal_covariance_metrics` over `effective_opd = opd_beta * opd_term * incorrect_mask` |
| `grpo_opd_cliff` | `_compute_opd_dynamics_metrics` over `effective_opd = opd_beta * opd_term * cliff_mask`; plus `dynamics/cliff_gate_fire_rate` (frac of rollouts whose group is all-wrong). No signal-covariance metrics. |
| `grpo_rlsd` | `_compute_opd_dynamics_metrics` over `delta * response_mask` (gated on `correctness_scores` being present); plus `dynamics/rlsd_w_mean`, `dynamics/rlsd_w_clip_rate`, `dynamics/rlsd_lambda`, `dynamics/rlsd_sign_pos_rate`, `dynamics/rlsd_sign_neg_rate` |

`_compute_opd_dynamics_metrics` and `_compute_signal_covariance_metrics`
already exist in `ray_trainer.py` and are reused as-is — they accept any OPD
term tensor. Only `grpo_opd_mask` invokes the signal-covariance helper, matching
the reference (lines 522-527). The other three branches use only the lighter
dynamics helper.

## Shell scripts

All four new scripts are copies of `train_kdrl.sh` with these edits:

- `RUN_NAME` updated (e.g., `qwen3-1.7B-Base-qwen3-8B-srpo`).
- `#SBATCH --job-name=` updated to `srpo`, `kdrl-mask`, `hdpo-cliff`, `rlsd`.
- `ADV_ESTIMATOR` set to the matching enum value.
- For SRPO / KDRL-MASK / HDPO-cliff: drop the β-schedule block; set
  `OPD_BETA=0.02`; pass `+algorithm.opd_beta=$OPD_BETA`. (No
  `opd_beta_init/end/delta`.) Matches DESIGN.md defaults.
- For RLSD: drop the `OPD_BETA*` block entirely; add
  `+algorithm.rlsd_eps_w=0.2`, `+algorithm.rlsd_lambda=1.0`.
- `KL_LOSS_COEF=0.0` (matches DESIGN.md).

SLURM headers (a100, 4 GPUs, 400GB, 48h, account, output paths) and all other
flags are identical to `train_kdrl.sh`.

The existing `train_kdrl.sh` (additive `grpo_opd`) is left untouched so prior
runs remain reproducible.

## Verification

Visual diff against `reference_core_algos.py`, `reference_ray_trainer.py`, and
`reference_ray_grpo.py`. Confirm:

- Each new branch in `compute_advantage` matches the corresponding lines in
  the reference (495–528, 669–708, 709–742, 743–788).
- The inject/pop pattern matches `reference_ray_grpo.py:578-651`.
- The `compute_advantage` signature change is forwarded at the single call
  site.
- The reward fn writes the same shape/dtype that the trainer expects.

No unit tests, no smoke run.

## Out of scope / risks

- The reward fn change is gated on the adv_estimator string. If a future
  estimator name is added that needs `correctness_scores`, the gate must be
  updated.
- `correctness_scores` from the reward fn must be the same length as the
  flattened batch (one entry per rollout). The reference does this via a
  numpy object array; we follow the same convention.
- Submodule history: edits to `verl/verl/trainer/...` land in the `bakdop/verl`
  submodule, which then needs a submodule pointer bump in the DeepMath repo
  (the same flow used in commit `7c2d1e1 update verl submodule pointer`).
