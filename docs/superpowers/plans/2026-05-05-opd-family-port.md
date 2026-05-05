# OPD-family port (SRPO, KDRL-MASK, HDPO-cliff, RLSD) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port four OPD-family advantage estimators (`grpo_srpo`, `grpo_opd_mask`, `grpo_opd_cliff`, `grpo_rlsd`) into DeepMath's verl fork, matching the formulas in `DESIGN.md` verbatim.

**Architecture:** All four are inline branches in `compute_advantage` in the verl submodule. They share rollout, reward, KL handling, and the actor update with the existing `grpo_opd` path; they differ only in how the advantage tensor is built. The reward-side change is a one-block trainer-side derivation of binary `correctness_scores` from the existing `reward_tensor`, gated on `adv_estimator`.

**Tech Stack:** Python 3.11, PyTorch, verl (Ray + FSDP), Hydra config. Edits land in two repos: the `verl/` git submodule (`bakdop/verl` branch `deepmath`) and DeepMath proper.

**Verification approach:** No automated tests. Each task ends with `python -c "import …"` to syntax-check the edit, plus a `git diff` review against the relevant section of `reference_*.py`.

**Spec:** `docs/superpowers/specs/2026-05-05-opd-family-port-design.md`.

**Reference files (source of truth for formulas):** `reference_core_algos.py`, `reference_ray_trainer.py`, `reference_ray_grpo.py`.

---

## Preconditions

The `verl/` submodule has uncommitted in-flight edits unrelated to this port (TRRD validation in `verl/trainer/main_ppo.py` and `verl/workers/fsdp_workers.py`, plus a 1-line addition of `AdvantageEstimator.GRPO_OPD` to the `use_critic=False` gate in `verl/trainer/ppo/ray_trainer.py:568-572`). These are not part of this port. **Task 1 commits them as a separate prerequisite commit so the OPD-family edits are clean additions on top.**

The DeepMath repo working tree has unrelated modifications to a few `train_*.sh` files. Those are left alone by this plan.

---

## File structure

| File | Owner | Change |
|---|---|---|
| `verl/verl/trainer/ppo/core_algos.py` | submodule | Add 4 enum members to `AdvantageEstimator` |
| `verl/verl/trainer/ppo/ray_trainer.py` | submodule | Extend `compute_advantage` signature; add 4 new branches; extend the rollout-N gate; derive `correctness_scores` after each `reward_fn(...)` call site (2 locations); inject/pop `correctness_scores` around `compute_advantage`; thread RLSD kwargs |
| `verl/verl/trainer/config/ppo_trainer.yaml` | submodule | Add `rlsd_eps_w: 0.2`, `rlsd_lambda: 1.0` under `algorithm:` |
| `train_srpo.sh` (new) | DeepMath | SLURM entry point |
| `train_kdrl_mask.sh` (new) | DeepMath | SLURM entry point |
| `train_hdpo_cliff.sh` (new) | DeepMath | SLURM entry point |
| `train_rlsd.sh` (new) | DeepMath | SLURM entry point |
| (DeepMath root) | DeepMath | Submodule pointer bump |

Tasks 2–9 commit inside the `verl/` submodule. Tasks 10–14 commit in DeepMath proper. Task 15 bumps the submodule pointer.

---

### Task 1: Commit unrelated in-flight verl edits as a separate prerequisite commit

**Files:**
- Modify (already on disk, just need committing): `verl/verl/trainer/main_ppo.py`, `verl/verl/trainer/ppo/ray_trainer.py`, `verl/verl/workers/fsdp_workers.py`

- [ ] **Step 1: Switch into the submodule and inspect the in-flight diff**

```bash
cd /scratch/bl4363/DeepMath/verl
git status --short
git diff --stat
```

Expected: three modified files, ~9 insertions, ~3 deletions, all TRRD-related.

- [ ] **Step 2: Commit those edits to the submodule as a separate commit**

```bash
cd /scratch/bl4363/DeepMath/verl
git add verl/trainer/main_ppo.py verl/trainer/ppo/ray_trainer.py verl/workers/fsdp_workers.py
git commit -m "$(cat <<'EOF'
trrd: lift ref-required check from worker to main_task; add GRPO_OPD to non-critic estimator gate

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git log -1 --format='%h %s'
```

Expected: a new commit on top of `9721128 opd`. We do NOT bump the DeepMath submodule pointer for this commit yet — that happens together with the OPD-family changes in Task 15.

---

### Task 2: Add four enum members to `AdvantageEstimator`

**Files:**
- Modify: `verl/verl/trainer/ppo/core_algos.py:66-77`

- [ ] **Step 1: Read the current enum definition**

```bash
sed -n '66,78p' /scratch/bl4363/DeepMath/verl/verl/trainer/ppo/core_algos.py
```

Expected: an `AdvantageEstimator(str, Enum)` with members `GAE`, `GRPO`, `REINFORCE_PLUS_PLUS`, `REMAX`, `RLOO`, `OPD`, `OPD_NEG`, `GRPO_OPD`.

- [ ] **Step 2: Add four new members at the end of the enum**

Edit `verl/verl/trainer/ppo/core_algos.py`. Replace:

```python
    GRPO_OPD = 'grpo_opd'
```

with:

```python
    GRPO_OPD = 'grpo_opd'
    GRPO_OPD_MASK = 'grpo_opd_mask'        # KDRL-MASK: GRPO + teacher term on incorrect rollouts
    GRPO_OPD_CLIFF = 'grpo_opd_cliff'      # HDPO-cliff: teacher term only on all-wrong groups
    GRPO_SRPO = 'grpo_srpo'                # SRPO: GRPO on correct, OPD on incorrect (disjoint)
    GRPO_RLSD = 'grpo_rlsd'                # RLSD: sign-preserving token reweighting
```

- [ ] **Step 3: Syntax check**

```bash
python3 -c "from verl.trainer.ppo.core_algos import AdvantageEstimator; print([m.value for m in AdvantageEstimator])"
```

(Run from `/scratch/bl4363/DeepMath` with `source /scratch/bl4363/uvenvs/dm/bin/activate` first if not active.)

Expected output includes `'grpo_opd_mask'`, `'grpo_opd_cliff'`, `'grpo_srpo'`, `'grpo_rlsd'`.

- [ ] **Step 4: Commit (inside submodule)**

```bash
cd /scratch/bl4363/DeepMath/verl
git add verl/trainer/ppo/core_algos.py
git commit -m "$(cat <<'EOF'
opd: add GRPO_OPD_MASK / GRPO_OPD_CLIFF / GRPO_SRPO / GRPO_RLSD enum members

Enum-only addition. Branches in compute_advantage and the use_critic gate
follow in subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add the four new estimators to the `use_critic=False` gate

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py:567-576`

- [ ] **Step 1: Read the current gate**

```bash
sed -n '567,576p' /scratch/bl4363/DeepMath/verl/verl/trainer/ppo/ray_trainer.py
```

Expected (after Task 1):

```python
        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO, AdvantageEstimator.OPD, AdvantageEstimator.OPD_NEG,
                AdvantageEstimator.GRPO_OPD
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError
```

- [ ] **Step 2: Extend the list with the four new members**

Edit `verl/verl/trainer/ppo/ray_trainer.py`. Replace:

```python
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO, AdvantageEstimator.OPD, AdvantageEstimator.OPD_NEG,
                AdvantageEstimator.GRPO_OPD
        ]:
```

with:

```python
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO, AdvantageEstimator.OPD, AdvantageEstimator.OPD_NEG,
                AdvantageEstimator.GRPO_OPD, AdvantageEstimator.GRPO_OPD_MASK,
                AdvantageEstimator.GRPO_OPD_CLIFF, AdvantageEstimator.GRPO_SRPO,
                AdvantageEstimator.GRPO_RLSD
        ]:
```

- [ ] **Step 3: Syntax check**

```bash
python3 -c "import verl.trainer.ppo.ray_trainer as m; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Commit (inside submodule)**

```bash
cd /scratch/bl4363/DeepMath/verl
git add verl/trainer/ppo/ray_trainer.py
git commit -m "$(cat <<'EOF'
opd: route GRPO_OPD_MASK / GRPO_OPD_CLIFF / GRPO_SRPO / GRPO_RLSD through non-critic path

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Extend `compute_advantage` signature with `rlsd_eps_w` and `rlsd_lambda`

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py:347` (signature) and `:1444-1449` (call site)

- [ ] **Step 1: Update the signature**

Edit `verl/verl/trainer/ppo/ray_trainer.py`. Replace:

```python
def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, opd_beta=0.0):
```

with:

```python
def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1,
                      opd_beta=0.0, rlsd_eps_w=0.2, rlsd_lambda=1.0):
```

- [ ] **Step 2: Forward the new kwargs at the single call site**

Replace (currently around line 1444-1449):

```python
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  opd_beta=current_opd_beta)
```

with:

```python
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  opd_beta=current_opd_beta,
                                                  rlsd_eps_w=self.config.algorithm.get('rlsd_eps_w', 0.2),
                                                  rlsd_lambda=self.config.algorithm.get('rlsd_lambda', 1.0))
```

- [ ] **Step 3: Syntax check**

```bash
python3 -c "from verl.trainer.ppo.ray_trainer import compute_advantage; import inspect; print(list(inspect.signature(compute_advantage).parameters))"
```

Expected: `['data', 'adv_estimator', 'gamma', 'lam', 'num_repeat', 'opd_beta', 'rlsd_eps_w', 'rlsd_lambda']`.

- [ ] **Step 4: Commit (inside submodule)**

```bash
cd /scratch/bl4363/DeepMath/verl
git add verl/trainer/ppo/ray_trainer.py
git commit -m "$(cat <<'EOF'
opd: thread rlsd_eps_w/rlsd_lambda through compute_advantage

Backwards-compatible — defaults match the reference (eps_w=0.2, lambda=1.0).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Add the `grpo_opd_mask` (KDRL-MASK) branch

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py` — insert a new `elif` immediately after the existing `GRPO_OPD` branch (after line 502, before the `else: raise NotImplementedError`)

- [ ] **Step 1: Locate the insertion point**

```bash
sed -n '500,505p' /scratch/bl4363/DeepMath/verl/verl/trainer/ppo/ray_trainer.py
```

Expected: the tail of the existing `GRPO_OPD` branch ending with `data.meta_info['opd_dynamics'] = dyn`, immediately followed by `else: raise NotImplementedError`.

- [ ] **Step 2: Insert the KDRL-MASK branch**

Edit `verl/verl/trainer/ppo/ray_trainer.py`. Replace:

```python
            data.meta_info['opd_dynamics'] = dyn
    else:
        raise NotImplementedError
    return data
```

with:

```python
            data.meta_info['opd_dynamics'] = dyn
    elif adv_estimator == AdvantageEstimator.GRPO_OPD_MASK:
        # KDRL-MASK: GRPO advantage on every rollout; teacher term added only on
        # incorrect rollouts (correctness_scores < 1.0). Mirrors
        # reference_ray_trainer.py:499-528.
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask_grpo = attention_mask[:, -response_length:]
        student_logprob = data.batch['old_log_probs']
        teacher_logprob = data.batch['ref_log_prob']
        opd_mask = response_mask_grpo.clone()
        lengths = opd_mask.long().sum(dim=1)
        last_idx = (lengths - 1).clamp(min=0)
        opd_mask[torch.arange(opd_mask.size(0), device=opd_mask.device), last_idx] = 0
        with torch.no_grad():
            grpo_adv, grpo_returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards, eos_mask=response_mask_grpo, index=index)
            correctness_scores = data.batch['correctness_scores']                # (bs,) ∈ [0,1]
            incorrect_mask = (correctness_scores < 1.0).float()                  # (bs,)
            opd_term = (teacher_logprob - student_logprob) * opd_mask
            advantages = grpo_adv + opd_beta * opd_term * incorrect_mask.unsqueeze(-1)
        data.batch['advantages'] = advantages
        data.batch['returns'] = grpo_returns
        effective_opd = opd_beta * opd_term * incorrect_mask.unsqueeze(-1)
        dyn = _compute_opd_dynamics_metrics(
            correctness_scores, teacher_logprob, student_logprob,
            grpo_adv, effective_opd, advantages, response_mask_grpo)
        dyn.update(_compute_signal_covariance_metrics(
            correctness_scores, index, grpo_adv, effective_opd,
            opd_mask, response_mask_grpo))
        data.meta_info['opd_dynamics'] = dyn
    else:
        raise NotImplementedError
    return data
```

- [ ] **Step 3: Syntax check**

```bash
python3 -c "import verl.trainer.ppo.ray_trainer as m; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Diff the new branch against the reference**

```bash
diff <(sed -n '/elif adv_estimator == AdvantageEstimator\.GRPO_OPD_MASK:/,/data\.meta_info\[.opd_dynamics.\] = dyn$/p' /scratch/bl4363/DeepMath/verl/verl/trainer/ppo/ray_trainer.py) \
     <(sed -n '499,528p' /scratch/bl4363/DeepMath/reference_ray_trainer.py)
```

Expected: only cosmetic differences (e.g., we use `response_mask_grpo.clone()` directly instead of re-slicing the attention mask). Logic identical.

- [ ] **Step 5: Commit (inside submodule)**

```bash
cd /scratch/bl4363/DeepMath/verl
git add verl/trainer/ppo/ray_trainer.py
git commit -m "$(cat <<'EOF'
opd: add GRPO_OPD_MASK (KDRL-MASK) branch in compute_advantage

GRPO advantage on every rollout; β · (log π_T − log π_S) added only on
incorrect rollouts. Mirrors reference_ray_trainer.py:499-528.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Add the `grpo_opd_cliff` (HDPO-cliff) branch

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py` — insert immediately after the `GRPO_OPD_MASK` branch from Task 5

- [ ] **Step 1: Insert the cliff branch**

Edit `verl/verl/trainer/ppo/ray_trainer.py`. Find the line `data.meta_info['opd_dynamics'] = dyn` immediately preceding `else: raise NotImplementedError` (this is now the tail of the `GRPO_OPD_MASK` branch added in Task 5). Replace:

```python
        dyn.update(_compute_signal_covariance_metrics(
            correctness_scores, index, grpo_adv, effective_opd,
            opd_mask, response_mask_grpo))
        data.meta_info['opd_dynamics'] = dyn
    else:
        raise NotImplementedError
    return data
```

with:

```python
        dyn.update(_compute_signal_covariance_metrics(
            correctness_scores, index, grpo_adv, effective_opd,
            opd_mask, response_mask_grpo))
        data.meta_info['opd_dynamics'] = dyn
    elif adv_estimator == AdvantageEstimator.GRPO_OPD_CLIFF:
        # HDPO-cliff: teacher term added only on groups where every rollout
        # failed (max correctness < 1.0). Mirrors reference_ray_trainer.py:669-708.
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask_grpo = attention_mask[:, -response_length:]
        student_logprob = data.batch['old_log_probs']
        teacher_logprob = data.batch['ref_log_prob']
        opd_mask = response_mask_grpo.clone()
        lengths = opd_mask.long().sum(dim=1)
        last_idx = (lengths - 1).clamp(min=0)
        opd_mask[torch.arange(opd_mask.size(0), device=opd_mask.device), last_idx] = 0
        with torch.no_grad():
            grpo_adv, grpo_returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards, eos_mask=response_mask_grpo, index=index)
            correctness_scores = data.batch['correctness_scores']                # (bs,)
            # Build per-group max-correctness, then a per-rollout cliff gate.
            id2max = {}
            for gid, c in zip(index, correctness_scores.tolist()):
                gid_key = gid.item() if hasattr(gid, 'item') else gid
                if gid_key not in id2max or c > id2max[gid_key]:
                    id2max[gid_key] = c
            cliff_vals = [
                1.0 if id2max[(gid.item() if hasattr(gid, 'item') else gid)] < 1.0 else 0.0
                for gid in index
            ]
            cliff_mask = torch.tensor(cliff_vals, dtype=grpo_adv.dtype,
                                       device=grpo_adv.device)                    # (bs,)
            opd_term = (teacher_logprob - student_logprob) * opd_mask
            advantages = grpo_adv + opd_beta * opd_term * cliff_mask.unsqueeze(-1)
        data.batch['advantages'] = advantages
        data.batch['returns'] = grpo_returns
        effective_opd = opd_beta * opd_term * cliff_mask.unsqueeze(-1)
        dyn = _compute_opd_dynamics_metrics(
            correctness_scores, teacher_logprob, student_logprob,
            grpo_adv, effective_opd, advantages, response_mask_grpo)
        dyn['dynamics/cliff_gate_fire_rate'] = cliff_mask.mean().item()
        data.meta_info['opd_dynamics'] = dyn
    else:
        raise NotImplementedError
    return data
```

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import verl.trainer.ppo.ray_trainer as m; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Diff the new branch against the reference**

```bash
diff <(sed -n '/elif adv_estimator == AdvantageEstimator\.GRPO_OPD_CLIFF:/,/dyn\[.dynamics\/cliff_gate_fire_rate.\] = cliff_mask\.mean()/p' /scratch/bl4363/DeepMath/verl/verl/trainer/ppo/ray_trainer.py) \
     <(sed -n '669,707p' /scratch/bl4363/DeepMath/reference_ray_trainer.py)
```

Expected: cosmetic differences only.

- [ ] **Step 4: Commit (inside submodule)**

```bash
cd /scratch/bl4363/DeepMath/verl
git add verl/trainer/ppo/ray_trainer.py
git commit -m "$(cat <<'EOF'
opd: add GRPO_OPD_CLIFF (HDPO-cliff) branch in compute_advantage

Teacher term added only on groups where every rollout failed
(max correctness < 1.0). Mirrors reference_ray_trainer.py:669-708.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Add the `grpo_srpo` (SRPO) branch

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py` — insert immediately after the `GRPO_OPD_CLIFF` branch from Task 6

- [ ] **Step 1: Insert the SRPO branch**

Edit `verl/verl/trainer/ppo/ray_trainer.py`. Replace the tail of the cliff branch + the `else` block:

```python
        dyn['dynamics/cliff_gate_fire_rate'] = cliff_mask.mean().item()
        data.meta_info['opd_dynamics'] = dyn
    else:
        raise NotImplementedError
    return data
```

with:

```python
        dyn['dynamics/cliff_gate_fire_rate'] = cliff_mask.mean().item()
        data.meta_info['opd_dynamics'] = dyn
    elif adv_estimator == AdvantageEstimator.GRPO_SRPO:
        # SRPO: GRPO advantage contributes only on correct rollouts (r_i = 1);
        # teacher term contributes only on incorrect rollouts (r_i < 1). The
        # two paths are disjoint over samples. Mirrors
        # reference_ray_trainer.py:709-742.
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask_grpo = attention_mask[:, -response_length:]
        student_logprob = data.batch['old_log_probs']
        teacher_logprob = data.batch['ref_log_prob']
        opd_mask = response_mask_grpo.clone()
        lengths = opd_mask.long().sum(dim=1)
        last_idx = (lengths - 1).clamp(min=0)
        opd_mask[torch.arange(opd_mask.size(0), device=opd_mask.device), last_idx] = 0
        with torch.no_grad():
            grpo_adv, grpo_returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards, eos_mask=response_mask_grpo, index=index)
            correctness_scores = data.batch['correctness_scores']                # (bs,)
            correct_mask = (correctness_scores >= 1.0).to(grpo_adv.dtype)        # (bs,)
            incorrect_mask = 1.0 - correct_mask
            opd_term = (teacher_logprob - student_logprob) * opd_mask
            advantages = (grpo_adv * correct_mask.unsqueeze(-1)
                          + opd_beta * opd_term * incorrect_mask.unsqueeze(-1))
        data.batch['advantages'] = advantages
        data.batch['returns'] = grpo_returns
        effective_opd = opd_beta * opd_term * incorrect_mask.unsqueeze(-1)
        dyn = _compute_opd_dynamics_metrics(
            correctness_scores, teacher_logprob, student_logprob,
            grpo_adv, effective_opd, advantages, response_mask_grpo)
        dyn['dynamics/srpo_rl_fire_rate'] = correct_mask.mean().item()
        dyn['dynamics/srpo_opd_fire_rate'] = incorrect_mask.mean().item()
        data.meta_info['opd_dynamics'] = dyn
    else:
        raise NotImplementedError
    return data
```

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import verl.trainer.ppo.ray_trainer as m; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Diff against the reference**

```bash
diff <(sed -n '/elif adv_estimator == AdvantageEstimator\.GRPO_SRPO:/,/dyn\[.dynamics\/srpo_opd_fire_rate.\] = incorrect_mask\.mean()/p' /scratch/bl4363/DeepMath/verl/verl/trainer/ppo/ray_trainer.py) \
     <(sed -n '709,741p' /scratch/bl4363/DeepMath/reference_ray_trainer.py)
```

Expected: cosmetic differences only.

- [ ] **Step 4: Commit (inside submodule)**

```bash
cd /scratch/bl4363/DeepMath/verl
git add verl/trainer/ppo/ray_trainer.py
git commit -m "$(cat <<'EOF'
opd: add GRPO_SRPO branch in compute_advantage

Disjoint routing: GRPO on correct rollouts, β · (log π_T − log π_S) on
incorrect rollouts. Mirrors reference_ray_trainer.py:709-742.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Add the `grpo_rlsd` (RLSD) branch

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py` — insert immediately after the `GRPO_SRPO` branch from Task 7

- [ ] **Step 1: Insert the RLSD branch**

Edit `verl/verl/trainer/ppo/ray_trainer.py`. Replace:

```python
        dyn['dynamics/srpo_rl_fire_rate'] = correct_mask.mean().item()
        dyn['dynamics/srpo_opd_fire_rate'] = incorrect_mask.mean().item()
        data.meta_info['opd_dynamics'] = dyn
    else:
        raise NotImplementedError
    return data
```

with:

```python
        dyn['dynamics/srpo_rl_fire_rate'] = correct_mask.mean().item()
        dyn['dynamics/srpo_opd_fire_rate'] = incorrect_mask.mean().item()
        data.meta_info['opd_dynamics'] = dyn
    elif adv_estimator == AdvantageEstimator.GRPO_RLSD:
        # RLSD (arXiv 2604.03128): sign-preserving token reweighting. Sign of
        # every token-level advantage comes from the GRPO/reward signal; the
        # teacher reshapes magnitude only. No opd_beta. Mirrors
        # reference_ray_trainer.py:743-788.
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask_grpo = attention_mask[:, -response_length:]
        student_logprob = data.batch['old_log_probs']
        teacher_logprob = data.batch['ref_log_prob']
        with torch.no_grad():
            grpo_adv, _ = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards, eos_mask=response_mask_grpo, index=index)
            seq_adv = grpo_adv[:, 0]                                              # (bs,)
            sign_a = torch.sign(seq_adv).to(student_logprob.dtype)                # (bs,) ∈ {-1, 0, +1}
            delta = teacher_logprob - student_logprob                             # (bs, T) detached
            w = torch.exp(sign_a.unsqueeze(-1) * delta)                           # (bs, T)
            w_clip = w.clamp(min=1.0 - rlsd_eps_w, max=1.0 + rlsd_eps_w)
            blend = (1.0 - rlsd_lambda) + rlsd_lambda * w_clip                    # (bs, T)
            advantages = (grpo_adv * blend) * response_mask_grpo
            returns = advantages.clone()
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        if 'correctness_scores' in data.batch:
            correctness_scores = data.batch['correctness_scores']
            opd_term_for_log = delta * response_mask_grpo
            dyn = _compute_opd_dynamics_metrics(
                correctness_scores, teacher_logprob, student_logprob,
                grpo_adv, opd_term_for_log, advantages, response_mask_grpo)
            denom = response_mask_grpo.sum().clamp(min=1)
            clip_hits = ((w >= 1.0 + rlsd_eps_w) | (w <= 1.0 - rlsd_eps_w)).float() * response_mask_grpo
            dyn['dynamics/rlsd_w_mean'] = (w * response_mask_grpo).sum().item() / denom.item()
            dyn['dynamics/rlsd_w_clip_rate'] = clip_hits.sum().item() / denom.item()
            dyn['dynamics/rlsd_lambda'] = float(rlsd_lambda)
            dyn['dynamics/rlsd_sign_pos_rate'] = (sign_a > 0).float().mean().item()
            dyn['dynamics/rlsd_sign_neg_rate'] = (sign_a < 0).float().mean().item()
            data.meta_info['opd_dynamics'] = dyn
    else:
        raise NotImplementedError
    return data
```

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import verl.trainer.ppo.ray_trainer as m; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Diff against the reference**

```bash
diff <(sed -n '/elif adv_estimator == AdvantageEstimator\.GRPO_RLSD:/,/data\.meta_info\[.opd_dynamics.\] = dyn$/p' /scratch/bl4363/DeepMath/verl/verl/trainer/ppo/ray_trainer.py) \
     <(sed -n '743,788p' /scratch/bl4363/DeepMath/reference_ray_trainer.py)
```

Expected: cosmetic differences only.

- [ ] **Step 4: Commit (inside submodule)**

```bash
cd /scratch/bl4363/DeepMath/verl
git add verl/trainer/ppo/ray_trainer.py
git commit -m "$(cat <<'EOF'
opd: add GRPO_RLSD (sign-preserving token reweighting) branch in compute_advantage

Mirrors arXiv 2604.03128 Algorithm 1 with fixed mixing constant lambda. No
opd_beta. Mirrors reference_ray_trainer.py:743-788.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Derive `correctness_scores` after each `reward_fn(...)` call

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py:1237-1245` and `:1339-1347`

There are two locations in the trainer that call `self.reward_fn(new_batch, return_dict=True)` and assign `reward_tensor` to `new_batch.batch['token_level_scores']`: the main training step (~line 1237-1245) and the analogous block inside the `filter_groups` retry loop (~line 1339-1347).

- [ ] **Step 1: Confirm both call sites still match the snippet shape**

```bash
grep -n "new_batch.batch\['token_level_scores'\] = reward_tensor" /scratch/bl4363/DeepMath/verl/verl/trainer/ppo/ray_trainer.py
```

Expected: two matches near lines 1245 and 1347.

- [ ] **Step 2: Add the derivation block at both call sites**

For each occurrence of:

```python
                            new_batch.batch['token_level_scores'] = reward_tensor
```

insert immediately afterward (preserving leading indentation — match the indent of the line above):

```python
                            # OPD-family: emit per-sequence binary correctness derived from the
                            # rule-based reward (score in {-1, +1} -> correctness in {0, 1}).
                            if self.config.algorithm.adv_estimator in (
                                    AdvantageEstimator.GRPO_OPD,
                                    AdvantageEstimator.GRPO_OPD_MASK,
                                    AdvantageEstimator.GRPO_OPD_CLIFF,
                                    AdvantageEstimator.GRPO_SRPO,
                                    AdvantageEstimator.GRPO_RLSD):
                                seq_score = reward_tensor.sum(dim=-1)
                                correctness = (seq_score > 0).to(torch.float32).cpu().numpy()
                                new_batch.non_tensor_batch['correctness_scores'] = np.array(
                                    correctness.tolist(), dtype=object)
```

(Indent must match the surrounding block. The block at ~line 1245 is inside `with _timer('reward', timing_raw):` at one nesting level; the block at ~line 1347 is inside the `filter_groups` retry loop at a deeper nesting level. Use whatever indent the existing `new_batch.batch['token_level_scores'] = reward_tensor` line uses, then preserve it for the inserted lines.)

- [ ] **Step 3: Confirm `numpy as np` is already imported at the top of the file**

```bash
grep -n "^import numpy" /scratch/bl4363/DeepMath/verl/verl/trainer/ppo/ray_trainer.py
```

Expected: `import numpy as np` is present. If not, add it to the import block.

- [ ] **Step 4: Syntax check**

```bash
python3 -c "import verl.trainer.ppo.ray_trainer as m; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Commit (inside submodule)**

```bash
cd /scratch/bl4363/DeepMath/verl
git add verl/trainer/ppo/ray_trainer.py
git commit -m "$(cat <<'EOF'
opd: derive per-sequence correctness_scores from reward_tensor for OPD-family estimators

Gated on adv_estimator. Writes binary {0, 1} per sequence into
non_tensor_batch['correctness_scores'] at both reward_fn call sites
(main step and filter_groups retry).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Inject/pop `correctness_scores` around `compute_advantage`

**Files:**
- Modify: `verl/verl/trainer/ppo/ray_trainer.py:1433-1449` (the `with _timer('adv', ...)` block)

Per `DESIGN.md` lines 46-49: leaving `correctness_scores` in `data.batch` across the `update_actor` dispatch causes Ray/TensorDict to accumulate references and the loop stalls around step 7-8. Inject as a tensor before `compute_advantage`, pop after.

- [ ] **Step 1: Read the current adv block**

```bash
sed -n '1433,1452p' /scratch/bl4363/DeepMath/verl/verl/trainer/ppo/ray_trainer.py
```

- [ ] **Step 2: Wrap the `compute_advantage` call with inject/pop**

Edit `verl/verl/trainer/ppo/ray_trainer.py`. Replace (after Task 4's signature change):

```python
                    with _timer('adv', timing_raw):
                        # compute advantages, executed on the driver process
                        # Linear beta schedule: if opd_beta_init is set, compute current_beta per step
                        opd_beta_init = self.config.algorithm.get('opd_beta_init', None)
                        if opd_beta_init is not None:
                            opd_beta_end   = self.config.algorithm.get('opd_beta_end', 0.0)
                            opd_beta_delta = self.config.algorithm.get('opd_beta_delta', 0.0)
                            current_opd_beta = max(opd_beta_init - opd_beta_delta * self.global_steps, opd_beta_end)
                        else:
                            current_opd_beta = self.config.algorithm.get('opd_beta', 0.0)
                        metrics['opd/beta'] = current_opd_beta
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  opd_beta=current_opd_beta,
                                                  rlsd_eps_w=self.config.algorithm.get('rlsd_eps_w', 0.2),
                                                  rlsd_lambda=self.config.algorithm.get('rlsd_lambda', 1.0))
```

with:

```python
                    with _timer('adv', timing_raw):
                        # compute advantages, executed on the driver process
                        # Linear beta schedule: if opd_beta_init is set, compute current_beta per step
                        opd_beta_init = self.config.algorithm.get('opd_beta_init', None)
                        if opd_beta_init is not None:
                            opd_beta_end   = self.config.algorithm.get('opd_beta_end', 0.0)
                            opd_beta_delta = self.config.algorithm.get('opd_beta_delta', 0.0)
                            current_opd_beta = max(opd_beta_init - opd_beta_delta * self.global_steps, opd_beta_end)
                        else:
                            current_opd_beta = self.config.algorithm.get('opd_beta', 0.0)
                        metrics['opd/beta'] = current_opd_beta

                        # Inject correctness_scores into batch.batch only for the duration of
                        # compute_advantage. Leaving it in batch.batch across update_actor
                        # causes Ray/TensorDict to accumulate references and the loop stalls
                        # around step 7-8 (see DESIGN.md lines 46-49).
                        injected_correctness = False
                        if 'correctness_scores' in batch.non_tensor_batch:
                            cs_np = batch.non_tensor_batch['correctness_scores']
                            batch.batch['correctness_scores'] = torch.as_tensor(
                                np.asarray(cs_np.tolist(), dtype=np.float32))
                            injected_correctness = True

                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  opd_beta=current_opd_beta,
                                                  rlsd_eps_w=self.config.algorithm.get('rlsd_eps_w', 0.2),
                                                  rlsd_lambda=self.config.algorithm.get('rlsd_lambda', 1.0))

                        if injected_correctness and 'correctness_scores' in batch.batch.keys():
                            batch.batch.pop('correctness_scores')
```

- [ ] **Step 3: Syntax check**

```bash
python3 -c "import verl.trainer.ppo.ray_trainer as m; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Commit (inside submodule)**

```bash
cd /scratch/bl4363/DeepMath/verl
git add verl/trainer/ppo/ray_trainer.py
git commit -m "$(cat <<'EOF'
opd: inject/pop correctness_scores around compute_advantage to avoid Ray/TensorDict stall

Promotes the value from non_tensor_batch to batch.batch as a float32 tensor
for the duration of compute_advantage, then pops it. Per DESIGN.md, leaving
it in batch.batch across update_actor causes the training loop to stall
around step 7-8.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Add `rlsd_eps_w` and `rlsd_lambda` defaults to the Hydra config

**Files:**
- Modify: `verl/verl/trainer/config/ppo_trainer.yaml:184-191`

- [ ] **Step 1: Read the current `algorithm:` block**

```bash
sed -n '184,200p' /scratch/bl4363/DeepMath/verl/verl/trainer/config/ppo_trainer.yaml
```

- [ ] **Step 2: Add the two new keys**

Edit `verl/verl/trainer/config/ppo_trainer.yaml`. Replace:

```yaml
  opd_beta_delta: 0.0 # decrement per global step
  kl_penalty: kl  # how to estimate kl divergence
```

with:

```yaml
  opd_beta_delta: 0.0 # decrement per global step
  rlsd_eps_w: 0.2     # per-token reweighting clip for grpo_rlsd
  rlsd_lambda: 1.0    # mix factor between identity (0) and full reweighting (1) for grpo_rlsd
  kl_penalty: kl  # how to estimate kl divergence
```

- [ ] **Step 3: Syntax check**

```bash
python3 -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('/scratch/bl4363/DeepMath/verl/verl/trainer/config/ppo_trainer.yaml'); print(cfg.algorithm.rlsd_eps_w, cfg.algorithm.rlsd_lambda)"
```

Expected: `0.2 1.0`.

- [ ] **Step 4: Commit (inside submodule)**

```bash
cd /scratch/bl4363/DeepMath/verl
git add verl/trainer/config/ppo_trainer.yaml
git commit -m "$(cat <<'EOF'
opd: add rlsd_eps_w / rlsd_lambda defaults to ppo_trainer.yaml

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Create `train_kdrl_mask.sh`

**Files:**
- Create: `train_kdrl_mask.sh` (in DeepMath repo root)

- [ ] **Step 1: Write the new script**

Create `/scratch/bl4363/DeepMath/train_kdrl_mask.sh` with this content:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=kdrl-mask
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err

source /scratch/bl4363/uvenvs/deepmath/bin/activate

set -e
set -u


WORK_DIR=/scratch/bl4363/DeepMath
MODEL_DIR=$WORK_DIR/models
DATA_DIR=/scratch/bl4363/DeepMath/data
RUN_NAME=qwen3-1.7B-Base-qwen3-8B-kdrl-mask
mkdir -p $MODEL_DIR/$RUN_NAME

export WANDB_API_KEY=aa7783a61740a28c4310d058e383e96dbc08cf97
export WANDB_OFFICIAL=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export ARNOLD_WORKER_NUM=1

ADV_ESTIMATOR=grpo_opd_mask
OPD_BETA=0.02
MAX_RESPONSE_LENGTH=8192
MODEL_PATH=/scratch/bl4363/models/Qwen3-1.7B-Base
REF_MODEL_PATH=/scratch/bl4363/models/Qwen3-8B
KL_LOSS_COEF=0.0
OVERLONG_ENABLE=False
OVERLONG_BUFFER=1024
PPO_MINI_BATCH_SIZE=64
python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.rm_system_prompt=False \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.opd_beta=$OPD_BETA \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.ref.path=$REF_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.disable_log_stats=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    custom_reward_function.path=$WORK_DIR/utils/reward_utils/reward_func.py \
    custom_reward_function.name=reward_func \
    custom_reward_function.overlong_buffer.enable=$OVERLONG_ENABLE \
    trainer.critic_warmup=0 \
    trainer.project_name=deepmath-new \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=$MODEL_DIR/deepmath-new/$RUN_NAME \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=$ARNOLD_WORKER_NUM \
    trainer.save_freq=10 \
    trainer.save_rollout=True \
    trainer.test_freq=2 \
    trainer.total_epochs=999999 \
    trainer.total_training_steps=1800 2>&1 | tee -a $MODEL_DIR/$RUN_NAME/train.log
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x /scratch/bl4363/DeepMath/train_kdrl_mask.sh
```

- [ ] **Step 3: Lint with bash -n**

```bash
bash -n /scratch/bl4363/DeepMath/train_kdrl_mask.sh && echo ok
```

Expected: `ok`.

- [ ] **Step 4: Commit (in DeepMath repo)**

```bash
cd /scratch/bl4363/DeepMath
git add train_kdrl_mask.sh
git commit -m "$(cat <<'EOF'
scripts: add train_kdrl_mask.sh (grpo_opd_mask / KDRL-MASK)

Modeled on train_kdrl.sh; uses adv_estimator=grpo_opd_mask with fixed beta=0.02.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Create `train_hdpo_cliff.sh`

**Files:**
- Create: `train_hdpo_cliff.sh` (in DeepMath repo root)

- [ ] **Step 1: Write the new script**

Copy `train_kdrl_mask.sh` from Task 12 and change:
- `--job-name=kdrl-mask` → `--job-name=hdpo-cliff`
- `RUN_NAME=qwen3-1.7B-Base-qwen3-8B-kdrl-mask` → `RUN_NAME=qwen3-1.7B-Base-qwen3-8B-hdpo-cliff`
- `ADV_ESTIMATOR=grpo_opd_mask` → `ADV_ESTIMATOR=grpo_opd_cliff`

All other content identical.

```bash
cp /scratch/bl4363/DeepMath/train_kdrl_mask.sh /scratch/bl4363/DeepMath/train_hdpo_cliff.sh
sed -i 's/--job-name=kdrl-mask/--job-name=hdpo-cliff/' /scratch/bl4363/DeepMath/train_hdpo_cliff.sh
sed -i 's/qwen3-8B-kdrl-mask/qwen3-8B-hdpo-cliff/' /scratch/bl4363/DeepMath/train_hdpo_cliff.sh
sed -i 's/ADV_ESTIMATOR=grpo_opd_mask/ADV_ESTIMATOR=grpo_opd_cliff/' /scratch/bl4363/DeepMath/train_hdpo_cliff.sh
```

- [ ] **Step 2: Verify the changes**

```bash
grep -E '^#SBATCH --job-name|^RUN_NAME|^ADV_ESTIMATOR' /scratch/bl4363/DeepMath/train_hdpo_cliff.sh
```

Expected:
```
#SBATCH --job-name=hdpo-cliff
RUN_NAME=qwen3-1.7B-Base-qwen3-8B-hdpo-cliff
ADV_ESTIMATOR=grpo_opd_cliff
```

- [ ] **Step 3: Lint**

```bash
bash -n /scratch/bl4363/DeepMath/train_hdpo_cliff.sh && echo ok
```

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
cd /scratch/bl4363/DeepMath
git add train_hdpo_cliff.sh
git commit -m "$(cat <<'EOF'
scripts: add train_hdpo_cliff.sh (grpo_opd_cliff / HDPO-cliff)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Create `train_srpo.sh`

**Files:**
- Create: `train_srpo.sh` (in DeepMath repo root)

- [ ] **Step 1: Write the new script**

```bash
cp /scratch/bl4363/DeepMath/train_kdrl_mask.sh /scratch/bl4363/DeepMath/train_srpo.sh
sed -i 's/--job-name=kdrl-mask/--job-name=srpo/' /scratch/bl4363/DeepMath/train_srpo.sh
sed -i 's/qwen3-8B-kdrl-mask/qwen3-8B-srpo/' /scratch/bl4363/DeepMath/train_srpo.sh
sed -i 's/ADV_ESTIMATOR=grpo_opd_mask/ADV_ESTIMATOR=grpo_srpo/' /scratch/bl4363/DeepMath/train_srpo.sh
```

- [ ] **Step 2: Verify the changes**

```bash
grep -E '^#SBATCH --job-name|^RUN_NAME|^ADV_ESTIMATOR' /scratch/bl4363/DeepMath/train_srpo.sh
```

Expected:
```
#SBATCH --job-name=srpo
RUN_NAME=qwen3-1.7B-Base-qwen3-8B-srpo
ADV_ESTIMATOR=grpo_srpo
```

- [ ] **Step 3: Lint**

```bash
bash -n /scratch/bl4363/DeepMath/train_srpo.sh && echo ok
```

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
cd /scratch/bl4363/DeepMath
git add train_srpo.sh
git commit -m "$(cat <<'EOF'
scripts: add train_srpo.sh (grpo_srpo / SRPO)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Create `train_rlsd.sh`

**Files:**
- Create: `train_rlsd.sh` (in DeepMath repo root)

RLSD does not consume `opd_beta`; instead it consumes `rlsd_eps_w` and `rlsd_lambda`. The script drops the `OPD_BETA` block and adds the two RLSD overrides.

- [ ] **Step 1: Write the script**

Create `/scratch/bl4363/DeepMath/train_rlsd.sh` by copying `train_kdrl_mask.sh` and substituting the algorithm-specific block. The diff vs. `train_kdrl_mask.sh`:
- `--job-name=kdrl-mask` → `--job-name=rlsd`
- `RUN_NAME=qwen3-1.7B-Base-qwen3-8B-kdrl-mask` → `RUN_NAME=qwen3-1.7B-Base-qwen3-8B-rlsd`
- `ADV_ESTIMATOR=grpo_opd_mask` → `ADV_ESTIMATOR=grpo_rlsd`
- Remove the `OPD_BETA=0.02` line.
- Replace `algorithm.opd_beta=$OPD_BETA \` with `+algorithm.rlsd_eps_w=0.2 \` followed by `+algorithm.rlsd_lambda=1.0 \`.

```bash
cp /scratch/bl4363/DeepMath/train_kdrl_mask.sh /scratch/bl4363/DeepMath/train_rlsd.sh
sed -i 's/--job-name=kdrl-mask/--job-name=rlsd/' /scratch/bl4363/DeepMath/train_rlsd.sh
sed -i 's/qwen3-8B-kdrl-mask/qwen3-8B-rlsd/' /scratch/bl4363/DeepMath/train_rlsd.sh
sed -i 's/ADV_ESTIMATOR=grpo_opd_mask/ADV_ESTIMATOR=grpo_rlsd/' /scratch/bl4363/DeepMath/train_rlsd.sh
sed -i '/^OPD_BETA=0\.02$/d' /scratch/bl4363/DeepMath/train_rlsd.sh
sed -i 's|    algorithm.opd_beta=\$OPD_BETA \\|    +algorithm.rlsd_eps_w=0.2 \\\n    +algorithm.rlsd_lambda=1.0 \\|' /scratch/bl4363/DeepMath/train_rlsd.sh
```

- [ ] **Step 2: Verify the changes**

```bash
grep -E '^#SBATCH --job-name|^RUN_NAME|^ADV_ESTIMATOR|^OPD_BETA|rlsd_eps_w|rlsd_lambda|opd_beta=' /scratch/bl4363/DeepMath/train_rlsd.sh
```

Expected:
```
#SBATCH --job-name=rlsd
RUN_NAME=qwen3-1.7B-Base-qwen3-8B-rlsd
ADV_ESTIMATOR=grpo_rlsd
    +algorithm.rlsd_eps_w=0.2 \
    +algorithm.rlsd_lambda=1.0 \
```

(No `OPD_BETA=` line, no `algorithm.opd_beta=` line.)

- [ ] **Step 3: Lint**

```bash
bash -n /scratch/bl4363/DeepMath/train_rlsd.sh && echo ok
```

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
cd /scratch/bl4363/DeepMath
git add train_rlsd.sh
git commit -m "$(cat <<'EOF'
scripts: add train_rlsd.sh (grpo_rlsd / RLSD)

Drops opd_beta (RLSD does not consume it); passes rlsd_eps_w=0.2,
rlsd_lambda=1.0 instead.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: Bump the verl submodule pointer in DeepMath

**Files:**
- Modify: `verl` (gitlink) in DeepMath repo

After Tasks 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 the submodule has 11 new commits beyond the pointer DeepMath currently records. Bump the pointer.

- [ ] **Step 1: Confirm submodule HEAD is what we expect**

```bash
cd /scratch/bl4363/DeepMath/verl
git log --oneline -12
```

Expected: 11 new commits on top of `9721128 opd` (one per Task 1–11).

- [ ] **Step 2: Stage the submodule pointer in DeepMath**

```bash
cd /scratch/bl4363/DeepMath
git status --short | head
git add verl
```

Expected: `git status` shows `M verl` (gitlink update).

- [ ] **Step 3: Commit the pointer bump**

```bash
cd /scratch/bl4363/DeepMath
git commit -m "$(cat <<'EOF'
verl: bump submodule pointer for OPD-family port

Includes:
- TRRD validation lift (prerequisite, unrelated to OPD-family)
- AdvantageEstimator: GRPO_OPD_MASK / GRPO_OPD_CLIFF / GRPO_SRPO / GRPO_RLSD
- compute_advantage branches for all four
- Trainer-side correctness_scores derivation + inject/pop pattern
- rlsd_eps_w / rlsd_lambda config defaults

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Final sanity check**

```bash
cd /scratch/bl4363/DeepMath
git submodule status
ls -la train_srpo.sh train_kdrl_mask.sh train_hdpo_cliff.sh train_rlsd.sh
python3 -c "from verl.trainer.ppo.core_algos import AdvantageEstimator; print('grpo_srpo' in {m.value for m in AdvantageEstimator})"
```

Expected: submodule pointer matches the new HEAD; all four scripts exist and are executable; `True` printed.

---

## Self-review checklist (already run by the planner)

- **Spec coverage:** Each section of the spec maps to one or more tasks: enum (T2), gate (T3), signature (T4), formulas (T5–T8), correctness derivation (T9), inject/pop (T10), config (T11), scripts (T12–T15), submodule pointer (T16). The unrelated TRRD edits get committed in T1 to keep the OPD-family commits clean.
- **Placeholder scan:** No TBDs, no "implement later", no "similar to Task N". Every code block is complete.
- **Type consistency:** The `compute_advantage` signature in T4 matches the call site update in T4 and the inject/pop block in T10. The four enum members defined in T2 are used verbatim in T3, T5–T9, T10. Field name `correctness_scores` is consistent across all tasks.
- **Spec deviations recorded:**
  - T9 implements the correctness derivation in the trainer (not the reward fn) per the spec correction commit.
  - The spec describes `train_kdrl_mask.sh` etc. as "modeled on `train_kdrl.sh`"; this plan models them on each other (T12 first, then sed-edit) for less repetition. Behavior is the same.
