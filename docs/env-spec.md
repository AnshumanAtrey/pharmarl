# PharmaRL Environment Specification

Formal spec of state, action, and reward. Reference for judges and re-implementers.

---

## Environment metadata

| Field | Value |
|-------|-------|
| `spec_version` | 1 |
| `name` | `pharmarl` |
| `runtime` | `fastapi` |
| `port` | `8000` |
| `target` (Stage 1) | SARS-CoV-2 Mpro (3CLPro) |

## State

Held internally per episode. Exposed via `GET /state`.

```python
MoleculeState {
    episode_id:        str          # uuid4
    step_count:        int          # 0..max_steps
    target:            str          # "SARS-CoV-2_Mpro"
    difficulty:        DifficultyTier  # trivial | easy | hard
    max_steps:         int          # 10 / 15 / 20
    smiles:            str          # current canonical SMILES
    selfies:           str          # current SELFIES (always valid)
    starting_smiles:   str          # episode's initial scaffold
    edit_history:      List[Dict]   # last 5 {action, before, after}
    cumulative_reward: float
    final_oracle_scores: Optional[Dict[str, float]]
}
```

## Action space

```python
MoleculeAction {
    action_type: Literal[
        "ADD_FRAGMENT",       # attach a fragment from the vocab
        "REMOVE_FRAGMENT",    # remove a heavy atom
        "SUBSTITUTE_ATOM",    # replace an atom with C/N/O/S/F/Cl/Br/I/P
        "TERMINATE",          # submit the current molecule for terminal reward
    ]
    fragment: Optional[str]   # SMILES, must be in vocab — required for ADD_FRAGMENT
    position: Optional[int]   # 0-indexed atom — required for REMOVE / SUBSTITUTE / ADD
    new_atom: Optional[str]   # element symbol — required for SUBSTITUTE_ATOM
}
```

## Observation

Returned after every `step()` call.

```python
MoleculeObservation {
    smiles:               str
    selfies:              str
    target:               str
    difficulty:           DifficultyTier
    properties:           Dict[str, float]  # mw, logp, hbd, hba, lipinski_violations
    valid_actions:        List[str]         # subset of action_type allowed now
    available_fragments:  List[str]         # vocab for current difficulty
    steps_remaining:      int
    last_action_valid:    bool              # False if env rejected the edit
    message:              str               # human-readable status
    truncated:            bool              # True if step limit reached, not TERMINATE
    reward:               float
    done:                 bool
}
```

## Reward function

### Per-step (dense)

| Condition | Reward |
|-----------|--------|
| Action JSON unparseable / missing `action_type` | -0.5 |
| Action attempted but rejected (invalid position, etc.) | -0.1 |
| Action valid AND post-edit molecule passes Lipinski | +0.05 |
| Action valid AND fails Lipinski | 0.0 |

### Terminal (sparse, on TERMINATE or truncation)

```
composite = w_dock * docking + w_qed * qed + w_sa * sa + w_tox * (1 - toxicity)

with weights:
  w_dock = 0.40
  w_qed  = 0.25
  w_sa   = 0.15
  w_tox  = 0.20

If final molecule fails Lipinski:
  composite *= 0.5

terminal_reward = composite * 10
```

Curriculum modulates which components are active:

| Tier | Active components |
|------|-------------------|
| Trivial | qed only |
| Easy | qed, docking |
| Hard | qed, docking, sa, toxicity |

### Total episode reward

`cumulative = Σ(step_rewards) + terminal_reward`

## Curriculum (RLVE)

Maps `training_step` → tier:

| Steps | Tier | Vocab | Max steps | Reward components |
|-------|------|-------|-----------|---------------------|
| 0-100 | Trivial | 5 fragments | 10 | qed |
| 100-300 | Easy | 15 fragments | 15 | qed + docking |
| 300+ | Hard | 50 fragments | 20 | qed + docking + sa + toxicity |

In eval mode (`training_step=None`), the env samples uniformly across tiers.

## Validity guarantees

1. SELFIES round-trip: every state's `selfies` decodes to the canonical `smiles`.
2. Mutations are sanitized via RDKit; failures yield `last_action_valid=False`.
3. Disconnected molecules (after REMOVE) are rejected.
4. Atom substitution is restricted to {C, N, O, S, F, Cl, Br, I, P}.

## Procedural episode generation (anti-staleness)

Each episode samples a starting molecule from a 200-molecule pool partitioned by tier. Combined with stochastic GRPO sampling (G=8, temperature>0), the agent never sees the same trajectory twice.

## HTTP contract

| Endpoint | Method | Body / Query | Returns |
|----------|--------|--------------|---------|
| `/reset` | POST | `{difficulty?, training_step?, seed?}` | observation, episode_id |
| `/step` | POST | `MoleculeAction` JSON | observation, reward, done |
| `/state` | GET | — | full `MoleculeState` |
| `/health` | GET | — | `{status: "ok"}` |
