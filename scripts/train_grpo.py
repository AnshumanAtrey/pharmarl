"""Standalone GRPO trainer — Colab-equivalent, runs on any GPU.

Mirrors `colab/train_pharmarl.ipynb`: SFT format priming → multi-turn GRPO with
adaptive curriculum + per-component reward logging. Identical algorithm; the
only difference is no Jupyter — runs as `python -m scripts.train_grpo ...`.

Use this when:
  - Colab dies / disconnects mid-training (Path C fallback)
  - You want to run inside an HF Space with upgraded hardware
  - You want to use Modal / RunPod / Lambda Labs for training
  - You need to reproduce a run programmatically

Usage:
  # Local (your env on localhost:8000):
  python -m scripts.train_grpo \
      --env-url http://localhost:8000 \
      --max-steps 200 \
      --output-dir ./trained \
      --hf-repo YOUR-USER/pharmarl-llama-trained

  # HF Space GPU fallback (env on a separate Space):
  python -m scripts.train_grpo \
      --env-url https://YOUR-USER-pharmarl.hf.space \
      --model unsloth/Llama-3.2-1B-Instruct \
      --max-steps 200 \
      --num-generations 8 \
      --hf-repo YOUR-USER/pharmarl-llama-trained \
      --hf-token "$HF_TOKEN"

Outputs:
  - ./<output-dir>/lora_step_N/  (every SAVE_EVERY steps)
  - ./<output-dir>/lora_final/   (after run completes)
  - W&B run (if WANDB_API_KEY env var set)
  - HF Hub push (if --hf-repo and --hf-token provided)
"""

from __future__ import annotations

import argparse
import json
import os
import random as _rnd
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import requests


# ─── Utilities (duplicated from notebook for self-containment) ────────────

SYSTEM = (
    "You design drug-like molecules against the active binding target by editing SMILES. "
    "Respond with ONE JSON action per turn. Allowed: ADD_FRAGMENT, "
    "REMOVE_FRAGMENT, SUBSTITUTE_ATOM, TERMINATE."
)

_JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_action(text: str):
    for m in _JSON_RE.findall(text):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue
    return None


def smoke_env(env_url: str) -> None:
    r = requests.post(f"{env_url}/reset", json={"difficulty": "trivial"})
    r.raise_for_status()
    obs = r.json()["observation"]
    print(f"  env OK — reset returned SMILES={obs['smiles']!r}, "
          f"vocab_size={len(obs['available_fragments'])}")


# ─── SFT format priming ──────────────────────────────────────────────────

def synthesize_sft_pairs(env_url: str, n_pairs: int = 240, difficulty: str = "trivial"):
    pairs = []
    rng = _rnd.Random(0)
    for _ in range(n_pairs):
        r = requests.post(f"{env_url}/reset", json={"difficulty": difficulty}).json()
        obs = r["observation"]
        kind = rng.choice(["ADD_FRAGMENT", "ADD_FRAGMENT", "ADD_FRAGMENT",
                           "SUBSTITUTE_ATOM", "TERMINATE"])
        if kind == "ADD_FRAGMENT":
            action = {"action_type": "ADD_FRAGMENT",
                      "fragment": rng.choice(obs["available_fragments"]),
                      "position": 0}
        elif kind == "SUBSTITUTE_ATOM":
            action = {"action_type": "SUBSTITUTE_ATOM", "position": 0,
                      "new_atom": rng.choice(["F", "N", "O", "Cl"])}
        else:
            action = {"action_type": "TERMINATE"}
        prompt = (f"{SYSTEM}\n\nSMILES: {obs['smiles']}\n"
                  f"Fragments: {obs['available_fragments'][:8]}\n"
                  f"Valid actions: {obs['valid_actions']}\n"
                  f"Respond with JSON action:")
        pairs.append((prompt, json.dumps(action)))
    return pairs


def run_sft_warmup(model, tokenizer, env_url: str, n_steps: int):
    """Format-priming SFT — teach JSON action shape before GRPO."""
    import torch
    from torch.utils.data import Dataset

    pairs = synthesize_sft_pairs(env_url, n_pairs=max(200, n_steps * 4))
    print(f"[sft] generated {len(pairs)} pairs; example target: {pairs[0][1]}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class _SFTSet(Dataset):
        def __init__(self, pairs, tok, max_len=512):
            self.pairs, self.tok, self.max_len = pairs, tok, max_len
        def __len__(self): return len(self.pairs)
        def __getitem__(self, i):
            p, t = self.pairs[i]
            text = p + " " + t + self.tok.eos_token
            enc = self.tok(text, truncation=True, max_length=self.max_len,
                           return_tensors="pt", padding="max_length")
            ids = enc["input_ids"][0]
            labels = ids.clone()
            plen = self.tok(p + " ", truncation=True, max_length=self.max_len,
                            return_tensors="pt")["input_ids"].shape[1]
            labels[:plen] = -100
            labels[ids == self.tok.pad_token_id] = -100
            return {"input_ids": ids, "labels": labels,
                    "attention_mask": enc["attention_mask"][0]}

    from unsloth import FastLanguageModel
    FastLanguageModel.for_training(model)
    ds = _SFTSet(pairs, tokenizer, max_len=512)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4
    )
    device = next(model.parameters()).device
    step = 0
    for batch in loader:
        if step >= n_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optim.step()
        optim.zero_grad()
        if step % 10 == 0:
            print(f"[sft] step={step:3d}  loss={out.loss.item():.4f}")
        step += 1

    # Format check
    FastLanguageModel.for_inference(model)
    obs = requests.post(f"{env_url}/reset", json={"difficulty": "trivial"}).json()["observation"]
    prompt = (f"{SYSTEM}\n\nSMILES: {obs['smiles']}\n"
              f"Fragments: {obs['available_fragments'][:8]}\n"
              f"Valid actions: {obs['valid_actions']}\n"
              f"Respond with JSON action:")
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inp, max_new_tokens=80, do_sample=True, temperature=0.7,
                         pad_token_id=tokenizer.eos_token_id)
    txt = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    parsed = parse_action(txt)
    print(f"[sft] post-warmup parse check: parses_to_json={parsed is not None}")
    if parsed:
        print(f"[sft]   action: {parsed}")
    return parsed is not None


# ─── GRPO training loop ──────────────────────────────────────────────────

def run_grpo(model, tokenizer, env_url: str, *,
             max_steps: int, num_generations: int, lr: float,
             kl_coef: float, clip_eps: float, max_new_tokens: int,
             max_episode_steps: int, gen_temp: float, gen_top_p: float,
             save_every: int, audit_every: int, output_dir: Path,
             use_adaptive: bool):
    import torch
    import torch.nn.functional as F
    from unsloth import FastLanguageModel

    device = next(model.parameters()).device
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    has_disable_adapter = hasattr(model, "disable_adapter")

    adaptive = None
    if use_adaptive:
        try:
            from server.curriculum import AdaptiveCurriculum, DEFAULT_CONFIG
            adaptive = AdaptiveCurriculum(config=DEFAULT_CONFIG)
            print("[grpo] adaptive RLVE curriculum enabled")
        except ImportError:
            print("[grpo] server.curriculum not on path — using step-threshold schedule")

    def _action_logprobs(prompt_ids, action_ids):
        full = torch.cat([prompt_ids, action_ids], dim=0).unsqueeze(0).to(device)
        plen, alen = prompt_ids.shape[0], action_ids.shape[0]
        logits = model(full).logits[0, plen - 1: plen + alen - 1, :]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return log_probs.gather(-1, action_ids.to(device).unsqueeze(-1)).squeeze(-1)

    def _ref_logprobs(prompt_ids, action_ids, old_lp):
        if has_disable_adapter:
            try:
                with torch.no_grad(), model.disable_adapter():
                    return _action_logprobs(prompt_ids, action_ids).detach()
            except Exception as e:
                print(f"[warn] disable_adapter failed ({e}); using old_lp as ref")
        return old_lp.to(device)

    @torch.no_grad()
    def rollout(difficulty: str):
        obs = requests.post(f"{env_url}/reset",
                            json={"difficulty": difficulty}).json()["observation"]
        transitions, cum = [], 0.0
        parse_ok = parse_total = 0
        invalid = 0
        final_components = {}
        final_smiles = obs.get("smiles", "")
        # Per-rollout observability — addresses the hackathon FAQ §17 monitoring list:
        # truncation rate, Lipinski pass rate, episode length, action-type histogram.
        action_type_counts: dict = {}
        episode_truncated = False
        final_lipinski_passes = True
        starting_smiles = obs.get("smiles", "")
        for _ in range(max_episode_steps):
            prompt = (f"{SYSTEM}\n\nSMILES: {obs['smiles']}\n"
                      f"Fragments: {obs['available_fragments'][:8]}\n"
                      f"Valid actions: {obs['valid_actions']}\n"
                      f"Respond with JSON action:")
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            plen = prompt_ids.shape[1]
            gen = model.generate(
                prompt_ids, max_new_tokens=max_new_tokens,
                temperature=gen_temp, top_p=gen_top_p, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            action_ids = gen[0, plen:]
            if action_ids.shape[0] == 0:
                break
            prompt_cpu = prompt_ids[0].detach().cpu()
            action_cpu = action_ids.detach().cpu()
            old_lp = _action_logprobs(prompt_cpu, action_cpu).detach().cpu()

            txt = tokenizer.decode(action_ids, skip_special_tokens=True)
            parse_total += 1
            parsed = parse_action(txt)
            if parsed is not None:
                parse_ok += 1
                action = parsed
            else:
                action = {"action_type": "ADD_FRAGMENT", "fragment": "C", "position": 0}
            # Track action-type histogram (FAQ §17: format adherence + diversity of strategies)
            at = str(action.get("action_type", "UNKNOWN"))
            action_type_counts[at] = action_type_counts.get(at, 0) + 1
            step = requests.post(f"{env_url}/step", json=action).json()
            cum += step["reward"]
            step_obs = step["observation"]
            if not step_obs.get("last_action_valid", True):
                invalid += 1
            meta = step.get("metadata") or step_obs.get("metadata") or {}
            if step["done"]:
                final_components = meta.get("final_oracle_scores") or {}
                final_smiles = step_obs.get("smiles", final_smiles)
                # FAQ §17 — track truncation rate (auto-truncated vs proper TERMINATE)
                episode_truncated = bool(step_obs.get("truncated", False))
                # Lipinski compliance on final molecule — directly observable from properties
                lipinski_violations = step_obs.get("properties", {}).get("lipinski_violations", 0)
                final_lipinski_passes = (lipinski_violations == 0)

            transitions.append({
                "prompt_ids": prompt_cpu,
                "action_ids": action_cpu,
                "old_log_probs": old_lp,
            })
            obs = step_obs
            if step["done"]:
                break
        return {
            "transitions": transitions,
            "cumulative": cum,
            "final_smiles": final_smiles,
            "starting_smiles": starting_smiles,
            "final_components": final_components,
            "parse_rate": parse_ok / max(parse_total, 1),
            "invalid_action_rate": invalid / max(parse_total, 1),
            # Hackathon FAQ §17 monitoring — extras for richer W&B dashboards
            "episode_length": parse_total,        # number of policy decisions
            "truncated": episode_truncated,        # hit step cap (proxy for "agent never terminates")
            "lipinski_passes": final_lipinski_passes,
            "action_type_counts": action_type_counts,
        }

    def loss_for(t, advantage):
        old_lp = t["old_log_probs"].to(device)
        new_lp = _action_logprobs(t["prompt_ids"], t["action_ids"])
        ref_lp = _ref_logprobs(t["prompt_ids"], t["action_ids"], old_lp)
        ratio = torch.exp(new_lp - old_lp)
        adv = torch.tensor(advantage, device=device, dtype=ratio.dtype)
        surr = torch.min(
            ratio * adv,
            torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv,
        )
        policy = -surr.mean()
        diff = ref_lp - new_lp
        kl = (torch.exp(diff) - diff - 1).mean()
        return policy + kl_coef * kl, policy.detach(), kl.detach()

    # W&B (optional)
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "pharmarl"),
            name=os.environ.get("WANDB_RUN_NAME", "fallback-grpo"),
            reinit=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for step in range(max_steps):
        if adaptive is not None:
            difficulty = adaptive.next_tier()
        else:
            difficulty = ("trivial" if step < 100
                          else "easy" if step < 300
                          else "hard")

        FastLanguageModel.for_inference(model)
        rollouts = [rollout(difficulty) for _ in range(num_generations)]
        cum = [r["cumulative"] for r in rollouts]
        mean_r = statistics.mean(cum)
        std_r = max(statistics.pstdev(cum), 1e-6)
        advs = [(r - mean_r) / std_r for r in cum]
        n_trans = sum(len(r["transitions"]) for r in rollouts)

        comp_keys = ("qed", "docking", "sa", "toxicity_clean")
        comp_means = {}
        for k in comp_keys:
            vals = [r["final_components"].get(k, 0.0) for r in rollouts if r["final_components"]]
            comp_means[k] = statistics.mean(vals) if vals else 0.0

        parse_rate = statistics.mean(r["parse_rate"] for r in rollouts)
        invalid_rate = statistics.mean(r["invalid_action_rate"] for r in rollouts)

        curr_diag = {}
        if adaptive is not None:
            curr_diag = adaptive.record(mean_r)

        if std_r < 1e-5 or n_trans == 0:
            elapsed = time.time() - t0
            print(f"[grpo] step={step:3d} {difficulty:8s} mean={mean_r:+.3f}  "
                  f"(skipped: no signal)  [{elapsed:.0f}s elapsed]")
            if use_wandb:
                wandb.log({"step": step, "difficulty": difficulty,
                           "mean_reward": mean_r, "skipped": 1,
                           "parse_rate": parse_rate, "invalid_action_rate": invalid_rate})
            continue

        FastLanguageModel.for_training(model)
        optim.zero_grad()
        pol_acc = kl_acc = loss_acc = 0.0
        for r, adv in zip(rollouts, advs):
            for t in r["transitions"]:
                loss, pol, kl = loss_for(t, adv)
                (loss / n_trans).backward()
                pol_acc += pol.item(); kl_acc += kl.item(); loss_acc += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optim.step()

        if use_wandb:
            # Hackathon FAQ §17 monitoring set — overall reward, per-component, format
            # adherence, truncation, Lipinski, episode length, diversity, and a periodic
            # qualitative sample. Catches reward hacking before the run finishes.
            truncation_rate = sum(1 for r in rollouts if r.get("truncated")) / len(rollouts)
            lipinski_pass_rate = sum(1 for r in rollouts if r.get("lipinski_passes")) / len(rollouts)
            episode_lengths = [r.get("episode_length", 0) for r in rollouts]
            mean_episode_length = statistics.mean(episode_lengths) if episode_lengths else 0
            unique_finals = len(set(r["final_smiles"] for r in rollouts if r.get("final_smiles")))
            diversity_pct = unique_finals / len(rollouts) if rollouts else 0
            # Action-type histogram aggregated across rollouts
            agg_action_types: dict = {}
            for r in rollouts:
                for at, c in (r.get("action_type_counts") or {}).items():
                    agg_action_types[at] = agg_action_types.get(at, 0) + c

            log_payload = {
                "step": step, "difficulty": difficulty,
                "mean_reward": mean_r, "max_reward": max(cum),
                "min_reward": min(cum), "reward_std": std_r,
                "policy_loss": pol_acc / n_trans, "kl": kl_acc / n_trans,
                "loss": loss_acc / n_trans, "grad_norm": float(grad_norm),
                "n_transitions": n_trans,
                "parse_rate": parse_rate, "invalid_action_rate": invalid_rate,
                # FAQ §17 verifier metrics
                "verifier/truncation_rate": truncation_rate,
                "verifier/lipinski_pass_rate": lipinski_pass_rate,
                "verifier/parse_rate": parse_rate,
                "verifier/invalid_action_rate": invalid_rate,
                # Episode-shape metrics
                "episode/mean_length": mean_episode_length,
                "episode/diversity_unique_pct": diversity_pct,
                **{f"action_type/{k}": v for k, v in agg_action_types.items()},
                **{f"reward_component/{k}": v for k, v in comp_means.items()},
            }
            if curr_diag:
                log_payload["curriculum/rolling_mean"] = curr_diag.get("rolling_mean", 0.0)
            wandb.log(log_payload)

            # Qualitative sample table — FAQ §15 explicitly says: "Inspect actual
            # generations during training. A rising reward is not enough if the model
            # is learning to exploit bugs." Log every audit_every steps.
            if step % audit_every == 0:
                try:
                    table = wandb.Table(
                        columns=["step", "difficulty", "starting_smiles", "final_smiles",
                                 "cumulative_reward", "episode_length", "lipinski_passes",
                                 "truncated"],
                        data=[
                            [step, difficulty, r.get("starting_smiles", ""),
                             r["final_smiles"], r["cumulative"], r.get("episode_length", 0),
                             r.get("lipinski_passes", False), r.get("truncated", False)]
                            for r in rollouts[:8]
                        ],
                    )
                    wandb.log({"samples/episode_outcomes": table, "step": step})
                except Exception as _:  # noqa: BLE001 — best-effort logging
                    pass

        if step % audit_every == 0:
            best = max(rollouts, key=lambda r: r["cumulative"])
            print(f"  AUDIT [step={step}]: best R={best['cumulative']:+.3f}  "
                  f"final={best['final_smiles']}  components={best['final_components']}")

        elapsed = time.time() - t0
        rate = (step + 1) / elapsed if elapsed > 0 else 0
        eta = (max_steps - step - 1) / rate if rate > 0 else float('inf')
        print(f"[grpo] step={step:3d} {difficulty:8s} "
              f"mean={mean_r:+.3f} max={max(cum):+.3f} "
              f"pol={pol_acc/n_trans:+.4f} kl={kl_acc/n_trans:.4f} "
              f"qed={comp_means['qed']:.2f} dock={comp_means['docking']:.2f} "
              f"parse={parse_rate:.0%} ETA={eta/60:.0f}min")

        if (step + 1) % save_every == 0:
            ckpt = output_dir / f"lora_step_{step+1}"
            model.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            print(f"[ckpt] saved → {ckpt}")

    # Final save
    final_ckpt = output_dir / "lora_final"
    model.save_pretrained(str(final_ckpt))
    tokenizer.save_pretrained(str(final_ckpt))
    print(f"[grpo] training complete; final adapter at {final_ckpt}")


# ─── Entry point ─────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="PharmaRL standalone GRPO trainer")
    p.add_argument("--env-url", required=True,
                   help="URL of deployed PharmaRL env (e.g. https://YOUR-USER-pharmarl.hf.space)")
    p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct",
                   help="HF model id (Unsloth-supported)")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--num-generations", type=int, default=8,
                   help="G in GRPO (group size). Drop to 4 if OOM.")
    p.add_argument("--max-steps", type=int, default=200,
                   help="GRPO training steps (200 ≈ 3-6h on T4)")
    p.add_argument("--sft-warmup-steps", type=int, default=60,
                   help="0 to skip SFT priming")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--kl-coef", type=float, default=0.04)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--max-episode-steps", type=int, default=20)
    p.add_argument("--gen-temp", type=float, default=0.7)
    p.add_argument("--gen-top-p", type=float, default=0.95)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--audit-every", type=int, default=25)
    p.add_argument("--output-dir", default="./trained")
    p.add_argument("--no-adaptive-curriculum", action="store_true",
                   help="Use step-threshold instead of adaptive RLVE")
    p.add_argument("--hf-repo", default=None,
                   help="HF Hub repo id to push final adapter (e.g. user/pharmarl-trained)")
    p.add_argument("--hf-token", default=None,
                   help="HF token for push (defaults to HF_TOKEN env var)")
    args = p.parse_args(argv)

    print(f"[main] env={args.env_url}  model={args.model}")
    print(f"[main] config: G={args.num_generations} max_steps={args.max_steps} "
          f"sft={args.sft_warmup_steps} lr={args.lr}")

    smoke_env(args.env_url)

    print("[main] loading model with Unsloth...")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        fast_inference=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if args.sft_warmup_steps > 0:
        ok = run_sft_warmup(model, tokenizer, args.env_url, args.sft_warmup_steps)
        if not ok:
            print("[warn] SFT warmup ended without parseable output. "
                  "Consider --sft-warmup-steps 120.")

    output_dir = Path(args.output_dir)
    run_grpo(
        model, tokenizer, args.env_url,
        max_steps=args.max_steps, num_generations=args.num_generations,
        lr=args.lr, kl_coef=args.kl_coef, clip_eps=args.clip_eps,
        max_new_tokens=args.max_new_tokens, max_episode_steps=args.max_episode_steps,
        gen_temp=args.gen_temp, gen_top_p=args.gen_top_p,
        save_every=args.save_every, audit_every=args.audit_every,
        output_dir=output_dir,
        use_adaptive=not args.no_adaptive_curriculum,
    )

    if args.hf_repo:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            print("[warn] --hf-repo set but no HF token (--hf-token or $HF_TOKEN). Skipping push.")
        else:
            print(f"[push] uploading final adapter → {args.hf_repo}")
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.create_repo(repo_id=args.hf_repo, exist_ok=True, private=False)
            api.upload_folder(folder_path=str(output_dir / "lora_final"),
                              repo_id=args.hf_repo)
            print(f"[push] done — https://huggingface.co/{args.hf_repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
