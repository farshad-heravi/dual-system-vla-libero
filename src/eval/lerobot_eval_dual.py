"""Dual-system eval script.

Patches lerobot's rollout() with rollout_dual() — the only modification is
S2 sub-goal injection. Everything else (env setup, preprocessors, metrics,
video saving) comes from lerobot unchanged.

S2 config via env vars (avoids extending lerobot's draccus config):

  DUAL_ABLATION   a0 | a1 | a2 | a3   (default: a1)
  DUAL_N_TRIGGER  int                  (default: 45 steps)
  DUAL_S2_MODE    append | replace     (default: append)
  DUAL_S2_MODEL   HF model id          (default: Qwen/Qwen2.5-VL-3B-Instruct)
  DUAL_LOG_DIR    path                 (default: s2_logs)

Usage:
    # Day-3 sanity check (5 episodes, A1)
    DUAL_ABLATION=a1 MUJOCO_GL=egl \\
    python -m src.eval.lerobot_eval_dual \\
        --policy.path=lerobot/pi0_libero_finetuned \\
        --env.type=libero --env.task=libero_10 \\
        --eval.n_episodes=5 --eval.batch_size=1 \\
        --output_dir=./outputs/a1_sanity

    # Day-4 full ablation run (50 episodes, A0 baseline)
    DUAL_ABLATION=a0 MUJOCO_GL=egl \\
    python -m src.eval.lerobot_eval_dual \\
        --policy.path=lerobot/pi0_libero_finetuned \\
        --env.type=libero --env.task=libero_10 \\
        --eval.n_episodes=50 --eval.batch_size=1 \\
        --output_dir=./outputs/a0_baseline
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional

import einops
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from tqdm import trange

# lerobot internals — imported before we patch
import lerobot.scripts.lerobot_eval as _lerobot_eval
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    preprocess_observation,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.utils import inside_slurm

from src.system2.async_runner import AsyncS2Runner, RandomS2Runner
from src.system2.planner import Qwen25VLPlanner

# ------------------------------------------------------------------
# S2 config from env vars
# ------------------------------------------------------------------
_ABLATION      = os.getenv("DUAL_ABLATION",    "a1")
_N_TRIGGER     = int(os.getenv("DUAL_N_TRIGGER",  "45"))
_S2_MODE       = os.getenv("DUAL_S2_MODE",    "append")
_S2_MODEL      = os.getenv("DUAL_S2_MODEL",   "Qwen/Qwen2.5-VL-3B-Instruct")
_LOG_DIR       = os.getenv("DUAL_LOG_DIR",    "s2_logs")
_N_FRAMES      = int(os.getenv("DUAL_N_FRAMES",   "1"))
# DUAL_PERTURB_STEP: step at which to teleport the first movable object 5 cm
# in a random direction. -1 disables (default). Set to 50 for the error-
# recovery ablation (PLAN_V2.md §Day 5).
_PERTURB_STEP       = int(os.getenv("DUAL_PERTURB_STEP",       "-1"))
_PERTURB_SEED       = int(os.getenv("DUAL_PERTURB_SEED",       "42"))
# Directory for per-episode JSONL logs in the format expected by failure_classifier.py.
# Set to "" to disable this logging.
_EPISODE_LOG_DIR    = os.getenv("DUAL_EPISODE_LOG_DIR", "episodes")


# ------------------------------------------------------------------
# rollout_dual: lerobot's rollout() + S2 injection
#
# Lines marked [S2] are the ONLY additions. Anything else is a verbatim
# copy of lerobot/scripts/lerobot_eval.py::rollout() so diffs are auditable.
# ------------------------------------------------------------------

def rollout_dual(
    env: gym.vector.VectorEnv,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    # [S2] injected by _make_patched_rollout; never present in original signature
    s2_runner=None,
    n_trigger: int = 45,
    s2_log: Optional[list] = None,
    # [S2] per-episode JSONL log in failure_classifier format (one entry per trigger)
    episode_log: Optional[list] = None,
    # [PERTURB] mid-episode perturbation (error-recovery ablation)
    perturb_step: int = -1,
    perturb_rng: Optional[np.random.Generator] = None,
) -> dict:
    from torch import nn
    assert isinstance(policy, nn.Module)

    policy.reset()
    observation, info = env.reset(seed=seeds)

    # bddl_base_domain overrides robosuite's time-based done with _check_success()
    # only, so robosuite's internal done flag never propagates up and the next
    # env.step() call crashes once timestep >= horizon.
    for sub_env in env.envs:
        sub_env._env.env.ignore_done = True

    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions      = []
    all_rewards      = []
    all_successes    = []
    all_dones        = []

    step     = 0
    done     = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]

    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),
        leave=False,
    )
    check_env_attributes_and_types(env)

    while not np.all(done) and step < max_steps:
        observation = preprocess_observation(observation)

        if return_observations:
            all_observations.append(deepcopy(observation))

        observation = add_envs_task(env, observation)

        # [S2] On the first step, sync runner to the actual task instruction
        # (task is not known until the env is reset and add_envs_task runs).
        if s2_runner is not None and step == 0:
            task_instr = observation.get("task", [""])[0]
            s2_runner.reset(instruction=task_instr)

            # [S2] Fire S2 immediately on the reset frame so sub-goals are
            # available by step n_trigger rather than step 2*n_trigger.
            raw_frame = env.envs[0].render()
            s2_runner.update_obs(Image.fromarray(raw_frame))

            if episode_log is not None:
                # Store task instruction in first entry so the closure can
                # resolve task_id without needing to pass it explicitly.
                episode_log.append({
                    "step": 0,
                    "current_subtask": "",
                    "progress_pct": 0,
                    "should_replan": False,
                    "success": False,
                    "_task_instr": task_instr,
                })

        # [S2] Trigger S2 every n_trigger steps after step 0.
        if s2_runner is not None and step > 0 and step % n_trigger == 0:
            raw_frame = env.envs[0].render()
            s2_runner.update_obs(Image.fromarray(raw_frame))

            if episode_log is not None:
                sg = s2_runner.get_subgoal_object()
                episode_log.append({
                    "step": step,
                    "current_subtask": sg.current_subtask if sg else "",
                    "progress_pct": sg.progress_pct if sg else 0,
                    "should_replan": sg.should_replan if sg else False,
                    "success": False,
                })

        # [PERTURB] At the configured step, teleport the first movable object.
        if perturb_step > 0 and step == perturb_step:
            from src.eval.perturbation import inject_perturbation
            _rng = perturb_rng if perturb_rng is not None else np.random.default_rng()
            perturb_meta = inject_perturbation(env, _rng)
            if s2_log is not None:
                s2_log.append({"step": step, "perturbation": perturb_meta})

        # [S2] Replace observation["task"] with the latest sub-goal.
        # Injection happens HERE — after add_envs_task but before preprocessors
        # — so the text is encoded by the policy's own language encoder, not
        # pre-tokenised with the wrong string.
        if s2_runner is not None:
            subgoal_str = s2_runner.get_subgoal()
            observation["task"] = [subgoal_str] * len(observation.get("task", [""]))

            if s2_log is not None:
                sg = s2_runner.get_subgoal_object()
                s2_log.append({
                    "step":         step,
                    "override":     subgoal_str,
                    "subtask":      sg.current_subtask if sg else None,
                    "replan":       sg.should_replan   if sg else False,
                    "parse_failed": sg.parse_failed    if sg else False,
                })

        observation = env_preprocessor(observation)
        observation = preprocessor(observation)

        with torch.inference_mode():
            action = policy.select_action(observation)

        action = postprocessor(action)
        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition[ACTION]

        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2

        observation, reward, terminated, truncated, info = env.step(action_numpy)

        if render_callback is not None:
            render_callback(env)

        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported final_info format. Requires gymnasium >= 1.0."
                )
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_sr = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any")
            .numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_sr * 100:.1f}%"})
        progbar.update()

    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    ret = {
        ACTION:     torch.stack(all_actions,   dim=1),
        "reward":   torch.stack(all_rewards,   dim=1),
        "success":  torch.stack(all_successes, dim=1),
        "done":     torch.stack(all_dones,     dim=1),
    }
    if return_observations:
        stacked = {}
        for key in all_observations[0]:
            stacked[key] = torch.stack([o[key] for o in all_observations], dim=1)
        ret[OBS_STR] = stacked

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


# ------------------------------------------------------------------
# Patch factory
#
# Python looks up module-level names at call time, so replacing
# lerobot_eval.rollout replaces the reference seen by eval_policy().
# ------------------------------------------------------------------

def _make_patched_rollout(
    s2_runner,
    n_trigger: int,
    log_dir: Path,
    perturb_step: int = -1,
    perturb_base_seed: int = 42,
    episode_log_dir: Optional[Path] = None,
    task_meta: Optional[dict] = None,
):
    ep_idx = [0]

    # Build instruction→task_id lookup once, reused across all episodes.
    instr_to_task_id: dict[str, int] = {}
    if task_meta is not None:
        for key, info in task_meta.items():
            tid = int(key.split("_")[1])
            instr_to_task_id[info["description"].lower()] = tid

    def patched(*args, **kwargs):
        s2_log: list[dict[str, Any]] = []
        episode_log: list[dict[str, Any]] = [] if episode_log_dir is not None else None

        kwargs["s2_runner"]    = s2_runner
        kwargs["n_trigger"]    = n_trigger
        kwargs["s2_log"]       = s2_log
        kwargs["episode_log"]  = episode_log
        kwargs["perturb_step"] = perturb_step
        if perturb_step > 0:
            kwargs["perturb_rng"] = np.random.default_rng(perturb_base_seed + ep_idx[0])

        result = rollout_dual(*args, **kwargs)

        # Save legacy per-episode S2 trace (ep{N}_s2_trace.jsonl).
        if s2_log:
            log_path = log_dir / f"ep{ep_idx[0]:04d}_s2_trace.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                for entry in s2_log:
                    f.write(json.dumps(entry) + "\n")

        # Save new per-episode JSONL in failure_classifier format.
        if episode_log and episode_log_dir is not None:
            _write_episode_log(
                episode_log=episode_log,
                result=result,
                seeds=kwargs.get("seeds"),
                ep_idx=ep_idx[0],
                episode_log_dir=episode_log_dir,
                instr_to_task_id=instr_to_task_id,
            )

        ep_idx[0] += 1
        return result

    return patched


def _write_episode_log(
    episode_log: list[dict],
    result: dict,
    seeds: Optional[list],
    ep_idx: int,
    episode_log_dir: Path,
    instr_to_task_id: dict[str, int],
) -> None:
    # Resolve task_id from the instruction stored in the step-0 entry.
    task_instr = (episode_log[0].get("_task_instr") or "").lower().strip()
    task_id = instr_to_task_id.get(task_instr, ep_idx)  # fall back to ep index

    # Seed: first element of the seeds list (batch_size=1 assumed).
    seed = seeds[0] if seeds else ep_idx

    # Episode outcome: succeeded if any step had a positive reward signal.
    success = bool(result.get("success", torch.zeros(1)).any().item())

    # Stamp the final success onto the last entry and strip the private key.
    cleaned: list[dict] = []
    for i, entry in enumerate(episode_log):
        e = {k: v for k, v in entry.items() if not k.startswith("_")}
        if i == len(episode_log) - 1:
            e["success"] = success
        cleaned.append(e)

    episode_log_dir.mkdir(parents=True, exist_ok=True)
    out_path = episode_log_dir / f"episode_{task_id}_{seed}.jsonl"
    with open(out_path, "w") as f:
        for entry in cleaned:
            f.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------
# Build the S2 runner for the requested ablation
# ------------------------------------------------------------------

def _build_s2_runner(ablation: str, model_id: str, mode: str):
    if ablation == "a0":
        return None  # no S2

    if ablation == "a3":
        import pandas as pd
        from huggingface_hub import hf_hub_download
        # lerobot datasets store task strings as the index of meta/tasks.parquet,
        # not as a "task" column in the main data files (which only has "task_index").
        tasks_path = hf_hub_download(
            repo_id="lerobot/libero_10",
            filename="meta/tasks.parquet",
            repo_type="dataset",
        )
        all_tasks = sorted(pd.read_parquet(tasks_path).index.tolist())
        print(f"[A3] Random pool: {len(all_tasks)} tasks")
        return RandomS2Runner(pool=all_tasks)

    if ablation == "a4":
        from src.system2.oracle import OracleS2Runner
        print("[A4] Oracle sub-goals — no inference.")
        return OracleS2Runner(max_steps=520)

    if ablation == "a5":
        from src.system2.planner_v2 import Qwen25VLPlannerV2
        from src.system2.async_runner_v2 import AsyncS2RunnerV2
        print(f"[A5] VLM-gated oracle advancement  n_frames={_N_FRAMES}")
        planner = Qwen25VLPlannerV2(model_id=model_id)
        runner = AsyncS2RunnerV2(planner, mode=mode, n_frames=_N_FRAMES)
        runner.start()
        return runner

    planner = Qwen25VLPlanner(model_id=model_id)
    frozen  = (ablation == "a2")
    runner  = AsyncS2Runner(planner, mode=mode, frozen=frozen)
    runner.start()
    return runner


# ------------------------------------------------------------------
# Entry point: extends lerobot's eval_main with S2 monkey-patch
# ------------------------------------------------------------------

@parser.wrap()
def eval_dual_main(cfg: EvalPipelineConfig):
    from contextlib import nullcontext

    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import close_envs
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from lerobot.utils.utils import get_safe_torch_device
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.random_utils import set_seed
    from lerobot.utils.utils import init_logging

    init_logging()
    register_third_party_plugins()

    perturb_tag = f"  perturb_step={_PERTURB_STEP}" if _PERTURB_STEP > 0 else ""
    print(f"[dual-eval] ablation={_ABLATION}  n_trigger={_N_TRIGGER}  "
          f"mode={_S2_MODE}  model={_S2_MODEL}{perturb_tag}")

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    set_seed(cfg.seed)

    # [NOISE-FLOOR] Decouple the policy action-sampling RNG from the env seed.
    # cfg.seed still fixes the env object configs (via start_seed -> env.reset(seed=...)),
    # so setting DUAL_NOISE_SEED varies ONLY pi0's flow-matching noise while holding
    # every episode's scene identical. Run the same condition+--seed twice with two
    # different DUAL_NOISE_SEED values to measure the same-condition flip rate.
    _noise_seed = os.getenv("DUAL_NOISE_SEED")
    if _noise_seed is not None:
        torch.manual_seed(int(_noise_seed))
        torch.cuda.manual_seed_all(int(_noise_seed))
        print(f"[dual-eval] DUAL_NOISE_SEED={_noise_seed} "
              f"(action noise reseeded; env seed unchanged at cfg.seed={cfg.seed})")

    # -- env + policy (identical to lerobot's eval_main) --
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()

    pre_overrides = {
        "device_processor":              {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=pre_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=cfg.env, policy_cfg=cfg.policy
    )

    # -- S2 runner --
    s2_runner = _build_s2_runner(_ABLATION, _S2_MODEL, _S2_MODE)

    # -- Patch lerobot's rollout before calling eval_policy_all --
    log_dir = Path(cfg.output_dir) / _LOG_DIR

    episode_log_dir: Optional[Path] = None
    task_meta: Optional[dict] = None
    if _EPISODE_LOG_DIR and s2_runner is not None:
        episode_log_dir = Path(cfg.output_dir) / _EPISODE_LOG_DIR
        _task_meta_path = Path("configs/libero10_tasks.yaml")
        if _task_meta_path.exists():
            import yaml
            with open(_task_meta_path) as _f:
                task_meta = yaml.safe_load(_f)["tasks"]

    _lerobot_eval.rollout = _make_patched_rollout(
        s2_runner,
        _N_TRIGGER,
        log_dir,
        perturb_step=_PERTURB_STEP,
        perturb_base_seed=_PERTURB_SEED,
        episode_log_dir=episode_log_dir,
        task_meta=task_meta,
    )

    # -- Run eval (all logging, video saving, metrics handled by lerobot) --
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
    ):
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
        )

    print(f"\n[{_ABLATION.upper()}] SR: {info['overall']['pc_success']:.1f}%")

    # -- Cleanup --
    close_envs(envs)
    if s2_runner is not None and hasattr(s2_runner, "stop"):
        s2_runner.stop()

    # -- Save results --
    out = Path(cfg.output_dir) / "eval_info.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Results -> {out}")


def main():
    eval_dual_main()


if __name__ == "__main__":
    main()
