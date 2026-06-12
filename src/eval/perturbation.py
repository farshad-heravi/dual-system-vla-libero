"""Mid-episode perturbation injector for the error-recovery ablation.

Protocol (PLAN_V2.md §Day 5 / §4 Error Recovery):
  - At step DUAL_PERTURB_STEP (default 50), teleport the first movable
    object 5 cm in a uniformly random horizontal direction.
  - Run A0+perturb and A1+perturb, 30 episodes each.
  - Compare recovery SRs to get Table 2 in the report.

Wrapper chain used in lerobot + LIBERO:
  vec_env.envs[0]           → LiberoEnv (lerobot gym wrapper)
  .._env                    → ControlEnv (LIBERO wrapper)
  .._env.env                → BDDLBaseDomain (robosuite env)
  .._env.env.sim            → MuJoCo simulation
  .._env.env.objects_dict   → {name: MujocoObject} (movable objects only)
  .._env.env.obj_body_id    → {name: int} MuJoCo body IDs
"""

import logging
import numpy as np

log = logging.getLogger(__name__)


def _robosuite_env(vec_env):
    return vec_env.envs[0]._env.env


def inject_perturbation(
    vec_env,
    rng: np.random.Generator,
    delta_m: float = 0.05,
) -> dict:
    """Teleport the first movable scene object by `delta_m` m in a random XY direction.

    Args:
        vec_env:  gymnasium VectorEnv (the object passed to rollout_dual)
        rng:      numpy Generator — use a fixed seed per episode for reproducibility
        delta_m:  displacement in metres (default 0.05 = 5 cm)

    Returns:
        Metadata dict logged into the S2 trace (object name, old/new XY positions).
    """
    rs_env = _robosuite_env(vec_env)

    if not rs_env.objects_dict:
        log.warning("[PERTURB] No movable objects found — skipping.")
        return {"skipped": True}

    # Always pick the first movable object (insertion-ordered in Python 3.7+,
    # so the same object is chosen consistently for a given task).
    obj_name = next(iter(rs_env.objects_dict))
    obj = rs_env.objects_dict[obj_name]
    body_id = rs_env.obj_body_id[obj_name]

    # Current world-frame position and quaternion.
    # body_xquat is stored as (w, x, y, z) — same layout expected by set_joint_qpos
    # for the rotational part of a free joint.
    old_pos = rs_env.sim.data.body_xpos[body_id].copy()
    cur_quat = rs_env.sim.data.body_xquat[body_id].copy()

    angle = rng.uniform(0.0, 2.0 * np.pi)
    new_pos = old_pos.copy()
    new_pos[0] += delta_m * np.cos(angle)
    new_pos[1] += delta_m * np.sin(angle)
    # Z is unchanged — keeps the object on the table surface.

    rs_env.sim.data.set_joint_qpos(
        obj.joints[-1],
        np.concatenate([new_pos, cur_quat]),
    )
    rs_env.sim.forward()

    log.info(
        "[PERTURB] %s  XY: (%.3f, %.3f) → (%.3f, %.3f)",
        obj_name,
        old_pos[0], old_pos[1],
        new_pos[0], new_pos[1],
    )
    return {
        "object": obj_name,
        "old_pos": old_pos[:3].tolist(),
        "new_pos": new_pos[:3].tolist(),
        "angle_rad": float(angle),
        "delta_m": delta_m,
    }
