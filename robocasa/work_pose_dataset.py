"""Official-demo work-pose extraction and reproducible perturbation helpers."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


STATE_BASE_POSITION = slice(0, 3)
STATE_BASE_QUATERNION = slice(3, 7)
RAW_ACTION_BASE = slice(7, 10)
RAW_ACTION_MODE = 11
ANNOTATION_COLUMNS = (
    "annotation.human.subtask",
    "annotation.human.subtask_name",
    "annotation.human.subtask_stage",
    "subtask_idx",
)
DEFAULT_PERTURBATION_RANGES = {
    "mild": (0.10, 0.20, 5.0, 10.0),
    "moderate": (0.20, 0.40, 10.0, 25.0),
    "severe": (0.40, 0.60, 25.0, 40.0),
}


@dataclass(frozen=True)
class AnnotationSegment:
    subtask_idx: int
    start_frame: int
    end_frame: int
    subtask: str
    source_atomic_task: str
    stage: str

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def resolve_lerobot_root(path: str | Path) -> Path:
    """Accept a dataset directory or its nested lerobot directory."""

    root = Path(path).expanduser().resolve()
    if (root / "meta" / "info.json").is_file() and (root / "extras").is_dir():
        return root
    nested = root / "lerobot"
    if (nested / "meta" / "info.json").is_file() and (nested / "extras").is_dir():
        return nested
    raise FileNotFoundError(
        f"{root} is not a RoboCasa LeRobot dataset with meta and extras"
    )


def load_task_labels(dataset: str | Path) -> dict[int, str]:
    root = resolve_lerobot_root(dataset)
    labels: dict[int, str] = {}
    with (root / "meta" / "tasks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                labels[int(value["task_index"])] = str(value["task"])
    return labels


def episode_parquet_path(dataset: str | Path, episode_index: int) -> Path:
    root = resolve_lerobot_root(dataset)
    matches = sorted(
        (root / "data").glob(f"chunk-*/episode_{int(episode_index):06d}.parquet")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one parquet for episode {episode_index}, found {matches}"
        )
    return matches[0]


def load_episode_dataframe(dataset: str | Path, episode_index: int):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("work-pose extraction requires pandas and pyarrow") from exc
    return pd.read_parquet(episode_parquet_path(dataset, episode_index))


def annotation_segments(
    dataframe: Any,
    task_labels: Mapping[int, str],
) -> list[AnnotationSegment]:
    missing = [name for name in ANNOTATION_COLUMNS if name not in dataframe.columns]
    if missing:
        raise ValueError(f"dataset is missing subtask annotations: {missing}")
    keys = [
        tuple(int(value) for value in row)
        for row in dataframe.loc[:, ANNOTATION_COLUMNS].itertuples(
            index=False, name=None
        )
    ]
    if not keys:
        return []
    result: list[AnnotationSegment] = []
    start = 0
    current = keys[0]
    for frame in range(1, len(keys) + 1):
        if frame < len(keys) and keys[frame] == current:
            continue
        subtask_label, atomic_label, stage_label, subtask_idx = current
        result.append(
            AnnotationSegment(
                subtask_idx=subtask_idx,
                start_frame=start,
                end_frame=frame - 1,
                subtask=str(task_labels.get(subtask_label, subtask_label)),
                source_atomic_task=str(task_labels.get(atomic_label, atomic_label)),
                stage=str(task_labels.get(stage_label, stage_label)).lower(),
            )
        )
        if frame < len(keys):
            start, current = frame, keys[frame]
    return result


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def quaternion_yaw_xyzw(quaternion: Sequence[float]) -> float:
    values = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        raise ValueError("base quaternion must contain four finite XYZW values")
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        raise ValueError("base quaternion must be non-zero")
    x, y, z, w = values / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_quaternion_xyzw(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def base_pose_from_state(observation_state: Sequence[float]) -> dict[str, Any]:
    state = np.asarray(observation_state, dtype=np.float64).reshape(-1)
    if state.size < 7:
        raise ValueError(f"observation.state requires 7 values, got {state.size}")
    quaternion = state[STATE_BASE_QUATERNION]
    return {
        "position": state[STATE_BASE_POSITION].tolist(),
        "quaternion_xyzw": quaternion.tolist(),
        "yaw_rad": quaternion_yaw_xyzw(quaternion),
    }


def apply_local_pose_delta(
    expert_pose: Mapping[str, Any],
    *,
    forward_m: float,
    left_m: float,
    yaw_rad: float,
) -> dict[str, Any]:
    position = np.asarray(expert_pose["position"], dtype=np.float64).copy()
    expert_yaw = float(
        expert_pose.get(
            "yaw_rad", quaternion_yaw_xyzw(expert_pose["quaternion_xyzw"])
        )
    )
    cosine, sine = math.cos(expert_yaw), math.sin(expert_yaw)
    position[:2] += (
        cosine * forward_m - sine * left_m,
        sine * forward_m + cosine * left_m,
    )
    target_yaw = wrap_angle(expert_yaw + yaw_rad)
    return {
        "position": position.tolist(),
        "quaternion_xyzw": yaw_quaternion_xyzw(target_yaw),
        "yaw_rad": target_yaw,
    }


def pose_error(
    pose: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, float]:
    position = np.asarray(pose["position"], dtype=np.float64)
    target = np.asarray(reference["position"], dtype=np.float64)
    yaw = float(pose.get("yaw_rad", quaternion_yaw_xyzw(pose["quaternion_xyzw"])))
    target_yaw = float(
        reference.get(
            "yaw_rad", quaternion_yaw_xyzw(reference["quaternion_xyzw"])
        )
    )
    yaw_error = abs(wrap_angle(yaw - target_yaw))
    return {
        "translation_m": float(np.linalg.norm(position[:2] - target[:2])),
        "yaw_rad": yaw_error,
        "yaw_deg": math.degrees(yaw_error),
    }


def prefix_pose_stability(
    observation_states: np.ndarray,
    segment: AnnotationSegment,
    *,
    window: int = 10,
    translation_tolerance_m: float = 0.03,
    yaw_tolerance_deg: float = 5.0,
) -> dict[str, Any]:
    stop = min(segment.end_frame + 1, segment.start_frame + window)
    values = np.asarray(observation_states[segment.start_frame:stop], dtype=float)
    positions = values[:, STATE_BASE_POSITION]
    yaws = np.array(
        [quaternion_yaw_xyzw(item[STATE_BASE_QUATERNION]) for item in values]
    )
    position_reference = np.median(positions[:, :2], axis=0)
    yaw_reference = float(np.median(yaws))
    translation_span = float(
        np.max(np.linalg.norm(positions[:, :2] - position_reference, axis=1))
    )
    yaw_span = float(
        np.max(np.abs([wrap_angle(value - yaw_reference) for value in yaws]))
    )
    return {
        "window_start": segment.start_frame,
        "window_end": stop - 1,
        "translation_span_m": translation_span,
        "yaw_span_deg": math.degrees(yaw_span),
        "stable": bool(
            translation_span <= translation_tolerance_m
            and yaw_span <= math.radians(yaw_tolerance_deg)
        ),
    }


def _vegetable_name(episode_meta: Mapping[str, Any]) -> str:
    for config in episode_meta.get("object_cfgs", []):
        if isinstance(config, Mapping) and config.get("name") == "vegetable":
            info = config.get("info", {})
            if isinstance(info, Mapping) and info.get("cat"):
                return str(info["cat"]).replace("_", " ")
    return "vegetable"


def steam_operation_spec(
    segment: AnnotationSegment,
    episode_meta: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Convert an official pick/place annotation into one policy-only skill."""

    if segment.stage not in {"pick", "place"}:
        return None
    text = segment.subtask.lower()
    vegetable_name = _vegetable_name(episode_meta)
    if segment.stage == "pick":
        object_id = "bowl" if "bowl" in text else "vegetable"
        object_name = "bowl" if object_id == "bowl" else vegetable_name
        source_id = "counter" if object_id == "bowl" else "sink"
        prompt = segment.subtask.rstrip(".") + " and keep holding it."
        call = {
            "subgoal_id": f"dataset_{segment.subtask_idx}_pick_{object_id}",
            "atomic_task": "PickObject",
            "arguments": {
                "object_id": object_id,
                "object_name": object_name,
                "source_id": source_id,
                "source_name": source_id,
            },
            "termination_condition": {
                "predicate": "holding",
                "subject": object_id,
                "desired_value": True,
                "threshold": 0.05,
            },
            "policy_prompt": prompt,
            "metadata": {"source_atomic_task": segment.source_atomic_task},
        }
        return {
            "operation": "pick",
            "object_id": object_id,
            "object_name": object_name,
            "source_id": source_id,
            "target_id": object_id,
            "policy_prompt": prompt,
            "atomic_task_call": call,
        }

    microwave_place = "microwave" in text
    object_id = "bowl" if microwave_place else "vegetable"
    object_name = "bowl" if microwave_place else vegetable_name
    destination_id = "microwave" if microwave_place else "bowl"
    destination_kind = "fixture" if microwave_place else "object"
    prompt = segment.subtask.rstrip(".") + ", release it, and move the gripper away."
    conditions: list[dict[str, Any]] = [
        {
            "predicate": "inside",
            "subject": object_id,
            "object": destination_id,
            "desired_value": True,
        },
        {"predicate": "released", "subject": object_id, "desired_value": True},
        {
            "predicate": "gripper_far",
            "subject": object_id,
            "threshold": 0.25,
            "desired_value": True,
        },
    ]
    if destination_kind == "fixture":
        conditions.append(
            {
                "predicate": "eef_outside_fixture",
                "subject": destination_id,
                "margin": 0.02,
                "desired_value": True,
            }
        )
    call = {
        "subgoal_id": f"dataset_{segment.subtask_idx}_place_{object_id}",
        "atomic_task": "PlaceObject",
        "arguments": {
            "held_object_id": object_id,
            "object_id": object_id,
            "object_name": object_name,
            "destination_id": destination_id,
            "destination_name": destination_id,
            "destination_preposition": "inside",
        },
        "termination_condition": conditions,
        "policy_prompt": prompt,
        "metadata": {"source_atomic_task": segment.source_atomic_task},
    }
    return {
        "operation": "place",
        "object_id": object_id,
        "object_name": object_name,
        "destination_id": destination_id,
        "destination_kind": destination_kind,
        "target_id": destination_id,
        "policy_prompt": prompt,
        "atomic_task_call": call,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_expert_stages(
    *,
    dataset: str | Path,
    output_dir: str | Path,
    episode_indices: Iterable[int],
    stability_window: int = 10,
) -> dict[str, Any]:
    """Extract the four annotated Pick/Place entry states from each episode."""

    root = resolve_lerobot_root(dataset)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels = load_task_labels(root)
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for episode_index in map(int, episode_indices):
        source_extra = root / "extras" / f"episode_{episode_index:06d}"
        if not source_extra.is_dir():
            rejected.append(
                {"episode_index": episode_index, "reason": "missing_episode_extras"}
            )
            continue
        dataframe = load_episode_dataframe(root, episode_index)
        states = np.load(source_extra / "states.npz")["states"]
        observations = np.stack(dataframe["observation.state"].to_numpy())
        if len(dataframe) != len(states):
            rejected.append(
                {"episode_index": episode_index, "reason": "state_length_mismatch"}
            )
            continue
        ep_meta = json.loads((source_extra / "ep_meta.json").read_text())
        episode_output = output / f"episode_{episode_index:06d}"
        episode_output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_extra / "ep_meta.json", episode_output / "ep_meta.json")
        shutil.copy2(source_extra / "model.xml.gz", episode_output / "model.xml.gz")
        model_hash = sha256_file(episode_output / "model.xml.gz")
        for segment in annotation_segments(dataframe, labels):
            operation = steam_operation_spec(segment, ep_meta)
            if operation is None:
                continue
            frame = segment.start_frame
            stage_name = (
                f"stage_{segment.subtask_idx:02d}_{operation['operation']}_"
                f"{operation['object_id']}"
            )
            stage_dir = episode_output / stage_name
            stage_dir.mkdir(parents=True, exist_ok=True)
            state_path = stage_dir / "state.npz"
            np.savez_compressed(state_path, state=states[frame])
            digest = hashlib.sha256()
            digest.update(model_hash.encode("ascii"))
            digest.update(np.ascontiguousarray(states[frame]).tobytes())
            digest.update(f"{episode_index}:{frame}".encode("ascii"))
            record = {
                "record_version": 1,
                "dataset": str(root),
                "environment": "SteamInMicrowave",
                "episode_index": episode_index,
                "layout_id": int(ep_meta["layout_id"]),
                "style_id": int(ep_meta["style_id"]),
                "instruction": str(ep_meta.get("lang", "")),
                "segment": segment.to_dict(),
                "frame_index": frame,
                "expert_base_pose": base_pose_from_state(observations[frame]),
                "entry_pose_stability": prefix_pose_stability(
                    observations, segment, window=stability_window
                ),
                "expected_holding_at_entry": operation["operation"] == "place",
                **operation,
                "assets": {
                    "ep_meta": str(
                        (episode_output / "ep_meta.json").relative_to(output)
                    ),
                    "model_xml_gz": str(
                        (episode_output / "model.xml.gz").relative_to(output)
                    ),
                    "state": str(state_path.relative_to(output)),
                },
                "snapshot_id": digest.hexdigest(),
            }
            write_json(stage_dir / "metadata.json", record)
            records.append(record)
    index = {
        "benchmark": "RoboCasaOfficialWorkPose",
        "version": 1,
        "dataset": str(root),
        "environment": "SteamInMicrowave",
        "num_records": len(records),
        "records": records,
        "rejected": rejected,
    }
    write_json(output / "index.json", index)
    return index


def generate_pose_perturbations(
    records: Sequence[Mapping[str, Any]],
    *,
    difficulties: Sequence[str],
    samples_per_stage: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Generate deterministic random planar offsets in the expert robot frame."""

    if samples_per_stage <= 0:
        raise ValueError("samples_per_stage must be positive")
    unknown = sorted(set(difficulties).difference(DEFAULT_PERTURBATION_RANGES))
    if unknown:
        raise ValueError(f"unknown perturbation difficulties: {unknown}")
    rng = np.random.default_rng(seed)
    generated: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        expert_pose = record["expert_base_pose"]
        for difficulty in difficulties:
            radius_min, radius_max, yaw_min, yaw_max = (
                DEFAULT_PERTURBATION_RANGES[difficulty]
            )
            for sample_index in range(samples_per_stage):
                radius = float(rng.uniform(radius_min, radius_max))
                direction = float(rng.uniform(-math.pi, math.pi))
                yaw_magnitude = math.radians(float(rng.uniform(yaw_min, yaw_max)))
                yaw_delta = yaw_magnitude * (-1.0 if rng.random() < 0.5 else 1.0)
                forward = radius * math.cos(direction)
                left = radius * math.sin(direction)
                target_pose = apply_local_pose_delta(
                    expert_pose,
                    forward_m=forward,
                    left_m=left,
                    yaw_rad=yaw_delta,
                )
                sample_id = (
                    f"ep{int(record['episode_index']):06d}_"
                    f"s{int(record['segment']['subtask_idx']):02d}_"
                    f"{difficulty}_{sample_index:02d}"
                )
                generated.append(
                    {
                        "sample_id": sample_id,
                        "record_index": record_index,
                        "expert_snapshot_id": record["snapshot_id"],
                        "episode_index": int(record["episode_index"]),
                        "operation": record["operation"],
                        "object_id": record["object_id"],
                        "difficulty": difficulty,
                        "perturbation": {
                            "forward_m": forward,
                            "left_m": left,
                            "yaw_rad": yaw_delta,
                            "yaw_deg": math.degrees(yaw_delta),
                        },
                        "expert_base_pose": expert_pose,
                        "target_degraded_base_pose": target_pose,
                        "initial_pose_error": pose_error(target_pose, expert_pose),
                    }
                )
    return generated


def ensure_model_cameras(
    xml_string: str,
    camera_names: Sequence[str],
) -> str:
    """Inject configured base cameras when an older official XML lacks them."""

    try:
        from .utils.camera_utils import CAM_CONFIGS
    except ImportError:
        from robocasa.utils.camera_utils import CAM_CONFIGS
    root = ET.fromstring(xml_string)
    existing = {
        camera.get("name")
        for camera in root.iter("camera")
        if camera.get("name")
    }
    bodies = {
        body.get("name"): body
        for body in root.iter("body")
        if body.get("name")
    }
    for camera_name in camera_names:
        if camera_name in existing:
            continue
        config = CAM_CONFIGS["DEFAULT"].get(camera_name)
        if not isinstance(config, Mapping):
            raise KeyError(f"no camera configuration for {camera_name!r}")
        parent_name = str(config.get("parent_body", ""))
        parent = bodies.get(parent_name)
        if parent is None:
            raise KeyError(
                f"cannot inject {camera_name!r}: missing body {parent_name!r}"
            )
        attributes = {
            "name": camera_name,
            "pos": " ".join(str(value) for value in config["pos"]),
            "quat": " ".join(str(value) for value in config["quat"]),
        }
        attributes.update(
            {
                str(key): str(value)
                for key, value in dict(config.get("camera_attribs", {})).items()
            }
        )
        ET.SubElement(parent, "camera", attributes)
    return ET.tostring(root, encoding="unicode")


def create_dataset_environment(
    dataset: str | Path,
    *,
    camera_names: Sequence[str],
    camera_size: int = 256,
):
    """Create a raw pi0.5 environment compatible with the official states."""

    if camera_size <= 0:
        raise ValueError("camera_size must be positive")
    root = resolve_lerobot_root(dataset)
    import robosuite
    try:
        from .pi05_env import RawRoboCasaPi05Env
        from .utils.lerobot_utils import get_env_metadata
    except ImportError:
        from robocasa.pi05_env import RawRoboCasaPi05Env
        from robocasa.utils.lerobot_utils import get_env_metadata
    env_meta = deepcopy(get_env_metadata(root))
    kwargs = deepcopy(env_meta["env_kwargs"])
    kwargs.update(
        {
            "env_name": env_meta["env_name"],
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "camera_names": list(dict.fromkeys(camera_names)),
            "camera_heights": int(camera_size),
            "camera_widths": int(camera_size),
            "ignore_done": True,
        }
    )
    return RawRoboCasaPi05Env(robosuite.make(**kwargs))


def raw_task_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    candidates: list[Any] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        candidates.append(current)
        if hasattr(current, "unwrapped") and current.unwrapped is not current:
            current = current.unwrapped
        elif hasattr(current, "env"):
            current = current.env
        else:
            break
    for owner in reversed(candidates):
        if (
            callable(getattr(owner, "set_ep_meta", None))
            and callable(getattr(owner, "reset_from_xml_string", None))
        ):
            return owner
    raise ValueError("could not locate the RoboCasa task environment")


def set_flattened_state(env: Any, state: np.ndarray) -> dict[str, Any]:
    """Set one compatible MuJoCo state and refresh every observation cache."""

    raw_env = raw_task_env(env)
    raw_env.sim.set_state_from_flattened(np.asarray(state, dtype=np.float64))
    raw_env.sim.forward()
    if hasattr(raw_env, "update_state"):
        raw_env.update_state()
    observation = raw_env._get_observations(force_update=True)
    try:
        from .pi05_env import _process_observation
    except ImportError:
        from robocasa.pi05_env import _process_observation
    processed = _process_observation(observation)
    if hasattr(env, "_observation"):
        env._observation = processed
    return processed


def restore_official_state(
    env: Any,
    *,
    ep_meta_path: str | Path,
    model_xml_gz_path: str | Path,
    state: np.ndarray,
    camera_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Restore an exact official model/state and refresh wrapper observations."""

    raw_env = raw_task_env(env)
    ep_meta = json.loads(Path(ep_meta_path).read_text(encoding="utf-8"))
    with gzip.open(model_xml_gz_path, "rt", encoding="utf-8") as stream:
        model_xml = ensure_model_cameras(stream.read(), camera_names)
    raw_env.set_ep_meta(ep_meta)
    raw_env.reset()
    raw_env.reset_from_xml_string(raw_env.edit_model_xml(model_xml))
    raw_env.sim.reset()
    return set_flattened_state(env, state)


def current_base_pose(env: Any) -> dict[str, Any]:
    try:
        observation = env.get_observation()
    except (AttributeError, RuntimeError):
        observation = raw_task_env(env)._get_observations(force_update=True)
    quaternion = np.asarray(observation["robot0_base_quat"], dtype=float)
    return {
        "position": np.asarray(observation["robot0_base_pos"], dtype=float).tolist(),
        "quaternion_xyzw": quaternion.tolist(),
        "yaw_rad": quaternion_yaw_xyzw(quaternion),
    }


def move_base_to_pose(
    env: Any,
    target_pose: Mapping[str, Any],
    *,
    guard: Any | None = None,
    max_steps: int = 240,
    translation_tolerance_m: float = 0.025,
    yaw_tolerance_deg: float = 3.0,
    max_translation_command: float = 1.0,
    max_rotation_command: float = 0.50,
    settle_steps: int = 5,
) -> dict[str, Any]:
    """Move the holonomic base with native actions while holding arm pose."""

    target_position = np.asarray(target_pose["position"], dtype=np.float64)
    target_yaw = float(
        target_pose.get(
            "yaw_rad", quaternion_yaw_xyzw(target_pose["quaternion_xyzw"])
        )
    )
    if guard is not None:
        guard.start()
    stable_count = 0
    guard_failure = None
    trace: list[dict[str, Any]] = []
    for step_index in range(1, max_steps + 1):
        pose = current_base_pose(env)
        position = np.asarray(pose["position"], dtype=float)
        yaw = float(pose["yaw_rad"])
        world_error = target_position[:2] - position[:2]
        cosine, sine = math.cos(yaw), math.sin(yaw)
        forward_error = cosine * world_error[0] + sine * world_error[1]
        left_error = -sine * world_error[0] + cosine * world_error[1]
        yaw_error = wrap_angle(target_yaw - yaw)
        translation_error = float(np.linalg.norm(world_error))
        within = (
            translation_error <= translation_tolerance_m
            and abs(yaw_error) <= math.radians(yaw_tolerance_deg)
        )
        stable_count = stable_count + 1 if within else 0
        trace.append(
            {
                "step": step_index,
                "translation_error_m": translation_error,
                "yaw_error_deg": math.degrees(yaw_error),
            }
        )
        if stable_count >= 3:
            break
        action = np.zeros(12, dtype=np.float32)
        action[RAW_ACTION_BASE.start] = np.clip(
            4.0 * forward_error, -max_translation_command, max_translation_command
        )
        action[RAW_ACTION_BASE.start + 1] = np.clip(
            4.0 * left_error, -max_translation_command, max_translation_command
        )
        action[RAW_ACTION_BASE.start + 2] = np.clip(
            2.0 * yaw_error, -max_rotation_command, max_rotation_command
        )
        action[RAW_ACTION_MODE] = 1.0
        applied = (
            guard.apply_action(action, step_index=step_index)
            if guard is not None
            else action
        )
        _, _, done, _ = env.step(applied)
        if guard is not None:
            guard_failure = guard.observe(step_index=step_index)
        if done or guard_failure is not None:
            break
    for settle_index in range(settle_steps):
        action = np.zeros(12, dtype=np.float32)
        action[RAW_ACTION_MODE] = 1.0
        step_index = len(trace) + settle_index + 1
        applied = (
            guard.apply_action(action, step_index=step_index)
            if guard is not None
            else action
        )
        env.step(applied)
        if guard is not None:
            guard_failure = guard.observe(step_index=step_index)
            if guard_failure is not None:
                break
    final_pose = current_base_pose(env)
    error = pose_error(final_pose, target_pose)
    return {
        "success": bool(
            guard_failure is None
            and error["translation_m"] <= translation_tolerance_m
            and error["yaw_deg"] <= yaw_tolerance_deg
        ),
        "steps": len(trace),
        "target_pose": dict(target_pose),
        "final_pose": final_pose,
        "final_error": error,
        "guard_failure": guard_failure,
        "trace": trace,
    }


def resolve_record_asset(
    expert_index_path: str | Path,
    record: Mapping[str, Any],
    key: str,
) -> Path:
    index_root = Path(expert_index_path).expanduser().resolve().parent
    value = record.get("assets", {}).get(key)
    if not value:
        raise KeyError(f"expert record has no asset {key!r}")
    return (index_root / str(value)).resolve()


def load_record_state(
    expert_index_path: str | Path,
    record: Mapping[str, Any],
) -> np.ndarray:
    data = np.load(resolve_record_asset(expert_index_path, record, "state"))
    key = "state" if "state" in data else "states"
    value = np.asarray(data[key])
    return value[0] if value.ndim == 2 and value.shape[0] == 1 else value


def scene_context(env: Any) -> dict[str, Any]:
    raw_env = raw_task_env(env)
    return {
        "fixtures": [
            {
                "alias": str(alias),
                "name": str(getattr(entity, "name", alias)),
                "natural_name": str(getattr(entity, "nat_lang", alias)),
            }
            for alias, entity in getattr(raw_env, "fixtures", {}).items()
        ],
        "objects": [
            {
                "alias": str(alias),
                "name": str(getattr(entity, "name", alias)),
                "natural_name": (
                    str(raw_env.get_obj_lang(alias))
                    if callable(getattr(raw_env, "get_obj_lang", None))
                    else str(alias)
                ),
            }
            for alias, entity in getattr(raw_env, "objects", {}).items()
        ],
    }


def holding_state(env: Any, object_id: str) -> bool | None:
    try:
        from .utils import object_utils
    except ImportError:
        from robocasa.utils import object_utils
    try:
        return bool(
            object_utils.check_obj_grasped(
                raw_task_env(env), object_id, threshold=0.05
            )
        )
    except (AssertionError, AttributeError, KeyError, TypeError, ValueError):
        return None
