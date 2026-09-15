"""Live MuJoCo collision geometry for articulated RoboCasa fixtures.

The static navigation map is based on ``Fixture.get_ext_sites``. Those sites
describe the fixture's layout footprint and do not follow door / drawer joint
state. This module deliberately starts from fixture joints, walks their MuJoCo
body descendants, and reads collision geoms from ``sim.data`` so the resulting
polygons represent the current episode state rather than the MJCF rest pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .navigation_types import raw_task_env


# MuJoCo's stable mjtGeom integer values. Keeping these local also lets the
# extraction utilities run against the lightweight fake models used in tests.
_GEOM_NAMES = {
    0: "plane",
    1: "hfield",
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh",
}
_ARTICULATED_JOINT_TOKENS = (
    "door",
    "hinge",
    "drawer",
    "slidejoint",
    "rack",
    "microjoint",
)
_NON_OBSTACLE_JOINT_TOKENS = (
    "button",
    "knob",
    "switch",
    "spout",
    "faucet",
)


@dataclass(frozen=True)
class ArticulatedCollisionGeometry:
    fixture_name: str
    fixture_type: str
    component: str
    joint_name: str
    joint_state: float | None
    geom_name: str
    geom_type: str
    world_position: np.ndarray
    world_rotation: np.ndarray
    size: np.ndarray
    polygon_xy: np.ndarray
    z_min: float
    z_max: float
    body_name: str

    def to_dict(self) -> dict[str, Any]:
        bounds = [
            float(self.polygon_xy[:, 0].min()),
            float(self.polygon_xy[:, 1].min()),
            float(self.polygon_xy[:, 0].max()),
            float(self.polygon_xy[:, 1].max()),
        ]
        return {
            "fixture_name": self.fixture_name,
            "fixture_type": self.fixture_type,
            "component": self.component,
            "joint_name": self.joint_name,
            "joint_state": self.joint_state,
            "geom_name": self.geom_name,
            "geom_type": self.geom_type,
            "world_position": self.world_position.tolist(),
            "world_rotation": self.world_rotation.tolist(),
            "size": self.size.tolist(),
            "polygon_xy": self.polygon_xy.tolist(),
            "projected_bounds": bounds,
            "z_range": [self.z_min, self.z_max],
            "body_name": self.body_name,
        }


def _model_name(model: Any, kind: str, index: int) -> str:
    converter = getattr(model, f"{kind}_id2name", None)
    if callable(converter):
        value = converter(int(index))
        return "" if value is None else str(value)
    names = getattr(model, f"{kind}_names", None)
    if names is not None and 0 <= int(index) < len(names):
        value = names[int(index)]
        return "" if value is None else str(value)
    return ""


def _model_id(model: Any, kind: str, name: str) -> int | None:
    converter = getattr(model, f"{kind}_name2id", None)
    if callable(converter):
        try:
            return int(converter(name))
        except (KeyError, TypeError, ValueError):
            return None
    names = getattr(model, f"{kind}_names", None)
    if names is not None:
        try:
            return list(names).index(name)
        except ValueError:
            return None
    return None


def _unique_strings(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _fixture_articulated_joint_names(fixture: Any) -> list[str]:
    """Resolve movable obstruction joints from fixture metadata, not geom names."""

    explicit: list[Any] = []
    try:
        explicit.extend(list(getattr(fixture, "door_joint_names", ()) or ()))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    for attribute in (
        "_fridge_door_joint_names",
        "_freezer_door_joint_names",
        "_drawer_joint_names",
    ):
        explicit.extend(list(getattr(fixture, attribute, ()) or ()))
    named = getattr(fixture, "_joint_names", {})
    if isinstance(named, Mapping):
        for role, joint_name in named.items():
            role_lower = str(role).lower()
            if any(token in role_lower for token in ("door", "drawer", "rack")):
                explicit.append(joint_name)

    candidates = list(explicit)
    joint_infos = getattr(fixture, "_joint_infos", {})
    if isinstance(joint_infos, Mapping):
        for joint_name in joint_infos:
            lower = str(joint_name).lower()
            if any(token in lower for token in _NON_OBSTACLE_JOINT_TOKENS):
                continue
            if any(token in lower for token in _ARTICULATED_JOINT_TOKENS):
                candidates.append(joint_name)
    return _unique_strings(candidates)


def _component_for_joint(joint_name: str) -> str:
    lower = joint_name.lower()
    if "rack" in lower:
        return "rack"
    if "drawer" in lower or "slidejoint" in lower:
        return "drawer"
    return "door"


def _descendant_body_ids(model: Any, root_body_id: int) -> set[int]:
    parent_ids = np.asarray(model.body_parentid, dtype=int).reshape(-1)
    descendants = {int(root_body_id)}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(parent_ids):
            if body_id not in descendants and int(parent_id) in descendants:
                descendants.add(int(body_id))
                changed = True
    return descendants


def _convex_hull(points: Sequence[Sequence[float]]) -> np.ndarray:
    values = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(values) <= 2:
        return np.asarray(values, dtype=float)

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (
            a[1] - origin[1]
        ) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in values:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-12:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(values):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-12:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _box_points(center: np.ndarray, rotation: np.ndarray, size: np.ndarray) -> np.ndarray:
    signs = np.asarray(
        [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=float,
    )
    return center + (rotation @ (signs * size[:3]).T).T


def _round_geom_points(
    center: np.ndarray,
    rotation: np.ndarray,
    size: np.ndarray,
    *,
    geom_type: int,
    samples: int = 24,
) -> np.ndarray:
    radius = float(size[0])
    half_length = float(size[1]) if geom_type in (3, 5) else 0.0
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    local: list[list[float]] = []
    if geom_type == 2:  # sphere
        for angle in angles:
            local.append([radius * math.cos(angle), radius * math.sin(angle), 0.0])
        local.extend(([0.0, 0.0, -radius], [0.0, 0.0, radius]))
    elif geom_type == 4:  # ellipsoid: three principal great-circle samples
        radii = np.asarray(size[:3], dtype=float)
        for first, second in ((0, 1), (0, 2), (1, 2)):
            for angle in angles:
                point = np.zeros(3, dtype=float)
                point[first] = radii[first] * math.cos(angle)
                point[second] = radii[second] * math.sin(angle)
                local.append(point.tolist())
    else:  # capsule / cylinder around local z
        for z in (-half_length, half_length):
            for angle in angles:
                local.append([radius * math.cos(angle), radius * math.sin(angle), z])
        if geom_type == 3:
            # Capsule end spheres contribute radius in the axial direction too.
            local.extend(
                ([0.0, 0.0, -half_length - radius], [0.0, 0.0, half_length + radius])
            )
    return center + (rotation @ np.asarray(local, dtype=float).T).T


def _mesh_points(
    model: Any,
    geom_id: int,
    center: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray | None:
    try:
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            return None
        return center + (rotation @ vertices.T).T
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def geom_world_points(model: Any, data: Any, geom_id: int) -> np.ndarray | None:
    """Return representative world points for a supported current MuJoCo geom."""

    geom_type = int(model.geom_type[geom_id])
    center = np.asarray(data.geom_xpos[geom_id], dtype=float).reshape(3)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=float).reshape(-1)
    if geom_type == 6 and size.size >= 3:
        return _box_points(center, rotation, size)
    if geom_type in (2, 3, 4, 5) and size.size >= (3 if geom_type == 4 else 1):
        return _round_geom_points(
            center, rotation, size, geom_type=geom_type
        )
    if geom_type == 7:
        return _mesh_points(model, geom_id, center, rotation)
    return None


def _projected_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def get_robot_navigation_collision_z_range(
    env: Any,
    *,
    fallback: Sequence[float] = (0.0, 1.8),
) -> tuple[float, float]:
    """Derive the live mobile-manipulator collision-height envelope.

    The map footprint describes the Omron base in XY, but a protruding door can
    collide with the support, arm, or gripper above the wheeled pedestal. We
    therefore use all collision geoms belonging to robot/mobilebase/gripper
    bodies for the vertical overlap test. The configured fallback is used only
    by wrappers that do not expose those MuJoCo arrays.
    """

    raw_env = raw_task_env(env)
    try:
        model, data = raw_env.sim.model, raw_env.sim.data
        lower: list[float] = []
        upper: list[float] = []
        for geom_id in range(int(model.ngeom)):
            body_name = _model_name(model, "body", int(model.geom_bodyid[geom_id]))
            geom_name = _model_name(model, "geom", geom_id)
            combined = f"{body_name} {geom_name}".lower()
            if not any(token in combined for token in ("robot", "mobilebase", "gripper")):
                continue
            if not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
                continue
            points = geom_world_points(model, data, geom_id)
            if points is not None and len(points):
                lower.append(float(points[:, 2].min()))
                upper.append(float(points[:, 2].max()))
            else:
                radius = float(model.geom_rbound[geom_id])
                z = float(data.geom_xpos[geom_id][2])
                lower.append(z - radius)
                upper.append(z + radius)
        if lower and upper:
            return min(lower), max(upper)
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    values = np.asarray(fallback, dtype=float).reshape(-1)
    if values.size != 2 or not np.isfinite(values).all() or values[0] >= values[1]:
        raise ValueError("robot collision z fallback must be [min, max]")
    return float(values[0]), float(values[1])


def get_articulated_collision_geometries(
    env: Any,
    *,
    robot_z_range: Sequence[float] | None = None,
    minimum_projected_area_m2: float = 0.001,
    minimum_projected_span_m: float = 0.015,
) -> list[ArticulatedCollisionGeometry]:
    """Read current collision geometry below articulated fixture joint bodies."""

    raw_env = raw_task_env(env)
    try:
        model, data = raw_env.sim.model, raw_env.sim.data
        int(model.ngeom)
    except (AttributeError, TypeError, ValueError):
        return []
    z_min, z_max = (
        get_robot_navigation_collision_z_range(env)
        if robot_z_range is None
        else (float(robot_z_range[0]), float(robot_z_range[1]))
    )
    output: list[ArticulatedCollisionGeometry] = []
    seen_fixtures: set[int] = set()
    seen_geoms: set[int] = set()
    for alias, fixture in getattr(raw_env, "fixtures", {}).items():
        identity = id(fixture)
        if identity in seen_fixtures:
            continue
        seen_fixtures.add(identity)
        fixture_name = str(getattr(fixture, "name", alias))
        fixture_type = type(fixture).__name__
        for joint_name in _fixture_articulated_joint_names(fixture):
            joint_id = _model_id(model, "joint", joint_name)
            if joint_id is None:
                continue
            try:
                root_body = int(model.jnt_bodyid[joint_id])
                bodies = _descendant_body_ids(model, root_body)
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
            try:
                qpos = np.asarray(data.get_joint_qpos(joint_name), dtype=float).reshape(-1)
                joint_state = float(qpos[0]) if qpos.size else None
            except (AttributeError, KeyError, TypeError, ValueError):
                joint_state = None
            for geom_id in range(int(model.ngeom)):
                if geom_id in seen_geoms or int(model.geom_bodyid[geom_id]) not in bodies:
                    continue
                if not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
                    continue
                points = geom_world_points(model, data, geom_id)
                if points is None or len(points) < 3 or not np.isfinite(points).all():
                    continue
                geom_z_min = float(points[:, 2].min())
                geom_z_max = float(points[:, 2].max())
                if geom_z_max < z_min or geom_z_min > z_max:
                    continue
                polygon = _convex_hull(points[:, :2])
                if len(polygon) < 3:
                    continue
                spans = polygon.max(axis=0) - polygon.min(axis=0)
                if max(float(spans[0]), float(spans[1])) < minimum_projected_span_m:
                    continue
                if _projected_area(polygon) < minimum_projected_area_m2:
                    continue
                geom_name = _model_name(model, "geom", geom_id) or f"geom_{geom_id}"
                body_name = _model_name(model, "body", int(model.geom_bodyid[geom_id]))
                output.append(
                    ArticulatedCollisionGeometry(
                        fixture_name=fixture_name,
                        fixture_type=fixture_type,
                        component=_component_for_joint(joint_name),
                        joint_name=joint_name,
                        joint_state=joint_state,
                        geom_name=geom_name,
                        geom_type=_GEOM_NAMES.get(
                            int(model.geom_type[geom_id]), "unknown"
                        ),
                        world_position=np.asarray(
                            data.geom_xpos[geom_id], dtype=float
                        ).copy(),
                        world_rotation=np.asarray(
                            data.geom_xmat[geom_id], dtype=float
                        ).reshape(3, 3).copy(),
                        size=np.asarray(model.geom_size[geom_id], dtype=float).copy(),
                        polygon_xy=polygon,
                        z_min=geom_z_min,
                        z_max=geom_z_max,
                        body_name=body_name,
                    )
                )
                seen_geoms.add(geom_id)
    return output
