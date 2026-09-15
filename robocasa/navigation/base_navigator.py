"""Execute smoothed navigation waypoints with the existing base controller."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from robocasa.work_pose_dataset import (
    current_base_pose,
    move_base_to_pose,
    yaw_quaternion_xyzw,
)

from .navigation_types import NavigationExecutionResult


def _circular_yaw_error_deg(target: float, actual: float) -> float:
    error = (float(target) - float(actual) + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(error))


class BaseNavigator:
    """Waypoint follower backed by the already validated world-pose controller."""

    def __init__(self, config: Mapping[str, Any]):
        # New names are authoritative; aliases preserve older navigation configs.
        self.waypoint_position_tolerance = float(
            config.get("waypoint_position_tolerance", config.get("waypoint_tolerance", 0.12))
        )
        self.final_position_tolerance = float(
            config.get("final_position_tolerance", config.get("goal_position_tolerance", 0.08))
        )
        self.controller_final_position_tolerance = float(
            config.get(
                "controller_final_position_tolerance",
                max(0.01, self.final_position_tolerance - 0.01),
            )
        )
        self.final_yaw_tolerance_deg = float(
            config.get("final_yaw_tolerance_deg", config.get("goal_yaw_tolerance_deg", 10.0))
        )
        self.waypoint_max_steps = int(config.get("waypoint_max_steps", 180))
        self.final_position_max_steps = int(
            config.get("final_position_max_steps", config.get("goal_max_steps", 240))
        )
        self.final_yaw_max_steps = int(
            config.get("final_yaw_max_steps", config.get("goal_max_steps", 240))
        )
        self.max_translation_command = float(config.get("max_translation_command", 0.75))
        self.max_rotation_command = float(config.get("max_rotation_command", 0.50))
        self.max_waypoint_retries = int(config.get("max_waypoint_retries", 1))
        if min(
            self.waypoint_position_tolerance,
            self.final_position_tolerance,
            self.final_yaw_tolerance_deg,
            self.controller_final_position_tolerance,
            self.waypoint_max_steps,
            self.final_position_max_steps,
            self.final_yaw_max_steps,
            self.max_translation_command,
            self.max_rotation_command,
        ) <= 0:
            raise ValueError("base navigation tolerances and limits must be positive")
        if self.controller_final_position_tolerance > self.final_position_tolerance:
            raise ValueError(
                "controller final position tolerance must not exceed final acceptance tolerance"
            )
        if self.max_waypoint_retries < 0:
            raise ValueError("max_waypoint_retries must be non-negative")

        # Compatibility attributes used by earlier callers and tests.
        self.waypoint_tolerance = self.waypoint_position_tolerance
        self.goal_position_tolerance = self.final_position_tolerance
        self.goal_yaw_tolerance_deg = self.final_yaw_tolerance_deg
        self.goal_max_steps = self.final_position_max_steps

    @staticmethod
    def _pose(xy: Sequence[float], z: float, yaw: float) -> dict[str, Any]:
        return {
            "position": [float(xy[0]), float(xy[1]), float(z)],
            "quaternion_xyzw": yaw_quaternion_xyzw(float(yaw)),
            "yaw_rad": float(yaw),
        }

    @staticmethod
    def _distance(pose: Mapping[str, Any], xy: Sequence[float]) -> float:
        position = pose["position"]
        return math.hypot(float(xy[0]) - float(position[0]), float(xy[1]) - float(position[1]))

    @staticmethod
    def _guard_state(guard: Any | None) -> dict[str, Any] | None:
        return guard.to_dict() if guard is not None else None

    def _failure(
        self,
        *,
        reason: str,
        reached: int,
        total: int,
        steps: int,
        traces: list[dict[str, Any]],
        guard: Any | None,
        goal_xy: Sequence[float],
        goal_yaw: float,
        env: Any,
    ) -> NavigationExecutionResult:
        final_pose = current_base_pose(env)
        return NavigationExecutionResult(
            success=False,
            reason=reason,
            reached_waypoints=reached,
            total_waypoints=total,
            final_pose=final_pose,
            position_error=self._distance(final_pose, goal_xy),
            yaw_error=_circular_yaw_error_deg(goal_yaw, float(final_pose["yaw_rad"])),
            steps=steps,
            waypoint_traces=traces,
            held_object_guard=self._guard_state(guard),
        )

    def execute_path(
        self,
        *,
        env: Any,
        path_world: Sequence[Sequence[float]],
        goal_yaw: float,
        held_object_context: Any | None = None,
    ) -> NavigationExecutionResult:
        """Execute only smoothed waypoints, then rotate separately at the goal."""

        if not path_world:
            raise ValueError("navigation path must contain at least one waypoint")
        guard = held_object_context
        initial = current_base_pose(env)
        z = float(initial["position"][2])
        goals = list(path_world[1:])
        if not goals:
            goals = [path_world[-1]]
        final_xy = goals[-1]
        total = len(goals)
        reached = 0
        total_steps = 0
        traces: list[dict[str, Any]] = []

        for waypoint_index, waypoint in enumerate(goals, start=1):
            is_final = waypoint_index == total
            tolerance = (
                self.controller_final_position_tolerance
                if is_final
                else self.waypoint_position_tolerance
            )
            max_steps = self.final_position_max_steps if is_final else self.waypoint_max_steps
            for attempt in range(self.max_waypoint_retries + 1):
                pose_before = current_base_pose(env)
                dx = float(waypoint[0]) - float(pose_before["position"][0])
                dy = float(waypoint[1]) - float(pose_before["position"][1])
                distance_before = math.hypot(dx, dy)
                heading = math.atan2(dy, dx) if distance_before > 1e-9 else float(pose_before["yaw_rad"])
                heading_error = _circular_yaw_error_deg(heading, float(pose_before["yaw_rad"]))
                print(
                    "[NAV-EXEC] waypoint {}/{} start attempt={} "
                    "current=({:.3f},{:.3f},{:.1f}deg) target=({:.3f},{:.3f}) "
                    "distance={:.3f} heading_error={:.1f} action=TRANSLATE".format(
                        waypoint_index,
                        total,
                        attempt + 1,
                        float(pose_before["position"][0]),
                        float(pose_before["position"][1]),
                        math.degrees(float(pose_before["yaw_rad"])),
                        float(waypoint[0]),
                        float(waypoint[1]),
                        distance_before,
                        heading_error,
                    )
                )
                # The Omron base is holonomic. Preserve current yaw during XY motion;
                # intermediate waypoints never require yaw convergence.
                motion = move_base_to_pose(
                    env,
                    self._pose(waypoint, z, float(pose_before["yaw_rad"])),
                    guard=guard,
                    max_steps=max_steps,
                    translation_tolerance_m=tolerance,
                    yaw_tolerance_deg=180.0,
                    max_translation_command=self.max_translation_command,
                    max_rotation_command=self.max_rotation_command,
                    settle_steps=0,
                )
                total_steps += int(motion.get("steps", 0))
                pose_after = current_base_pose(env)
                distance_after = self._distance(pose_after, waypoint)
                trace = {
                    "waypoint_index": waypoint_index,
                    "total_waypoints": total,
                    "is_final_position": is_final,
                    "attempt": attempt + 1,
                    "target": [float(waypoint[0]), float(waypoint[1])],
                    "distance_before_m": distance_before,
                    "heading_error_before_deg": heading_error,
                    "distance_after_m": distance_after,
                    "tolerance_m": tolerance,
                    "motion": motion,
                }
                post_waypoint_guard_failure = (
                    guard.observe(step_index=max(total_steps, 1))
                    if guard is not None
                    else None
                )
                trace["post_waypoint_guard_failure"] = post_waypoint_guard_failure
                traces.append(trace)
                if (
                    motion.get("guard_failure") is not None
                    or post_waypoint_guard_failure is not None
                ):
                    print(
                        "[NAV-EXEC] waypoint {}/{} failure reason=OBJECT_DROPPED".format(
                            waypoint_index, total
                        )
                    )
                    return self._failure(
                        reason="OBJECT_DROPPED",
                        reached=reached,
                        total=total,
                        steps=total_steps,
                        traces=traces,
                        guard=guard,
                        goal_xy=final_xy,
                        goal_yaw=goal_yaw,
                        env=env,
                    )
                if distance_after <= tolerance:
                    reached += 1
                    print(
                        "[NAV-EXEC] waypoint {}/{} complete distance={:.3f}m action=REACHED".format(
                            waypoint_index, total, distance_after
                        )
                    )
                    break
                if attempt < self.max_waypoint_retries:
                    print(
                        "[NAV-EXEC] waypoint {}/{} failure reason=MOTION_STALLED "
                        "distance={:.3f}m retry=1".format(
                            waypoint_index, total, distance_after
                        )
                    )
                    continue
                print(
                    "[NAV-EXEC] waypoint {}/{} failure reason=MOTION_STALLED distance={:.3f}m".format(
                        waypoint_index, total, distance_after
                    )
                )
                return self._failure(
                    reason="MOTION_STALLED",
                    reached=reached,
                    total=total,
                    steps=total_steps,
                    traces=traces,
                    guard=guard,
                    goal_xy=final_xy,
                    goal_yaw=goal_yaw,
                    env=env,
                )

        # Final orientation is intentionally separate from waypoint translation.
        for attempt in range(self.max_waypoint_retries + 1):
            pose_before = current_base_pose(env)
            yaw_before = _circular_yaw_error_deg(goal_yaw, float(pose_before["yaw_rad"]))
            print(
                "[NAV-EXEC] final yaw start attempt={} current={:.1f}deg "
                "target={:.1f}deg error={:.1f}deg action=ROTATE".format(
                    attempt + 1,
                    math.degrees(float(pose_before["yaw_rad"])),
                    math.degrees(float(goal_yaw)),
                    yaw_before,
                )
            )
            motion = move_base_to_pose(
                env,
                self._pose(final_xy, z, goal_yaw),
                guard=guard,
                max_steps=self.final_yaw_max_steps,
                translation_tolerance_m=self.controller_final_position_tolerance,
                yaw_tolerance_deg=self.final_yaw_tolerance_deg,
                max_translation_command=self.max_translation_command,
                max_rotation_command=self.max_rotation_command,
                settle_steps=3,
            )
            total_steps += int(motion.get("steps", 0))
            pose_after = current_base_pose(env)
            position_error = self._distance(pose_after, final_xy)
            yaw_error = _circular_yaw_error_deg(goal_yaw, float(pose_after["yaw_rad"]))
            post_yaw_guard_failure = (
                guard.observe(step_index=max(total_steps, 1))
                if guard is not None
                else None
            )
            traces.append(
                {
                    "stage": "final_yaw",
                    "attempt": attempt + 1,
                    "target_yaw_deg": math.degrees(float(goal_yaw)),
                    "position_error_m": position_error,
                    "yaw_error_deg": yaw_error,
                    "motion": motion,
                    "post_waypoint_guard_failure": post_yaw_guard_failure,
                }
            )
            if (
                motion.get("guard_failure") is not None
                or post_yaw_guard_failure is not None
            ):
                print("[NAV-EXEC] final yaw failure reason=OBJECT_DROPPED")
                return self._failure(
                    reason="OBJECT_DROPPED",
                    reached=reached,
                    total=total,
                    steps=total_steps,
                    traces=traces,
                    guard=guard,
                    goal_xy=final_xy,
                    goal_yaw=goal_yaw,
                    env=env,
                )
            if (
                position_error <= self.final_position_tolerance
                and yaw_error <= self.final_yaw_tolerance_deg
            ):
                print(
                    "[NAV-EXEC] final complete position_error={:.3f}m yaw_error={:.1f}deg".format(
                        position_error, yaw_error
                    )
                )
                return NavigationExecutionResult(
                    success=True,
                    reason="SUCCESS",
                    reached_waypoints=reached,
                    total_waypoints=total,
                    final_pose=pose_after,
                    position_error=position_error,
                    yaw_error=yaw_error,
                    steps=total_steps,
                    waypoint_traces=traces,
                    held_object_guard=self._guard_state(guard),
                )
            if attempt < self.max_waypoint_retries:
                print(
                    "[NAV-EXEC] final yaw failure reason=FINAL_POSE_NOT_REACHED "
                    "position_error={:.3f}m yaw_error={:.1f}deg retry=1".format(
                        position_error, yaw_error
                    )
                )

        print(
            "[NAV-EXEC] final failure reason=FINAL_POSE_NOT_REACHED "
            "position_error={:.3f}m yaw_error={:.1f}deg".format(
                position_error, yaw_error
            )
        )
        return self._failure(
            reason="FINAL_POSE_NOT_REACHED",
            reached=reached,
            total=total,
            steps=total_steps,
            traces=traces,
            guard=guard,
            goal_xy=final_xy,
            goal_yaw=goal_yaw,
            env=env,
        )

    def navigate_path(
        self,
        env: Any,
        waypoints: Sequence[Sequence[float]],
        goal_yaw: float,
        *,
        guard: Any | None = None,
    ) -> dict[str, Any]:
        """Backward-compatible dict wrapper for existing pipeline callers."""

        return self.execute_path(
            env=env,
            path_world=waypoints,
            goal_yaw=goal_yaw,
            held_object_context=guard,
        ).to_dict()
