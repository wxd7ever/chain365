"""Central factory for remote manipulation-policy backends."""

from __future__ import annotations

from typing import Any, Mapping

try:
    from .openpi_client import OpenPIWebsocketClient
    from .xr1_policy_client import XR1RoboCasaPolicy, check_xr1_connectivity
except ImportError:  # Direct script execution from the robocasa directory.
    from openpi_client import OpenPIWebsocketClient
    from xr1_policy_client import XR1RoboCasaPolicy, check_xr1_connectivity


def create_remote_policy_client(
    *,
    args: Any,
    initial_observation: Mapping[str, Any],
    env: Any | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    """Create exactly one configured backend and return serializable metadata."""

    if args.policy_backend == "pi05":
        client = OpenPIWebsocketClient(
            host=args.pi05_host,
            port=args.pi05_port,
            api_key=args.pi05_api_key,
            connect_timeout_s=args.pi05_connect_timeout_s,
            infer_timeout_s=args.pi05_infer_timeout_s,
            max_retries=args.pi05_max_retries,
        )
        metadata = {
            "policy_backend": "pi05",
            "uri": client.uri,
            "server_metadata": client.server_metadata,
        }
        print(f"[Pi05] Connected to {client.uri}")
        return client, "pi05_server.json", metadata

    if args.policy_backend == "xr1":
        print(f"[XR1] connectivity check {args.xr1_host}:{args.xr1_port}")
        check_xr1_connectivity(
            args.xr1_host, args.xr1_port, args.xr1_connect_timeout_s
        )
        client = XR1RoboCasaPolicy(
            host=args.xr1_host,
            port=args.xr1_port,
            model_path=args.xr1_model_path,
            connect_timeout_s=args.xr1_connect_timeout_s,
            infer_timeout_s=args.xr1_infer_timeout_s,
            obs_history=args.xr1_obs_history,
            obs_interval=args.xr1_obs_interval,
            replan_steps=args.xr1_replan_steps,
            crop_ratio=args.xr1_crop_ratio,
        )
        add_listener = getattr(env, "add_observation_listener", None)
        if callable(add_listener):
            add_listener(client.observe)
            client.environment_observer_attached = True
        # Only an episode reset clears temporal history. Atomic skill switches
        # are handled later through set_instruction(), which preserves history.
        client.reset(initial_observation)
        return client, "xr1_server.json", client.diagnostics()

    raise ValueError(f"Unsupported policy backend {args.policy_backend!r}")
