from __future__ import annotations

import json
import time
import webbrowser
from dataclasses import dataclass
from typing import Any
from urllib import request

from physicsos.cloud.config import CloudConfig, DEFAULT_RUNNER_URL, save_cloud_config


@dataclass(frozen=True)
class DeviceLoginResult:
    runner_url: str
    user_code: str
    verification_url: str
    access_token: str


def _json_request(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(
        url,
        data=body,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=30) as response:
        text = response.read().decode("utf-8")
    return json.loads(text) if text else {}


def start_device_login(
    runner_url: str = DEFAULT_RUNNER_URL,
    *,
    open_browser: bool = True,
    poll_interval_seconds: float = 2.0,
    timeout_seconds: float = 600.0,
) -> DeviceLoginResult:
    """Run the foamvm device-code login flow and persist the returned CLI token."""
    base = runner_url.rstrip("/")
    started = _json_request("POST", f"{base}/api/cli/device/start", {"client": "physicsos-cli"})
    user_code = str(started["user_code"])
    device_code = str(started["device_code"])
    verification_url = str(started["verification_url"])

    print(f"Open {verification_url}")
    print(f"Device code: {user_code}")
    if open_browser:
        webbrowser.open(verification_url)

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        polled = _json_request("POST", f"{base}/api/cli/device/poll", {"device_code": device_code})
        status = polled.get("status")
        if status == "approved":
            access_token = str(polled["access_token"])
            save_cloud_config(CloudConfig(runner_url=base, access_token=access_token))
            return DeviceLoginResult(
                runner_url=base,
                user_code=user_code,
                verification_url=verification_url,
                access_token=access_token,
            )
        if status in {"expired", "revoked"}:
            raise PermissionError(f"Device login {status}.")
        time.sleep(poll_interval_seconds)

    raise TimeoutError("Device login timed out.")
