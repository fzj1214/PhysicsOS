from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import parse, request

from physicsos.cloud.config import CloudConfig, load_cloud_config


@dataclass(frozen=True)
class FoamVMClient:
    runner_url: str
    access_token: str

    @classmethod
    def from_config(cls, config: CloudConfig | None = None) -> "FoamVMClient":
        loaded = config or load_cloud_config()
        if not loaded.access_token:
            raise PermissionError("Missing PhysicsOS CLI token. Run `physicsos auth login` first.")
        return cls(runner_url=loaded.runner_url.rstrip("/"), access_token=loaded.access_token)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = request.Request(
            self.runner_url + path,
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.access_token}",
            },
        )
        with request.urlopen(req, timeout=30) as response:
            text = response.read().decode("utf-8")
        return json.loads(text) if text else {}

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/api/physicsos/me")

    def submit_job(self, manifest: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/physicsos/jobs", manifest)

    def job_status(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/physicsos/jobs/{parse.quote(job_id)}")

    def job_events(self, job_id: str, after: int | None = None) -> dict[str, Any]:
        suffix = f"?after={after}" if after is not None else ""
        return self._request("GET", f"/api/physicsos/jobs/{parse.quote(job_id)}/events{suffix}")

    def job_artifacts(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/physicsos/jobs/{parse.quote(job_id)}/artifacts")
