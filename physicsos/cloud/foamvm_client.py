from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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

    def _request_bytes(self, path_or_url: str) -> tuple[bytes, str | None]:
        url = path_or_url if path_or_url.startswith(("http://", "https://")) else self.runner_url + path_or_url
        req = request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/octet-stream",
                "Authorization": f"Bearer {self.access_token}",
            },
        )
        with request.urlopen(req, timeout=120) as response:
            filename = response.headers.get_filename()
            return response.read(), filename

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

    def download_artifact(self, job_id: str, artifact_id: str, output_dir: str | Path = ".") -> dict[str, Any]:
        artifacts_payload = self.job_artifacts(job_id)
        artifacts = artifacts_payload.get("artifacts", [])
        artifact = next((item for item in artifacts if str(item.get("id")) == artifact_id), None)
        if artifact is None:
            raise ValueError(f"Artifact not found for job {job_id}: {artifact_id}")
        url = artifact.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError(f"Artifact has no download URL: {artifact_id}")
        data, header_filename = self._request_bytes(url)
        filename = header_filename or str(artifact.get("filename") or artifact_id)
        target = Path(output_dir) / Path(filename).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return {
            "job_id": job_id,
            "artifact_id": artifact_id,
            "filename": target.name,
            "path": str(target),
            "size_bytes": len(data),
        }

    def download_all_artifacts(self, job_id: str, output_dir: str | Path = ".") -> dict[str, Any]:
        artifacts_payload = self.job_artifacts(job_id)
        downloads = []
        for artifact in artifacts_payload.get("artifacts", []):
            artifact_id = str(artifact.get("id") or "")
            if artifact_id:
                downloads.append(self.download_artifact(job_id, artifact_id, output_dir=output_dir))
        return {"job_id": job_id, "downloaded": downloads}
