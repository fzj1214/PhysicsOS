from __future__ import annotations

import argparse
import json
from pathlib import Path

from physicsos.cloud.auth import start_device_login
from physicsos.cloud.foamvm_client import FoamVMClient


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="physicsos")
    sub = parser.add_subparsers(dest="command", required=True)

    auth = sub.add_parser("auth")
    auth_sub = auth.add_subparsers(dest="auth_command", required=True)
    login = auth_sub.add_parser("login")
    login.add_argument("--runner-url", default="https://foamvm.vercel.app")
    login.add_argument("--no-browser", action="store_true")

    sub.add_parser("account")

    runner = sub.add_parser("runner")
    runner_sub = runner.add_subparsers(dest="runner_command", required=True)
    submit = runner_sub.add_parser("submit")
    submit.add_argument("manifest")
    status = runner_sub.add_parser("status")
    status.add_argument("job_id")
    logs = runner_sub.add_parser("logs")
    logs.add_argument("job_id")
    logs.add_argument("--after", type=int)
    artifacts = runner_sub.add_parser("artifacts")
    artifacts.add_argument("job_id")

    args = parser.parse_args(argv)

    if args.command == "auth" and args.auth_command == "login":
        result = start_device_login(args.runner_url, open_browser=not args.no_browser)
        _print_json({"runner_url": result.runner_url, "user_code": result.user_code, "status": "logged_in"})
        return 0

    if args.command == "account":
        _print_json(FoamVMClient.from_config().me())
        return 0

    if args.command == "runner":
        client = FoamVMClient.from_config()
        if args.runner_command == "submit":
            manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            _print_json(client.submit_job(manifest))
            return 0
        if args.runner_command == "status":
            _print_json(client.job_status(args.job_id))
            return 0
        if args.runner_command == "logs":
            _print_json(client.job_events(args.job_id, after=args.after))
            return 0
        if args.runner_command == "artifacts":
            _print_json(client.job_artifacts(args.job_id))
            return 0

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
