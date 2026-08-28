#!/usr/bin/env python3
"""End-to-end checks for an OpenCode client talking to a fastmlx server."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import SplitResult, urlsplit, urlunsplit
from urllib.request import Request, urlopen


CASE_ORDER = (
    "multi_turn",
    "concurrency",
    "tool_recovery",
    "large_file",
    "sigint_recovery",
    "reasoning_variant",
)
CASE_ALIASES = {
    "all": "all",
    "multi_turn": "multi_turn",
    "multi-turn": "multi_turn",
    "multi": "multi_turn",
    "concurrency": "concurrency",
    "concurrent": "concurrency",
    "tool_recovery": "tool_recovery",
    "tool-recovery": "tool_recovery",
    "tool_failure": "tool_recovery",
    "large_file": "large_file",
    "large-file": "large_file",
    "file_read": "large_file",
    "sigint_recovery": "sigint_recovery",
    "sigint-recovery": "sigint_recovery",
    "interrupt": "sigint_recovery",
    "reasoning_variant": "reasoning_variant",
    "reasoning-variant": "reasoning_variant",
    "reasoning": "reasoning_variant",
}

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class RunResult:
    returncode: int | None
    stdout: str
    stderr: str
    events: list[dict[str, Any]] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    timed_out: bool = False
    signal_sent: bool = False
    elapsed_s: float = 0.0
    active_event_seen: bool = False

    @property
    def session_id(self) -> str | None:
        for event in self.events:
            value = event.get("sessionID")
            if isinstance(value, str) and value:
                return value
        return None

    @property
    def text(self) -> str:
        parts: list[str] = []
        for event in self.events:
            if event.get("type") != "text":
                continue
            part = event.get("part")
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(event.get("text"), str):
                parts.append(event["text"])
        return "\n".join(parts)


@dataclass
class CaseResult:
    name: str
    status: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightResult:
    status: str
    summary: str
    health_status: int | None = None
    models_status: int | None = None
    model_ids: list[str] = field(default_factory=list)


def _sentinel(case: str, ordinal: int = 0) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", case).strip("_").upper()
    digest = hashlib.sha256(f"fastmlx-opencode-e2e:{case}:{ordinal}".encode()).hexdigest()[:12].upper()
    return f"FASTMLX_E2E_{slug}_{digest}"


def _parse_ndjson(stdout: str) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(stdout.splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"stdout line {line_number} is not valid JSON")
            continue
        if not isinstance(item, dict):
            errors.append(f"stdout line {line_number} is not a JSON object")
            continue
        events.append(item)
    return events, errors


def _event_has_reasoning(event: dict[str, Any]) -> bool:
    if event.get("type") == "reasoning":
        return True
    part = event.get("part")
    return isinstance(part, dict) and part.get("type") == "reasoning"


def _tool_states(events: Iterable[dict[str, Any]], tool_name: str | None = None) -> list[str]:
    states: list[str] = []
    for event in events:
        if event.get("type") != "tool_use":
            continue
        part = event.get("part")
        if not isinstance(part, dict) or part.get("type") != "tool":
            continue
        if tool_name is not None and part.get("tool") not in {None, tool_name}:
            continue
        state = part.get("state")
        if isinstance(state, dict) and isinstance(state.get("status"), str):
            states.append(state["status"])
    return states


def _tool_outputs(events: Iterable[dict[str, Any]], tool_name: str | None = None) -> list[str]:
    outputs: list[str] = []
    for event in events:
        if event.get("type") != "tool_use":
            continue
        part = event.get("part")
        if not isinstance(part, dict):
            continue
        if tool_name is not None and part.get("tool") not in {None, tool_name}:
            continue
        state = part.get("state")
        if isinstance(state, dict) and isinstance(state.get("output"), str):
            outputs.append(state["output"])
    return outputs


def _status_for_run(result: RunResult, sentinels: Iterable[str]) -> tuple[str, str]:
    if result.timed_out:
        return FAIL, "opencode timed out"
    if result.returncode != 0:
        return FAIL, f"opencode exited with status {result.returncode}"
    if result.parse_errors:
        return FAIL, "; ".join(result.parse_errors)
    if not result.events:
        return FAIL, "opencode produced no JSON events"
    missing = [value for value in sentinels if value not in result.text]
    if missing:
        return FAIL, "response text did not contain the required sentinel"
    if result.session_id is None:
        return FAIL, "JSON events did not contain a sessionID"
    return PASS, "response and event structure matched"


def _split_base_url(base_url: str) -> tuple[str, str]:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base URL must be an http(s) URL with a host")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        root_path = path[:-3].rstrip("/")
        api_path = path
    else:
        root_path = path
        api_path = f"{path}/v1" if path else "/v1"
    root = urlunsplit(SplitResult(parsed.scheme, parsed.netloc, root_path, "", ""))
    api = urlunsplit(SplitResult(parsed.scheme, parsed.netloc, api_path, "", ""))
    return f"{root}/health", f"{api}/models"


def _http_json(url: str, timeout: float) -> tuple[int, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
        if not body:
            return response.status, None
        return response.status, json.loads(body.decode("utf-8"))


def preflight(base_url: str, expected_model: str, timeout: float) -> PreflightResult:
    try:
        health_url, models_url = _split_base_url(base_url)
    except ValueError as exc:
        return PreflightResult(FAIL, str(exc))

    try:
        health_status, health = _http_json(health_url, timeout)
    except HTTPError as exc:
        return PreflightResult(INCONCLUSIVE, f"health endpoint returned HTTP {exc.code}; check server readiness")
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return PreflightResult(
            INCONCLUSIVE,
            f"health endpoint was unreachable ({type(exc).__name__}: {exc}); check the server and --base-url",
        )

    if health_status < 200 or health_status >= 300:
        return PreflightResult(INCONCLUSIVE, f"health endpoint returned HTTP {health_status}", health_status)
    if not isinstance(health, dict) or health.get("status") not in {"ok", "ready", "loaded"}:
        return PreflightResult(FAIL, "health endpoint did not report a ready status", health_status)

    try:
        models_status, payload = _http_json(models_url, timeout)
    except HTTPError as exc:
        return PreflightResult(FAIL, f"models endpoint returned HTTP {exc.code}; check server routing", health_status)
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return PreflightResult(
            INCONCLUSIVE,
            f"models endpoint was unreachable ({type(exc).__name__}: {exc}); check server routing and --base-url",
            health_status,
        )

    if models_status < 200 or models_status >= 300:
        return PreflightResult(FAIL, f"models endpoint returned HTTP {models_status}", health_status, models_status)
    rows = payload.get("data") if isinstance(payload, dict) else None
    model_ids = [row.get("id") for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)] if isinstance(rows, list) else []
    expected_ids = {expected_model}
    if "/" in expected_model:
        expected_ids.add(expected_model.rsplit("/", 1)[1])
    if not expected_ids.intersection(model_ids):
        return PreflightResult(
            FAIL,
            f"models endpoint did not advertise expected model id {expected_model!r}",
            health_status,
            models_status,
            model_ids,
        )
    return PreflightResult(PASS, "health and expected model preflight passed", health_status, models_status, model_ids)


def _isolated_env(config_path: Path, runtime_root: Path, config_content: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["OPENCODE_CONFIG"] = str(config_path)
    env.pop("OPENCODE_CONFIG_CONTENT", None)
    env["XDG_CONFIG_HOME"] = str(runtime_root / "config")
    env["XDG_DATA_HOME"] = str(runtime_root / "data")
    env["XDG_CACHE_HOME"] = str(runtime_root / "cache")
    if config_content is not None:
        env["OPENCODE_CONFIG_CONTENT"] = config_content
    return env


def _command(
    opencode: str,
    model: str,
    prompt: str,
    session_id: str | None = None,
    variant: str | None = None,
    thinking: bool = False,
) -> list[str]:
    command = [opencode, "run", "--pure", "--log-level", "ERROR", "--format", "json", "--model", model]
    if session_id:
        command.extend(["--session", session_id])
    if variant:
        command.extend(["--variant", variant])
    if thinking:
        command.append("--thinking")
    command.append(prompt)
    return command


def _spawn(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )


def _signal_process_tree(process: subprocess.Popen[str], value: signal.Signals) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, value)
            return
        except (ProcessLookupError, PermissionError):
            return
    process.send_signal(value)


def _finish_process(process: subprocess.Popen[str], timeout: float) -> tuple[str, str, bool]:
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return stdout, stderr, False
    except subprocess.TimeoutExpired:
        _signal_process_tree(process, signal.SIGINT)
        try:
            stdout, stderr = process.communicate(timeout=min(5.0, max(1.0, timeout / 4)))
            return stdout, stderr, True
        except subprocess.TimeoutExpired:
            _signal_process_tree(process, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=min(5.0, max(1.0, timeout / 4)))
                return stdout, stderr, True
            except subprocess.TimeoutExpired:
                _signal_process_tree(process, signal.SIGKILL)
                stdout, stderr = process.communicate()
                return stdout, stderr, True


def run_opencode(
    *,
    opencode: str,
    model: str,
    prompt: str,
    cwd: Path,
    config_path: Path,
    runtime_root: Path,
    timeout: float,
    session_id: str | None = None,
    variant: str | None = None,
    thinking: bool = False,
    config_content: str | None = None,
) -> RunResult:
    command = _command(opencode, model, prompt, session_id, variant, thinking)
    try:
        process = _spawn(command, cwd, _isolated_env(config_path, runtime_root, config_content))
    except OSError:
        return RunResult(None, "", "", parse_errors=["could not start the opencode executable"])
    started = time.monotonic()
    stdout, stderr, timed_out = _finish_process(process, timeout)
    events, parse_errors = _parse_ndjson(stdout)
    return RunResult(process.returncode, stdout, stderr, events, parse_errors, timed_out, False, time.monotonic() - started)


def run_opencode_interrupt(
    *,
    opencode: str,
    model: str,
    prompt: str,
    cwd: Path,
    config_path: Path,
    runtime_root: Path,
    timeout: float,
    interrupt_after: float,
    interrupt_grace: float,
) -> RunResult:
    command = _command(opencode, model, prompt)
    try:
        process = _spawn(command, cwd, _isolated_env(config_path, runtime_root))
    except OSError:
        return RunResult(None, "", "", parse_errors=["could not start the opencode executable"])
    started = time.monotonic()
    deadline = started + min(interrupt_after, timeout)
    prefix: list[str] = []
    active_event_seen = False
    selector = selectors.DefaultSelector()
    if process.stdout is not None:
        selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while process.poll() is None and time.monotonic() < deadline and not active_event_seen:
            remaining = max(0.0, deadline - time.monotonic())
            for key, _ in selector.select(min(0.05, remaining)):
                line = key.fileobj.readline()
                if not line:
                    continue
                prefix.append(line)
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type") if isinstance(event, dict) else None
                part = event.get("part") if isinstance(event, dict) else None
                part_type = part.get("type") if isinstance(part, dict) else None
                active_event_seen = event_type in {"step_start", "reasoning", "tool_use", "text"} or part_type in {
                    "step-start",
                    "reasoning",
                    "tool",
                    "text",
                }
                if active_event_seen:
                    break
    finally:
        selector.close()

    signal_sent = False
    if process.poll() is None:
        _signal_process_tree(process, signal.SIGINT)
        signal_sent = True
    try:
        stdout, stderr = process.communicate(timeout=interrupt_grace)
        timed_out = False
    except subprocess.TimeoutExpired:
        _signal_process_tree(process, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=min(5.0, max(1.0, interrupt_grace)))
            timed_out = True
        except subprocess.TimeoutExpired:
            _signal_process_tree(process, signal.SIGKILL)
            stdout, stderr = process.communicate()
            timed_out = True
    stdout = "".join(prefix) + stdout
    events, parse_errors = _parse_ndjson(stdout)
    return RunResult(
        process.returncode,
        stdout,
        stderr,
        events,
        parse_errors,
        timed_out,
        signal_sent,
        time.monotonic() - started,
        active_event_seen,
    )


def _variant_config(config: dict[str, Any], model: str) -> dict[str, Any]:
    result = copy.deepcopy(config)
    providers = result.setdefault("provider", {})
    if not isinstance(providers, dict):
        providers = {}
        result["provider"] = providers

    provider_id, separator, model_id = model.partition("/")
    if not separator:
        model_id = provider_id
        provider_id = next(iter(providers), "fastmlx")
        for candidate_id, candidate in providers.items():
            if isinstance(candidate, dict) and isinstance(candidate.get("models"), dict) and model_id in candidate["models"]:
                provider_id = candidate_id
                break
    provider = providers.setdefault(provider_id, {})
    if not isinstance(provider, dict):
        provider = {}
        providers[provider_id] = provider
    models = provider.setdefault("models", {})
    if not isinstance(models, dict):
        models = {}
        provider["models"] = models
    model_config = models.setdefault(model_id, {})
    if not isinstance(model_config, dict):
        model_config = {}
        models[model_id] = model_config
    model_config["reasoning"] = True
    variants = model_config.setdefault("variants", {})
    if not isinstance(variants, dict):
        variants = {}
        model_config["variants"] = variants
    high = variants.setdefault("high", {})
    if not isinstance(high, dict):
        high = {}
        variants["high"] = high
    high["reasoningEffort"] = "high"
    return result


def _result(name: str, status: str, summary: str, **details: Any) -> CaseResult:
    return CaseResult(name, status, summary, details)


class Harness:
    def __init__(self, args: argparse.Namespace, config: dict[str, Any], config_path: Path):
        self.args = args
        self.config = config
        self.config_path = config_path

    def run(self, selected: list[str]) -> Iterable[CaseResult]:
        with tempfile.TemporaryDirectory(prefix="fastmlx-opencode-e2e-") as temp_name:
            root = Path(temp_name)
            for name in selected:
                try:
                    if name == "multi_turn":
                        result = self.multi_turn(root)
                    elif name == "concurrency":
                        result = self.concurrency(root)
                    elif name == "tool_recovery":
                        result = self.tool_recovery(root)
                    elif name == "large_file":
                        result = self.large_file(root)
                    elif name == "sigint_recovery":
                        result = self.sigint_recovery(root)
                    elif name == "reasoning_variant":
                        result = self.reasoning_variant(root)
                    else:
                        result = _result(name, FAIL, "unknown case")
                except Exception as exc:
                    result = _result(name, FAIL, f"case raised {type(exc).__name__}")
                yield result

    def _run(self, prompt: str, cwd: Path, runtime_name: str, **kwargs: Any) -> RunResult:
        runtime = cwd / runtime_name
        runtime.mkdir(parents=True, exist_ok=True)
        return run_opencode(
            opencode=self.args.opencode,
            model=self.args.model,
            prompt=prompt,
            cwd=cwd,
            config_path=self.config_path,
            runtime_root=runtime,
            timeout=self.args.timeout,
            **kwargs,
        )

    def multi_turn(self, root: Path) -> CaseResult:
        sentinels = [_sentinel("multi_turn", index) for index in range(self.args.conversation_turns)]
        first = sentinels[0]
        cwd = root / "multi-turn"
        cwd.mkdir()
        first_result = self._run(
            f"Remember this exact sentinel for the rest of the conversation and include it in a brief reply: {first}. Do not use tools.",
            cwd,
            "runtime",
        )
        status, summary = _status_for_run(first_result, [first])
        if status != PASS:
            return _result("multi_turn", status, f"first turn: {summary}")
        session_id = first_result.session_id
        for turn, current in enumerate(sentinels[1:], 2):
            continued = self._run(
                f"This is turn {turn}. Recall the sentinel from turn 1, then include it and this new exact sentinel in a brief reply: {current}. Do not use tools.",
                cwd,
                "runtime",
                session_id=session_id,
            )
            status, summary = _status_for_run(continued, [first, current])
            if status != PASS:
                return _result("multi_turn", status, f"turn {turn}: {summary}", session_id=session_id)
            if continued.session_id != session_id:
                return _result(
                    "multi_turn",
                    FAIL,
                    f"turn {turn} returned a different sessionID",
                    session_id=session_id,
                )
        return _result(
            "multi_turn",
            PASS,
            f"{self.args.conversation_turns} turns retained the first-turn sentinel",
            session_id=session_id,
            turns=self.args.conversation_turns,
        )

    def concurrency(self, root: Path) -> CaseResult:
        prompts: list[tuple[str, Path, str]] = []
        for ordinal in range(2):
            sentinel = _sentinel("concurrency", ordinal)
            cwd = root / f"concurrent-{ordinal}"
            cwd.mkdir()
            prompts.append((f"Reply briefly with this exact sentinel and no other sentinel: {sentinel}. Do not use tools.", cwd, sentinel))

        def worker(item: tuple[str, Path, str]) -> RunResult:
            prompt, cwd, _ = item
            return self._run(prompt, cwd, "runtime")

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        futures = [executor.submit(worker, item) for item in prompts]
        try:
            results = [future.result(timeout=self.args.timeout + 10.0) for future in futures]
        except concurrent.futures.TimeoutError:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            return _result("concurrency", FAIL, "concurrent requests did not complete before the deadlock guard")
        except Exception:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True, cancel_futures=True)

        session_ids = [result.session_id for result in results]
        for ordinal, (result, (_, _, own)) in enumerate(zip(results, prompts)):
            other = prompts[1 - ordinal][2]
            status, summary = _status_for_run(result, [own])
            if status != PASS:
                return _result("concurrency", status, f"request {ordinal}: {summary}")
            if other in result.text:
                return _result("concurrency", FAIL, f"request {ordinal} contained the other request sentinel")
        if session_ids[0] == session_ids[1]:
            return _result("concurrency", FAIL, "concurrent requests unexpectedly shared a sessionID")
        return _result("concurrency", PASS, "both requests completed with isolated sentinel text")

    def tool_recovery(self, root: Path) -> CaseResult:
        missing = root / "tool-recovery" / "missing-e2e-file.txt"
        missing.parent.mkdir()
        sentinel = _sentinel("tool_recovery")
        result = self._run(
            f"Use the built-in read tool on this exact nonexistent path: {missing}. Do not create it or retry with another tool. After the tool error, recover with one brief text reply containing {sentinel}.",
            missing.parent,
            "runtime",
        )
        status, summary = _status_for_run(result, [sentinel])
        states = _tool_states(result.events, "read")
        if status == PASS and not any(state == "error" for state in states):
            status, summary = FAIL, "no completed tool error event was observed"
        if missing.exists():
            status, summary = FAIL, "tool failure fixture was unexpectedly created"
        return _result("tool_recovery", status, summary, tool_states=states)

    def large_file(self, root: Path) -> CaseResult:
        fixture_dir = root / "large-file"
        fixture_dir.mkdir()
        path = fixture_dir / "over-100k.txt"
        sentinel = _sentinel("large_file")
        lines = [sentinel]
        lines.extend(f"filler-{index:05d}-0123456789abcdef0123456789abcdef" for index in range(5000))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        size = path.stat().st_size
        result = self._run(
            f"Use the built-in read tool on {path}. It is intentionally larger than 100 KiB and may be truncated by the tool. Do not modify it. After reading, reply briefly with the exact sentinel {sentinel}.",
            fixture_dir,
            "runtime",
        )
        status, summary = _status_for_run(result, [sentinel])
        states = _tool_states(result.events, "read")
        outputs = _tool_outputs(result.events, "read")
        read_events = [
            event
            for event in result.events
            if event.get("type") == "tool_use"
            and isinstance(event.get("part"), dict)
            and event["part"].get("tool") in {None, "read"}
        ]
        serialised = json.dumps(read_events, ensure_ascii=False).lower()
        truncated = any(marker in serialised for marker in ("truncat", "output truncated", "content clipped"))
        if outputs and any(len(output.encode("utf-8")) < size for output in outputs):
            truncated = True
        if status == PASS and not any(state == "completed" for state in states):
            status, summary = FAIL, "no completed file-read tool event was observed"
        elif status == PASS and not truncated:
            status, summary = FAIL, "the over-100KiB read showed no truncation evidence"
        return _result("large_file", status, summary, fixture_bytes=size, tool_states=states, truncation_observed=truncated)

    def sigint_recovery(self, root: Path) -> CaseResult:
        cwd = root / "sigint"
        cwd.mkdir()
        interrupted_sentinel = _sentinel("sigint_recovery", 0)
        recovery_sentinel = _sentinel("sigint_recovery", 1)
        interrupted = run_opencode_interrupt(
            opencode=self.args.opencode,
            model=self.args.model,
            prompt=(
                "Generate a deliberately very long response without tools and do not finish quickly. "
                f"Only after completing it would you include {interrupted_sentinel}."
            ),
            cwd=cwd,
            config_path=self.config_path,
            runtime_root=cwd / "interrupted-runtime",
            timeout=self.args.timeout,
            interrupt_after=self.args.sigint_after,
            interrupt_grace=self.args.interrupt_grace,
        )
        recovery = self._run(
            f"The prior request was interrupted. Start a fresh successful response and include this exact sentinel: {recovery_sentinel}. Do not use tools.",
            cwd,
            "recovery-runtime",
        )
        recovery_status, recovery_summary = _status_for_run(recovery, [recovery_sentinel])
        if recovery_status != PASS:
            return _result("sigint_recovery", FAIL, f"recovery after SIGINT: {recovery_summary}", signal_sent=interrupted.signal_sent)
        if not interrupted.active_event_seen:
            return _result(
                "sigint_recovery",
                INCONCLUSIVE,
                "no active OpenCode event was observed before the interrupt deadline",
                signal_sent=interrupted.signal_sent,
            )
        if not interrupted.signal_sent:
            return _result("sigint_recovery", INCONCLUSIVE, "request finished before SIGINT could be delivered", signal_sent=False)
        if interrupted.timed_out:
            return _result("sigint_recovery", FAIL, "process did not terminate during SIGINT grace period", signal_sent=True)
        return _result("sigint_recovery", PASS, "SIGINT was delivered and a subsequent request recovered", signal_sent=True)

    def reasoning_variant(self, root: Path) -> CaseResult:
        if not self.args.reasoning_variant:
            return _result("reasoning_variant", INCONCLUSIVE, "opt in with --reasoning-variant to test config and display behavior")
        cwd = root / "reasoning-variant"
        cwd.mkdir()
        sentinel = _sentinel("reasoning_variant")
        override = json.dumps(_variant_config(self.config, self.args.model), separators=(",", ":"))
        result = self._run(
            f"Use the configured high reasoning variant, then answer briefly with this exact sentinel: {sentinel}.",
            cwd,
            "runtime",
            variant="high",
            thinking=True,
            config_content=override,
        )
        status, summary = _status_for_run(result, [sentinel])
        reasoning_events = sum(_event_has_reasoning(event) for event in result.events)
        if status == PASS and reasoning_events == 0:
            status, summary = FAIL, "--thinking produced no reasoning event for the opted-in reasoning variant"
        return _result("reasoning_variant", status, summary, reasoning_events=reasoning_events, variant="high", config_override=True)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("config must contain a JSON object")
    return value


def _parse_cases(values: list[str] | None) -> list[str]:
    if not values:
        return list(CASE_ORDER)
    selected: list[str] = []
    for value in values:
        for raw in value.split(","):
            name = raw.strip().lower()
            canonical = CASE_ALIASES.get(name)
            if canonical is None:
                choices = ", ".join(CASE_ORDER)
                raise ValueError(f"unknown case {raw!r}; choose from {choices}")
            if canonical == "all":
                selected = list(CASE_ORDER)
                continue
            if canonical not in selected:
                selected.append(canonical)
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenCode JSON-event E2E checks against a running OpenAI-compatible server.",
        epilog=(
            "The server is never started by this harness. Each case uses a temporary working directory; "
            "reasoning-variant is INCONCLUSIVE unless --reasoning-variant is supplied."
        ),
    )
    parser.add_argument("--config", required=True, type=Path, help="path to the opencode.json used by OpenCode")
    parser.add_argument("--model", required=True, help="OpenCode model id (for example provider/model)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765/v1", help="server API base URL (default: %(default)s)")
    parser.add_argument("--opencode", default="opencode", help="OpenCode executable (default: %(default)s)")
    parser.add_argument("--case", "--cases", dest="cases", action="append", metavar="NAME[,NAME...]", help=f"case selection; repeatable (default: all: {', '.join(CASE_ORDER)})")
    parser.add_argument("--timeout", type=float, default=90.0, help="per-request timeout in seconds (default: %(default)s)")
    parser.add_argument("--conversation-turns", type=int, default=6, help="turn count for the long-conversation case; at least 2 (default: %(default)s)")
    parser.add_argument("--preflight-timeout", type=float, default=5.0, help="HTTP preflight timeout in seconds (default: %(default)s)")
    parser.add_argument("--sigint-after", type=float, default=1.0, help="maximum seconds to wait for an active JSON event before SIGINT (default: %(default)s)")
    parser.add_argument("--interrupt-grace", type=float, default=5.0, help="seconds to wait after SIGINT (default: %(default)s)")
    parser.add_argument("--reasoning-variant", "--enable-reasoning-variant", action="store_true", help="opt in to an ephemeral reasoning=true/high reasoningEffort config override")
    parser.add_argument("--dry-run", action="store_true", help="run config validation and HTTP preflight only")
    parser.add_argument("--json-summary", "--json", action="store_true", help="emit one machine-readable JSON summary")
    return parser


def _summary_text(summary: dict[str, Any]) -> str:
    lines = [f"status: {summary['status']}", f"preflight: {summary['preflight']['status']} ({summary['preflight']['summary']})"]
    for case in summary.get("cases", []):
        lines.append(f"{case['status']:13} {case['name']}: {case['summary']}")
    if summary.get("dry_run"):
        lines.append("dry-run: no OpenCode cases executed")
    return "\n".join(lines)


def _safe_base_url(base_url: str) -> str:
    try:
        parsed = urlsplit(base_url)
    except ValueError:
        return "<invalid>"
    if not parsed.scheme or not parsed.hostname:
        return "<invalid>"
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit(SplitResult(parsed.scheme, netloc, parsed.path, "", ""))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    for name in ("timeout", "preflight_timeout", "sigint_after", "interrupt_grace"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.conversation_turns < 2:
        parser.error("--conversation-turns must be at least 2")

    config_path = args.config.expanduser().absolute()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": FAIL,
        "config": str(config_path),
        "model": args.model,
        "base_url": _safe_base_url(args.base_url),
        "dry_run": args.dry_run,
        "cases": [],
    }
    try:
        config = _load_config(config_path)
        selected = _parse_cases(args.cases)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        summary["preflight"] = asdict(PreflightResult(FAIL, f"input validation failed: {exc}"))
        summary["status"] = FAIL
        print(json.dumps(summary, ensure_ascii=False) if args.json_summary else _summary_text(summary))
        return 1

    pf = preflight(args.base_url, args.model, args.preflight_timeout)
    summary["preflight"] = asdict(pf)
    if pf.status != PASS:
        summary["cases"] = [asdict(_result(name, INCONCLUSIVE, "not run because preflight did not pass")) for name in selected]
        summary["status"] = pf.status
        print(json.dumps(summary, ensure_ascii=False) if args.json_summary else _summary_text(summary))
        return 1 if pf.status == FAIL else 2

    if args.dry_run:
        summary["status"] = PASS
        print(json.dumps(summary, ensure_ascii=False) if args.json_summary else _summary_text(summary))
        return 0

    results = list(Harness(args, config, config_path).run(selected))
    summary["cases"] = [asdict(result) for result in results]
    statuses = {result.status for result in results}
    summary["status"] = FAIL if FAIL in statuses else INCONCLUSIVE if INCONCLUSIVE in statuses else PASS
    print(json.dumps(summary, ensure_ascii=False) if args.json_summary else _summary_text(summary))
    return 1 if summary["status"] == FAIL else 2 if summary["status"] == INCONCLUSIVE else 0


if __name__ == "__main__":
    raise SystemExit(main())
