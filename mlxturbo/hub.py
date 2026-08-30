"""mlxturbo hub: model discovery/search/download for the GUI companion app.

This exists so a separate SwiftUI menu-bar app never has to re-implement
Hugging Face Hub protocol details (resumable downloads, chunked transfer,
auth token lookup) in Swift. Every subcommand here talks to
``huggingface_hub`` instead and prints machine-readable JSON to stdout --
there is no human-oriented output, no progress bars, nothing meant to be
read by a person in a terminal.

v1 scope: discovery, search, download only. Swapping the model a running
server has loaded is out of scope here -- that still means restarting the
process with a different --model (see mlxturbo/server.py / cli.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_MODELS_DIR = os.path.expanduser("~/models")

# tqdm's own per-file update calls happen many times a second on a fast local
# link; this caps how often we actually write a JSON line so the GUI isn't
# flooded with near-duplicate progress events.
_PROGRESS_MIN_INTERVAL_S = 0.2


# ---------- local model discovery (shared by `list` and `search --already-local`) ----------


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _find_context_length(config: dict):
    """config.json (or its nested text_config, for the vision-capable
    architectures in this fleet) sometimes uses different key names for the
    same idea depending on the exporter. Try the common ones before giving
    up."""
    for key in (
        "max_position_embeddings",
        "context_length",
        "max_seq_len",
        "n_positions",
        "seq_length",
    ):
        value = config.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return _find_context_length(text_config)
    return None


def _has_mtp(model_dir: Path, index: dict | None) -> bool:
    if (model_dir / "mtp.safetensors").exists():
        return True
    if index:
        weight_map = index.get("weight_map", {})
        return any(k.startswith("mtp.") for k in weight_map)
    return False


def _read_local_model(model_dir: Path):
    """Returns the GUI-facing record for one model directory, or None if it
    doesn't look like a model directory at all (no readable config.json --
    silently skipped per spec, since --dir may contain unrelated folders)."""
    config_path = model_dir / "config.json"
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    index = None
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            with open(index_path) as f:
                index = json.load(f)
        except (OSError, json.JSONDecodeError):
            index = None

    return {
        "path": str(model_dir),
        "name": model_dir.name,
        "model_type": config.get("model_type"),
        "size_bytes": _dir_size_bytes(model_dir),
        "quantization": config.get("quantization"),
        "has_mtp": _has_mtp(model_dir, index),
        "context_length": _find_context_length(config),
    }


def _hf_cache_snapshot_dirs():
    """Yields (repo_id, snapshot_path) for every cached model repo under
    huggingface_hub's default cache location."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return
    try:
        cache_info = scan_cache_dir()
    except Exception:
        return
    for repo in cache_info.repos:
        if repo.repo_type != "model":
            continue
        for revision in repo.revisions:
            yield repo.repo_id, Path(revision.snapshot_path)


def _scan_local_models(base_dir: Path):
    """--dir's direct subdirectories, plus the huggingface_hub cache,
    deduplicated by resolved path. Each entry from the cache also carries
    the real repo_id it came from (used by `search` for already_local)."""
    seen = set()
    models = []
    repo_ids_by_path = {}

    if base_dir.exists():
        for child in sorted(base_dir.iterdir()):
            if not child.is_dir():
                continue
            resolved = child.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            info = _read_local_model(child)
            if info is not None:
                models.append(info)

    for repo_id, snapshot_dir in _hf_cache_snapshot_dirs():
        resolved = snapshot_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        info = _read_local_model(snapshot_dir)
        if info is not None:
            repo_ids_by_path[info["path"]] = repo_id
            models.append(info)

    return models, repo_ids_by_path


# ---------- subcommands ----------


def cmd_list(args) -> None:
    models, _repo_ids = _scan_local_models(Path(args.dir).expanduser())
    print(json.dumps(models))


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def cmd_search(args) -> None:
    from huggingface_hub import HfApi

    local_models, repo_ids_by_path = _scan_local_models(Path(args.dir).expanduser())
    local_cache_repo_ids = set(repo_ids_by_path.values())
    # Directories under --dir carry no provenance back to an HF repo_id (this
    # fleet's own models under ~/models are internally renamed/baked
    # artifacts -- see docs/BAKE-PLAN.md -- and generally don't share a name
    # with any single upstream repo). Best-effort only: match the search
    # result's last path segment against a local directory name.
    local_normalized_names = {_normalize(m["name"]) for m in local_models}

    api = HfApi()
    try:
        # `expand` replaces the API's default field set rather than adding to
        # it -- passing only ["lastModified"] here would silently drop
        # downloads/likes from the response (found the hard way).
        found = list(
            api.list_models(
                search=args.query,
                filter="mlx",
                limit=args.limit,
                expand=["downloads", "likes", "lastModified"],
            )
        )
    except Exception as exc:
        print(json.dumps({"event": "error", "message": str(exc)}))
        sys.exit(1)

    results = []
    for model in found:
        size_bytes = None
        try:
            info = api.model_info(model.id, files_metadata=True)
            sizes = [s.size for s in info.siblings if s.size]
            if sizes:
                size_bytes = sum(sizes)
        except Exception:
            size_bytes = None

        already_local = (
            model.id in local_cache_repo_ids
            or _normalize(model.id.split("/")[-1]) in local_normalized_names
        )
        last_modified = getattr(model, "last_modified", None)
        results.append(
            {
                "id": model.id,
                "downloads": model.downloads,
                "likes": model.likes,
                "last_modified": last_modified.isoformat() if last_modified else None,
                "size_bytes": size_bytes,
                "already_local": already_local,
            }
        )

    print(json.dumps(results))


def _make_progress_emitter(shared: dict):
    """Returns a fresh tqdm-compatible class bound to `shared` (a plain dict
    used as mutable state across every per-file instance huggingface_hub
    creates -- see cmd_download). Implements only what
    huggingface_hub/file_download.py's http_get() actually drives: the
    constructor kwargs it passes, .update(n), the context-manager protocol,
    and .close().
    """

    class _ProgressEmitter:
        def __init__(self, *, desc=None, total=None, initial=0, **_ignored):
            self.filename = desc
            shared["done"] += initial or 0
            self._emit()

        def update(self, n=1):
            if not n:
                return
            shared["done"] += n
            now = time.monotonic()
            if now - shared["last_emit"] >= _PROGRESS_MIN_INTERVAL_S:
                shared["last_emit"] = now
                self._emit()

        def close(self):
            self._emit()

        def update_transfer(self, n=1):
            # Deliberately a no-op, but its mere presence changes behavior:
            # when hf_xet is installed, huggingface_hub's xet_get() checks
            # `callable(getattr(tqdm_class, "update_transfer", None))` to
            # decide whether a tqdm_class already aggregates a file's
            # "network transfer" and "disk reconstruction" progress into one
            # bar. Without this method it creates two separate instances of
            # this class per file -- one for each -- and both call .update(),
            # which silently double-counted every byte here (caught by
            # comparing done_bytes to total_bytes on a real download).
            # With it, only one instance is created and network-transfer
            # amounts are routed through this
            # no-op instead of .update(), leaving .update() to report only
            # reconstruction (actual on-disk) bytes -- the right measure of
            # "how much of this file is done" even when xet's dedup means
            # fewer bytes crossed the network than the file's final size.
            pass

        def _emit(self):
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "done_bytes": shared["done"],
                        "total_bytes": shared["total"],
                        "file": self.filename,
                    }
                ),
                flush=True,
            )

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self.close()
            return False

        # Called by http_get when it has a filename/rate to report; both are
        # already covered by our own _emit(), so these are no-ops rather than
        # AttributeErrors.
        def set_postfix_str(self, *_a, **_k):
            pass

        def set_transfer_postfix_str(self, *_a, **_k):
            pass

        def set_description(self, *_a, **_k):
            pass

        def refresh(self, *_a, **_k):
            pass

    return _ProgressEmitter


def cmd_download(args) -> None:
    # No SIGTERM handler is installed on purpose: the GUI stops a download by
    # killing this process, and the default disposition for SIGTERM already
    # terminates it immediately. Whatever file was mid-write is left exactly
    # as huggingface_hub's own ``.incomplete`` + Range-resume mechanism
    # expects to find it on the next run of this same command -- there is
    # nothing for us to clean up without working against that mechanism.
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    try:
        info = api.model_info(args.repo_id, files_metadata=True)
    except Exception as exc:
        print(json.dumps({"event": "error", "message": str(exc)}), flush=True)
        sys.exit(1)

    filenames = [s.rfilename for s in info.siblings]
    total_bytes = sum(s.size for s in info.siblings if s.size)

    target_name = args.name or args.repo_id.split("/")[-1]
    target_dir = Path(args.dir).expanduser() / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    shared = {"done": 0, "total": total_bytes, "last_emit": 0.0}
    progress_cls = _make_progress_emitter(shared)

    try:
        for filename in filenames:
            hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                revision=info.sha,
                local_dir=str(target_dir),
                tqdm_class=progress_cls,
            )
    except Exception as exc:
        print(json.dumps({"event": "error", "message": str(exc)}), flush=True)
        sys.exit(1)

    print(json.dumps({"event": "done", "path": str(target_dir)}), flush=True)


# ---------- argparse wiring ----------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mlxturbo hub")
    sub = ap.add_subparsers(dest="hub_command", required=True)

    p_list = sub.add_parser("list", help="ローカルにあるモデルを列挙する")
    p_list.add_argument("--dir", default=DEFAULT_MODELS_DIR)
    p_list.set_defaults(func=cmd_list)

    p_search = sub.add_parser("search", help="Hugging Face Hub を検索する")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=20)
    p_search.add_argument(
        "--dir",
        default=DEFAULT_MODELS_DIR,
        help="already_local 判定に使うローカル走査先 (list と同じ既定)",
    )
    p_search.set_defaults(func=cmd_search)

    p_download = sub.add_parser("download", help="Hugging Face Hub からダウンロードする")
    p_download.add_argument("repo_id")
    p_download.add_argument("--dir", default=DEFAULT_MODELS_DIR)
    p_download.add_argument(
        "--name",
        default=None,
        help="保存先ディレクトリ名 (既定: repo_id の最後のセグメント)",
    )
    p_download.set_defaults(func=cmd_download)

    return ap


def main(argv=None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
