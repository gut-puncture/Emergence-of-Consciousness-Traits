"""Source metadata manifests for deterministic Hugging Face source scheduling."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import yaml

from twc.data.hf_stream import EXPECTED_FINEWEB_EDU_FIELDS
from twc.data.decision_schema import stable_json_hash


SOURCE_METADATA_SCHEMA_VERSION = 1
METADATA_TOOL_VERSION = "source_metadata_v1"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
HF_API = "https://huggingface.co/api/datasets"
DEFAULT_FOOTER_CACHE_DELTA_CAP_BYTES = 256 * 1024 * 1024
DEFAULT_FOOTER_READ_BLOCK_SIZE_BYTES = 8 * 1024 * 1024
DEFAULT_REQUIRED_COLUMNS = EXPECTED_FINEWEB_EDU_FIELDS


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_hash(payload: Any) -> str:
    return stable_json_hash(payload)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_headers(headers: dict[str, Any]) -> dict[str, str]:
    return {str(key).casefold(): str(value) for key, value in headers.items()}


def _strip_quotes(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().strip('"')


def _strong_hash(value: str | None) -> bool:
    if value is None:
        return False
    stripped = _strip_quotes(value) or ""
    return len(stripped) >= 40 and all(char in "0123456789abcdefABCDEF" for char in stripped)


def _file_path_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path)
    marker = "/parquet/"
    if marker in path:
        return path.split(marker, 1)[1].lstrip("/")
    return Path(path).name


def _footer_lookup_keys(file_entry: dict[str, Any]) -> tuple[str, ...]:
    source_path = _file_path_from_url(file_entry["url"])
    filename = str(file_entry["filename"])
    keys = [source_path, filename]
    if source_path.startswith("sample-100BT/train/"):
        keys.append(source_path.removeprefix("sample-100BT/train/"))
    if source_path.startswith("sample-10BT/train/"):
        keys.append(source_path.removeprefix("sample-10BT/train/"))
    if source_path.startswith("sample-350BT/train/"):
        keys.append(source_path.removeprefix("sample-350BT/train/"))
    return tuple(dict.fromkeys(keys))


def _find_by_filename_or_path(mapping: dict[str, Any], file_entry: dict[str, Any]) -> Any:
    for key in _footer_lookup_keys(file_entry):
        if key in mapping:
            return mapping[key]
    return None


def _api_entry(name: str, api_responses: dict[str, Any]) -> dict[str, Any]:
    entry = api_responses.get(name) or {}
    payload = entry.get("payload")
    canonical_payload = _canonicalize_api_payload(payload)
    return {
        "name": name,
        "endpoint": entry.get("endpoint"),
        "status_code": entry.get("status_code"),
        "canonical_json_sha256": _canonical_hash(canonical_payload),
        "required_fields_present": payload is not None,
        "payload": payload,
    }


def _canonicalize_api_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    canonical = deepcopy_json(payload)
    if isinstance(canonical.get("parquet_files"), list):
        canonical["parquet_files"] = sorted(
            canonical["parquet_files"],
            key=lambda file: (str(file.get("url", "")), str(file.get("filename", ""))),
        )
    if isinstance(canonical.get("splits"), list):
        canonical["splits"] = sorted(
            canonical["splits"],
            key=lambda item: (
                str(item.get("dataset", "")),
                str(item.get("config", "")),
                str(item.get("split", "")),
            ),
        )
    size = canonical.get("size")
    if isinstance(size, dict) and isinstance(size.get("splits"), list):
        size["splits"] = sorted(
            size["splits"],
            key=lambda item: (
                str(item.get("dataset", "")),
                str(item.get("config", "")),
                str(item.get("split", "")),
            ),
        )
    return canonical


def deepcopy_json(payload: Any) -> Any:
    return json.loads(json.dumps(payload, sort_keys=True))


def _conversion_layer(parquet_files: list[dict[str, Any]]) -> dict[str, Any]:
    urls = [str(file.get("url", "")) for file in parquet_files]
    uses_conversion = any("refs%2Fconvert%2Fparquet" in url or "refs/convert/parquet" in url for url in urls)
    return {
        "uses_dataset_viewer_conversion": uses_conversion,
        "conversion_ref": "refs/convert/parquet" if uses_conversion else None,
        "conversion_identity_strength": "per_file_strong" if uses_conversion else "original_repo_or_unknown",
    }


def _file_identity(
    file_entry: dict[str, Any],
    headers: dict[str, Any],
) -> tuple[dict[str, Any], str, list[str]]:
    normalized = _normalize_headers(headers)
    url = str(file_entry["url"])
    filename = str(file_entry["filename"])
    source_path = _file_path_from_url(url)
    declared_size = int(file_entry.get("size") or 0)
    content_length = normalized.get("content-length")
    linked_size = normalized.get("x-linked-size")
    identity = {
        "url": url,
        "source_path": source_path,
        "declared_size": declared_size,
        "content_length": int(content_length) if content_length and content_length.isdigit() else None,
        "x_linked_size": int(linked_size) if linked_size and linked_size.isdigit() else None,
        "etag": _strip_quotes(normalized.get("etag")),
        "last_modified": normalized.get("last-modified"),
        "x_repo_commit": normalized.get("x-repo-commit"),
        "x_linked_etag": _strip_quotes(normalized.get("x-linked-etag")),
        "x_xet_hash": normalized.get("x-xet-hash"),
    }
    errors: list[str] = []
    if identity["content_length"] is not None and identity["content_length"] != declared_size:
        errors.append("file_size_conflict")
    if identity["x_linked_size"] is not None and identity["x_linked_size"] != declared_size:
        errors.append("file_size_conflict")
    strong_etag = _strong_hash(identity["etag"]) and (
        identity["content_length"] == declared_size or identity["x_linked_size"] == declared_size
    )
    has_strong = (
        _strong_hash(identity["x_xet_hash"])
        or _strong_hash(identity["x_linked_etag"])
        or _strong_hash(identity["x_repo_commit"])
        or strong_etag
    )
    strength = "strong" if has_strong else "weak"
    identity["identity_hash"] = _canonical_hash(identity)
    identity["filename"] = filename
    return identity, strength, errors


def _file_id(dataset_repo_id: str, config_name: str, split: str, url: str) -> str:
    return _sha256_text(f"{dataset_repo_id}\0{config_name}\0{split}\0{url}")[:24]


def _schema_fingerprint(columns: list[str]) -> str:
    return _canonical_hash(columns)


def _normalize_footer(raw_footer: dict[str, Any] | None) -> dict[str, Any] | None:
    if raw_footer is None:
        return None
    columns = list(raw_footer.get("columns") or [])
    row_groups = []
    for index, group in enumerate(raw_footer.get("row_groups") or []):
        row_groups.append(
            {
                "row_group_index": int(group.get("row_group_index", index)),
                "num_rows": int(group.get("num_rows") or 0),
                "total_byte_size": int(group.get("total_byte_size") or 0),
                "compressed_byte_size": int(group.get("compressed_byte_size") or 0),
            }
        )
    return {
        "columns": columns,
        "schema_fingerprint": raw_footer.get("schema_fingerprint") or _schema_fingerprint(columns),
        "row_group_count": len(row_groups),
        "total_rows": sum(group["num_rows"] for group in row_groups),
        "row_groups": row_groups,
    }


def _dataset_viewer_errors(api_entries: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for entry in api_entries.values():
        payload = entry["payload"]
        if entry["status_code"] != 200 or payload is None:
            errors.append("dataset_viewer_unavailable")
            continue
        if payload.get("partial") is True:
            errors.append("dataset_viewer_partial")
        if payload.get("pending"):
            errors.append("dataset_viewer_pending")
        if payload.get("failed"):
            errors.append("dataset_viewer_failed")
    return errors


def _required_columns_missing(footer: dict[str, Any] | None) -> bool:
    if footer is None:
        return False
    return bool(set(EXPECTED_FINEWEB_EDU_FIELDS) - set(footer["columns"]))


def build_source_metadata_manifest(
    *,
    dataset_repo_id: str,
    config_name: str,
    split: str,
    requested_revision: str | None,
    resolved_repo_sha: str | None,
    api_responses: dict[str, Any],
    headers_by_filename: dict[str, dict[str, Any]],
    parquet_footers: dict[str, dict[str, Any]],
    storage_observations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a no-raw-text source metadata manifest from already fetched metadata."""

    storage_observations = storage_observations or {}
    api_entries = {name: _api_entry(name, api_responses) for name in ("splits", "size", "parquet", "hub_dataset")}
    parquet_payload = api_entries["parquet"]["payload"] or {}
    parquet_files = sorted(
        list(parquet_payload.get("parquet_files") or []),
        key=lambda file: (str(file.get("url", "")), str(file.get("filename", ""))),
    )
    conversion = _conversion_layer(parquet_files)

    files = []
    footers = []
    errors = _dataset_viewer_errors(api_entries)
    warnings: list[str] = []

    split_entries = (api_entries["splits"]["payload"] or {}).get("splits") or []
    if not any(
        entry.get("dataset") == dataset_repo_id
        and entry.get("config") == config_name
        and entry.get("split") == split
        for entry in split_entries
    ):
        errors.append("missing_requested_split")
    if not parquet_files:
        errors.append("missing_parquet_files")
    if not resolved_repo_sha or len(resolved_repo_sha) != 40:
        errors.append("missing_resolved_repo_sha")

    for file_entry in parquet_files:
        header_key = _find_by_filename_or_path(headers_by_filename, file_entry)
        headers = header_key if isinstance(header_key, dict) else {}
        identity, strength, file_errors = _file_identity(file_entry, headers)
        errors.extend(file_errors)
        if strength != "strong":
            errors.append("weak_file_identity")

        file_id = _file_id(dataset_repo_id, config_name, split, identity["url"])
        footer = _normalize_footer(_find_by_filename_or_path(parquet_footers, file_entry))
        if footer is None:
            errors.append("missing_parquet_footer")
        elif _required_columns_missing(footer):
            errors.append("missing_required_column")

        files.append(
            {
                "file_id": file_id,
                "dataset_repo_id": dataset_repo_id,
                "config": config_name,
                "split": split,
                "filename": identity["filename"],
                "source_path": identity["source_path"],
                "parquet_url": identity["url"],
                "size": identity["declared_size"],
                "identity_strength": strength,
                "parquet_identity": identity,
                "metadata_errors": sorted(set(file_errors + ([] if strength == "strong" else ["weak_file_identity"]))),
            }
        )
        footers.append(
            {
                "file_id": file_id,
                "filename": identity["filename"],
                "source_path": identity["source_path"],
                **(footer or {"columns": [], "schema_fingerprint": None, "row_group_count": 0, "total_rows": 0, "row_groups": []}),
            }
        )

    size_payload = api_entries["size"]["payload"] or {}
    split_sizes = ((size_payload.get("size") or {}).get("splits") or [])
    if split_sizes:
        declared_bytes = split_sizes[0].get("num_bytes_parquet_files")
        if declared_bytes is not None and sum(file["size"] for file in files) != int(declared_bytes):
            warnings.append("parquet_file_size_sum_differs_from_size_endpoint")

    footer_cache_delta = int(storage_observations.get("footer_cache_delta_bytes") or 0)
    footer_cache_cap = int(
        storage_observations.get("footer_cache_delta_cap_bytes")
        or DEFAULT_FOOTER_CACHE_DELTA_CAP_BYTES
    )
    if footer_cache_delta > footer_cache_cap:
        errors.append("footer_cache_delta_over_cap")

    duplicate_counts = Counter((file["filename"] for file in files))
    if any(count > 1 for count in duplicate_counts.values()):
        warnings.append("duplicate_filenames_disambiguated_by_url")

    unique_errors = sorted(set(errors))
    freezable = not unique_errors
    validation = {
        "freezable": freezable,
        "diagnostic_only": not freezable,
        "errors": unique_errors,
        "warnings": sorted(set(warnings)),
        "raw_text_serialized": False,
        "token_serialized": False,
        "footer_cache_delta_bytes": footer_cache_delta,
        "footer_cache_delta_cap_bytes": footer_cache_cap,
    }
    body = {
        "schema_version": SOURCE_METADATA_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "metadata_tool_version": METADATA_TOOL_VERSION,
        "dataset": {
            "repo_id": dataset_repo_id,
            "config": config_name,
            "split": split,
            "requested_revision": requested_revision,
            "resolved_repo_sha": resolved_repo_sha,
        },
        "api_responses": api_entries,
        "conversion_layer": conversion,
        "files": files,
        "parquet_footers": footers,
        "storage_observations": storage_observations,
        "validation": validation,
        "raw_text_serialized": False,
        "token_serialized": False,
        "datasets_version": _package_version("datasets"),
        "huggingface_hub_version": _package_version("huggingface_hub"),
        "pyarrow_version": _package_version("pyarrow"),
    }
    stable_body = {
        key: value
        for key, value in body.items()
        if key not in {"generated_at_utc"}
    }
    stable_body["api_responses"] = {
        name: {
            **entry,
            "payload": _canonicalize_api_payload(entry.get("payload")),
        }
        for name, entry in body["api_responses"].items()
    }
    manifest_hash = _canonical_hash(stable_body)
    body["source_metadata_hash"] = manifest_hash
    body["manifest_hash"] = manifest_hash
    return body


def validate_source_metadata_manifest(manifest: dict[str, Any], *, require_freezable: bool = False) -> None:
    if manifest.get("schema_version") != SOURCE_METADATA_SCHEMA_VERSION:
        raise ValueError("unsupported source metadata manifest schema")
    validation = manifest.get("validation") or {}
    if require_freezable and not validation.get("freezable"):
        raise ValueError(f"source metadata manifest is not freezable: {validation.get('errors')}")
    if manifest.get("raw_text_serialized") is not False or manifest.get("token_serialized") is not False:
        raise ValueError("source metadata manifest must not serialize raw text or tokens")


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "twc-source-scheduler/1"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_headers(url: str) -> dict[str, str]:
    request = Request(url, method="HEAD", headers={"User-Agent": "twc-source-scheduler/1"})
    with urlopen(request, timeout=60) as response:
        return dict(response.headers.items())


def collect_live_dataset_viewer_metadata(
    *,
    dataset_repo_id: str,
    config_name: str,
    split: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    splits_url = f"{DATASETS_SERVER}/splits?dataset={dataset_repo_id}"
    size_url = f"{DATASETS_SERVER}/size?dataset={dataset_repo_id}&config={config_name}&split={split}"
    parquet_url = f"{DATASETS_SERVER}/parquet?dataset={dataset_repo_id}&config={config_name}&split={split}"
    hub_url = f"{HF_API}/{dataset_repo_id}"
    api_responses = {
        "splits": {"endpoint": splits_url, "status_code": 200, "payload": _fetch_json(splits_url)},
        "size": {"endpoint": size_url, "status_code": 200, "payload": _fetch_json(size_url)},
        "parquet": {"endpoint": parquet_url, "status_code": 200, "payload": _fetch_json(parquet_url)},
        "hub_dataset": {"endpoint": hub_url, "status_code": 200, "payload": _fetch_json(hub_url)},
    }
    headers_by_filename = {}
    for file_entry in api_responses["parquet"]["payload"].get("parquet_files", []):
        headers_by_filename[_file_path_from_url(file_entry["url"])] = _fetch_headers(file_entry["url"])
    return api_responses, headers_by_filename


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for current, _, files in os.walk(path):
        for file in files:
            try:
                total += (Path(current) / file).stat().st_size
            except OSError:
                continue
    return total


@contextmanager
def _open_parquet_footer_source(url: str, *, block_size: int = DEFAULT_FOOTER_READ_BLOCK_SIZE_BYTES):
    """Open a remote Parquet artifact as a seekable file-like object.

    PyArrow does not infer a filesystem for raw Hugging Face HTTPS URLs. fsspec
    gives ParquetFile the random-access file interface it needs without treating
    the HTTPS URL itself as a local/Arrow filesystem URI.
    """

    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - optional data dependency
        raise RuntimeError("fsspec is required for HTTPS footer probing") from exc

    with fsspec.open(url, mode="rb", block_size=block_size, cache_type="readahead") as source:
        yield source


def collect_parquet_footer_metadata(
    parquet_files: list[dict[str, Any]],
    *,
    cache_dir: str | Path,
    max_files: int | None,
    cache_delta_cap_bytes: int,
    read_block_size_bytes: int = DEFAULT_FOOTER_READ_BLOCK_SIZE_BYTES,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read Parquet footer metadata with a measured cache delta.

    This function is opt-in for live runs. It records only schema and row-group
    metadata; callers should keep ``max_files`` small until cache behavior is
    verified on the current machine/network path.
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyarrow is required for footer probing") from exc
    if max_files is not None and max_files < 0:
        raise ValueError("max_files must be non-negative or None")
    if read_block_size_bytes < 1:
        raise ValueError("read_block_size_bytes must be positive")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    before = _directory_size(cache_path)
    old_home = os.environ.get("HF_HOME")
    old_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
    old_hub_cache = os.environ.get("HF_HUB_CACHE")
    os.environ["HF_HOME"] = str(cache_path / "hf_home")
    os.environ["HF_DATASETS_CACHE"] = str(cache_path / "datasets")
    os.environ["HF_HUB_CACHE"] = str(cache_path / "hub")
    footers: dict[str, dict[str, Any]] = {}
    try:
        selected = parquet_files[:max_files] if max_files is not None else parquet_files
        for file_entry in selected:
            with _open_parquet_footer_source(
                str(file_entry["url"]),
                block_size=read_block_size_bytes,
            ) as source:
                parquet_file = pq.ParquetFile(source)
                metadata = parquet_file.metadata
                columns = list(parquet_file.schema.names)
            groups = []
            for group_index in range(metadata.num_row_groups):
                group = metadata.row_group(group_index)
                compressed = 0
                for column_index in range(group.num_columns):
                    compressed += int(group.column(column_index).total_compressed_size or 0)
                groups.append(
                    {
                        "row_group_index": group_index,
                        "num_rows": int(group.num_rows),
                        "total_byte_size": int(group.total_byte_size or 0),
                        "compressed_byte_size": compressed,
                    }
                )
            footers[_file_path_from_url(file_entry["url"])] = {
                "columns": columns,
                "schema_fingerprint": _schema_fingerprint(columns),
                "row_groups": groups,
            }
            after_each = _directory_size(cache_path)
            if after_each - before > cache_delta_cap_bytes:
                raise RuntimeError("footer cache delta exceeded configured cap")
    finally:
        for key, value in (
            ("HF_HOME", old_home),
            ("HF_DATASETS_CACHE", old_datasets_cache),
            ("HF_HUB_CACHE", old_hub_cache),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    after = _directory_size(cache_path)
    observations = {
        "footer_cache_dir": str(cache_path),
        "footer_cache_delta_bytes": after - before,
        "footer_cache_delta_cap_bytes": cache_delta_cap_bytes,
        "footer_probe_file_count": len(footers),
        "footer_probe_requested_file_count": len(selected),
        "footer_probe_total_file_count": len(parquet_files),
        "footer_probe_complete": len(footers) == len(parquet_files),
        "footer_read_block_size_bytes": read_block_size_bytes,
        "footer_read_transport": "fsspec_https_readahead_file_like",
    }
    return footers, observations


def _load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_live_metadata_manifest(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    dataset = config["dataset"]
    api_responses, headers_by_filename = collect_live_dataset_viewer_metadata(
        dataset_repo_id=dataset["path"],
        config_name=dataset["config"],
        split=args.split,
    )
    resolved_sha = (api_responses["hub_dataset"]["payload"] or {}).get("sha")
    footer_cap = config.get("source_schedule", {}).get(
        "max_footer_cache_delta_bytes",
        DEFAULT_FOOTER_CACHE_DELTA_CAP_BYTES,
    )
    parquet_footers: dict[str, dict[str, Any]] = {}
    storage_observations = {
        "footer_cache_delta_bytes": 0,
        "footer_cache_delta_cap_bytes": footer_cap,
        "footer_probe_file_count": 0,
        "footer_probe_skipped": True,
    }
    if args.probe_footers:
        parquet_footers, storage_observations = collect_parquet_footer_metadata(
            api_responses["parquet"]["payload"].get("parquet_files", []),
            cache_dir=args.cache_dir,
            max_files=args.max_footer_files,
            cache_delta_cap_bytes=footer_cap,
        )
        storage_observations["footer_probe_skipped"] = False
    manifest = build_source_metadata_manifest(
        dataset_repo_id=dataset["path"],
        config_name=dataset["config"],
        split=args.split,
        requested_revision=dataset.get("revision"),
        resolved_repo_sha=resolved_sha,
        api_responses=api_responses,
        headers_by_filename=headers_by_filename,
        parquet_footers=parquet_footers,
        storage_observations=storage_observations,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--probe-footers", action="store_true")
    parser.add_argument("--max-footer-files", type=int, default=None)
    parser.add_argument("--cache-dir", default="reports/cache_probe/source_metadata")
    args = parser.parse_args()
    manifest = write_live_metadata_manifest(args)
    print(
        json.dumps(
            {
                "out": args.out,
                "file_count": len(manifest["files"]),
                "freezable": manifest["validation"]["freezable"],
                "errors": manifest["validation"]["errors"],
                "source_metadata_hash": manifest["source_metadata_hash"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
