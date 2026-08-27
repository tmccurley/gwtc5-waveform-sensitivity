"""Resumably download and verify the locked missing GWTC-5 event files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


CHUNK_SIZE = 8 * 1024 * 1024
NETWORK_CHUNK_SIZE = 1 * 1024 * 1024
WRITE_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=6)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_record(output_dir: Path, record: dict[str, Any]) -> None:
    with WRITE_LOCK:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "download_log.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def download_once(row: dict[str, str], download_dir: Path) -> dict[str, Any]:
    event = row["event"]
    filename = row["event_filename"]
    expected_size = int(row["official_size_bytes"])
    expected_md5 = row["official_md5"].lower()
    target = download_dir / filename
    partial = download_dir / f"{filename}.part"
    # Zenodo's public record download route works without API authorization.
    # The API content URL is retained in the manifest for provenance only.
    url = row["download_url"]

    if target.is_file():
        actual_size = target.stat().st_size
        actual_md5 = md5_file(target) if actual_size == expected_size else ""
        if actual_size == expected_size and actual_md5 == expected_md5:
            return {
                "event": event,
                "event_filename": filename,
                "status": "already_present_verified",
                "source_url": url,
                "local_path": str(target.resolve()),
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
                "expected_md5": expected_md5,
                "actual_md5": actual_md5,
                "verified": True,
                "completed_utc": utc_now(),
            }
        raise RuntimeError(f"Existing canonical file failed verification: {target}")

    resume_at = partial.stat().st_size if partial.is_file() else 0
    if resume_at > expected_size:
        partial.unlink()
        resume_at = 0

    headers = {"User-Agent": "GWTC5-full-catalog-downloader/1.0"}
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"
    with requests.get(
        url,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=(30, 30),
    ) as response:
        response.raise_for_status()
        status = response.status_code
        if resume_at and status != 206:
            # The server ignored Range. Restart safely rather than append a full
            # response to a partial file.
            resume_at = 0
            mode = "wb"
        else:
            mode = "ab" if resume_at else "wb"
        with partial.open(mode) as handle:
            for block in response.iter_content(chunk_size=NETWORK_CHUNK_SIZE):
                if block:
                    handle.write(block)

    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Incomplete size for {event}: {actual_size} != {expected_size}"
        )
    actual_md5 = md5_file(partial)
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"Checksum mismatch for {event}: {actual_md5} != {expected_md5}"
        )
    os.replace(partial, target)
    return {
        "event": event,
        "event_filename": filename,
        "status": "downloaded_verified",
        "source_url": url,
        "local_path": str(target.resolve()),
        "expected_size_bytes": expected_size,
        "actual_size_bytes": actual_size,
        "expected_md5": expected_md5,
        "actual_md5": actual_md5,
        "verified": True,
        "completed_utc": utc_now(),
    }


def download_with_retries(
    row: dict[str, str],
    download_dir: Path,
    output_dir: Path,
    retries: int,
) -> dict[str, Any]:
    event = row["event"]
    for attempt in range(1, retries + 1):
        started = utc_now()
        try:
            result = download_once(row, download_dir)
            result["attempt"] = attempt
            result["started_utc"] = started
            log_record(output_dir, result)
            return result
        except (OSError, RuntimeError, requests.RequestException) as error:
            failure = {
                "event": event,
                "event_filename": row["event_filename"],
                "status": "attempt_failed",
                "attempt": attempt,
                "started_utc": started,
                "failed_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
            log_record(output_dir, failure)
            if attempt == retries:
                return {
                    **failure,
                    "status": "failed_after_retries",
                    "verified": False,
                }
            time.sleep(min(60, 2 ** (attempt - 1)))
    raise AssertionError("retry loop exited unexpectedly")


def write_verification(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "event",
        "event_filename",
        "status",
        "attempt",
        "source_url",
        "local_path",
        "expected_size_bytes",
        "actual_size_bytes",
        "expected_md5",
        "actual_md5",
        "verified",
        "started_utc",
        "completed_utc",
        "failed_utc",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["event"]))


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("--workers must be between 1 and 4")
    rows = read_csv(args.manifest)
    if len(rows) != 56 or len({row["event"] for row in rows}) != 56:
        raise RuntimeError("The locked missing-file manifest must have 56 events.")

    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_event = {
            pool.submit(
                download_with_retries,
                row,
                args.download_dir,
                args.output_dir,
                args.retries,
            ): row["event"]
            for row in rows
        }
        for future in as_completed(future_to_event):
            result = future.result()
            results.append(result)
            write_verification(args.output_dir / "download_verification.csv", results)
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "total": len(rows),
                        "event": result["event"],
                        "status": result["status"],
                    }
                ),
                flush=True,
            )

    failures = [row for row in results if not row.get("verified", False)]
    summary = {
        "completed_utc": utc_now(),
        "manifest_event_count": len(rows),
        "verified_event_count": len(rows) - len(failures),
        "failure_count": len(failures),
        "total_expected_bytes": sum(
            int(row["official_size_bytes"]) for row in rows
        ),
        "workers": args.workers,
    }
    (args.output_dir / "download_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
