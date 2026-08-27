"""Download and verify all 104 official GWTC-5.0 event files.

The frozen analysis capsule records a complete catalog inventory, but its own
acquisition helper was scoped to files missing during the original run. This
repository-level helper starts cleanly from an empty directory and uses only
the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUFFER_SIZE = 4 * 1024 * 1024
EXPECTED_EVENT_COUNT = 104
REQUIRED_COLUMNS = {
    "event",
    "event_filename",
    "official_size_bytes",
    "official_md5",
    "download_url",
}


def default_manifest() -> Path:
    return (
        Path(__file__).resolve().parent
        / "full-catalog-generalization"
        / "design"
        / "records"
        / "full_catalog_event_inventory.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest(),
        help="Frozen 104-event inventory CSV.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        required=True,
        help="Directory that will contain the official HDF5 files.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Verification-record directory (default: DOWNLOAD_DIR/_gwtc5_verification).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Concurrent transfers, from 1 to 4 (default: 2).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Attempts per incomplete download (default: 5).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Network timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check local files without making network requests.",
    )
    return parser.parse_args()


def read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise RuntimeError(
                "Inventory is missing required columns: " + ", ".join(sorted(missing))
            )
        rows = list(reader)
    events = {row["event"] for row in rows}
    filenames = {row["event_filename"] for row in rows}
    if len(rows) != EXPECTED_EVENT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENT_COUNT} inventory rows, found {len(rows)}."
        )
    if len(events) != EXPECTED_EVENT_COUNT or len(filenames) != EXPECTED_EVENT_COUNT:
        raise RuntimeError("Inventory event names and filenames must each be unique.")
    return rows


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_path(path: Path, expected_size: int, expected_md5: str) -> tuple[bool, int, str]:
    if not path.is_file():
        return False, 0, ""
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        return False, actual_size, ""
    actual_md5 = md5_file(path)
    return actual_md5.lower() == expected_md5.lower(), actual_size, actual_md5


def result_record(
    row: dict[str, str],
    target: Path,
    status: str,
    verified: bool,
    actual_size: int = 0,
    actual_md5: str = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "event": row["event"],
        "event_filename": row["event_filename"],
        "status": status,
        "verified": verified,
        "expected_size_bytes": int(row["official_size_bytes"]),
        "actual_size_bytes": actual_size,
        "expected_md5": row["official_md5"].lower(),
        "actual_md5": actual_md5.lower(),
        "source_url": row["download_url"],
        "local_path": str(target.resolve()),
        "completed_utc": utc_now(),
        "error": error,
    }


def download_once(row: dict[str, str], download_dir: Path, timeout: int) -> dict[str, Any]:
    filename = row["event_filename"]
    target = download_dir / filename
    partial = download_dir / f"{filename}.part"
    expected_size = int(row["official_size_bytes"])
    expected_md5 = row["official_md5"].lower()

    verified, actual_size, actual_md5 = verify_path(
        target, expected_size, expected_md5
    )
    if verified:
        return result_record(
            row,
            target,
            "already_present_verified",
            True,
            actual_size,
            actual_md5,
        )
    if target.exists():
        return result_record(
            row,
            target,
            "existing_file_failed_verification",
            False,
            actual_size,
            actual_md5,
            "Move the conflicting file aside before retrying; it was not overwritten.",
        )

    resume_at = partial.stat().st_size if partial.is_file() else 0
    if resume_at > expected_size:
        partial.unlink()
        resume_at = 0
    headers = {"User-Agent": "GWTC5-full-catalog-reproduction/1.0"}
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"

    request = urllib.request.Request(row["download_url"], headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status_code = getattr(response, "status", response.getcode())
        if resume_at and status_code == 206:
            mode = "ab"
        else:
            mode = "wb"
            resume_at = 0
        with partial.open(mode) as handle:
            while block := response.read(BUFFER_SIZE):
                handle.write(block)

    verified, actual_size, actual_md5 = verify_path(
        partial, expected_size, expected_md5
    )
    if not verified:
        raise RuntimeError(
            f"Downloaded file failed verification: size={actual_size}, md5={actual_md5 or 'not computed'}"
        )
    os.replace(partial, target)
    return result_record(
        row, target, "downloaded_verified", True, actual_size, actual_md5
    )


def process_row(
    row: dict[str, str],
    download_dir: Path,
    retries: int,
    timeout: int,
    verify_only: bool,
) -> dict[str, Any]:
    target = download_dir / row["event_filename"]
    expected_size = int(row["official_size_bytes"])
    expected_md5 = row["official_md5"].lower()
    if verify_only:
        verified, actual_size, actual_md5 = verify_path(
            target, expected_size, expected_md5
        )
        status = "verified" if verified else "missing_or_invalid"
        return result_record(
            row, target, status, verified, actual_size, actual_md5
        )

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            result = download_once(row, download_dir, timeout)
            result["attempt"] = attempt
            return result
        except (
            OSError,
            RuntimeError,
            TimeoutError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < retries:
                time.sleep(min(30, 2 ** (attempt - 1)))
    result = result_record(
        row, target, "failed_after_retries", False, error=last_error
    )
    result["attempt"] = retries
    return result


def write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "event",
        "event_filename",
        "status",
        "verified",
        "attempt",
        "expected_size_bytes",
        "actual_size_bytes",
        "expected_md5",
        "actual_md5",
        "source_url",
        "local_path",
        "completed_utc",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["event"]))


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("--workers must be between 1 and 4.")
    if args.retries < 1:
        raise ValueError("--retries must be at least 1.")
    if args.timeout < 1:
        raise ValueError("--timeout must be at least 1 second.")

    rows = read_inventory(args.manifest.resolve())
    download_dir = args.download_dir.resolve()
    log_dir = (args.log_dir or download_dir / "_gwtc5_verification").resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_row,
                row,
                download_dir,
                args.retries,
                args.timeout,
                args.verify_only,
            ): row["event"]
            for row in rows
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write_records(log_dir / "full_catalog_file_verification.csv", results)
            print(
                f"{len(results):3d}/{len(rows)} {result['event']} {result['status']}",
                flush=True,
            )

    failures = [result for result in results if not result["verified"]]
    summary = {
        "completed_utc": utc_now(),
        "mode": "verify_only" if args.verify_only else "download_and_verify",
        "manifest": str(args.manifest.resolve()),
        "download_dir": str(download_dir),
        "expected_event_count": EXPECTED_EVENT_COUNT,
        "verified_event_count": len(results) - len(failures),
        "failure_count": len(failures),
        "total_expected_bytes": sum(
            int(row["official_size_bytes"]) for row in rows
        ),
    }
    (log_dir / "full_catalog_download_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
