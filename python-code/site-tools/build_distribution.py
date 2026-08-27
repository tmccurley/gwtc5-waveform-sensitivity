"""Build and verify the complete figures, analysis-code, and website archive."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from datetime import date
from pathlib import Path


INCLUDED_DIRECTORIES = (
    "python-code",
    "website",
    "figures",
    "preview-assets",
    "reproducibility",
    "results/confirmatory_analysis",
    "waveform-comparison",
)
INCLUDED_FILES = (
    ".gitignore",
    ".nojekyll",
    "index.html",
    "about.html",
    "README.md",
    "Confirmatory_protocol.md",
    "package.json",
    "results/README.md",
    "website/favicon.svg",
    "website/site-shell.js",
    "website/styles.css",
)
SCRIPT_SUFFIXES = {".py", ".ipynb", ".bat"}
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".staging",
    ".venv",
    "__pycache__",
    "node_modules",
    "reproduced_results",
}
EXCLUDED_SUFFIXES = {".h5", ".hdf5", ".part", ".pyc", ".pyo"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Archive path (default: downloads/GWTC5_all_figures_and_code_only.zip)",
    )
    return parser.parse_args()


def find_repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "website" / "index.html").is_file():
            return candidate
    raise RuntimeError("Could not locate the organized repository root.")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def should_package(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    return not (
        any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts)
        or path.suffix.lower() in EXCLUDED_SUFFIXES
    )


def collect_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for relative_dir in INCLUDED_DIRECTORIES:
        directory = root / relative_dir
        if not directory.is_dir():
            raise FileNotFoundError(f"Required directory is missing: {directory}")
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and should_package(root, path)
        )
    for relative_file in INCLUDED_FILES:
        path = root / relative_file
        if not path.is_file():
            raise FileNotFoundError(f"Required file is missing: {path}")
        files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def repository_scripts(root: Path) -> dict[str, str]:
    scripts: dict[str, str] = {}
    for path in root.rglob("*"):
        if (
            not path.is_file()
            or path.suffix.lower() not in SCRIPT_SUFFIXES
            or not should_package(root, path)
        ):
            continue
        relative = path.relative_to(root)
        if relative.parts[0] == "downloads":
            continue
        scripts[relative.as_posix()] = sha256_bytes(path.read_bytes())
    return scripts


def build_archive(root: Path, output: Path) -> tuple[int, int]:
    bundle_name = f"GWTC5_figures_and_code_only_{date.today().isoformat()}"
    files = collect_files(root)
    scripts = repository_scripts(root)
    file_hashes = {
        path.relative_to(root).as_posix(): sha256_bytes(path.read_bytes())
        for path in files
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")

    manifest_lines = ["sha256  bytes  path"]
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            archive_name = f"{bundle_name}/{relative}"
            content = path.read_bytes()
            archive.writestr(archive_name, content)
            manifest_lines.append(
                f"{sha256_bytes(content)}  {len(content)}  {relative}"
            )
        archive.writestr(
            f"{bundle_name}/FILE_MANIFEST.sha256",
            "\n".join(manifest_lines) + "\n",
        )

    with zipfile.ZipFile(temporary) as archive:
        names = set(archive.namelist())
        for relative, expected_hash in file_hashes.items():
            archive_name = f"{bundle_name}/{relative}"
            if archive_name not in names:
                raise RuntimeError(f"Packaged file is missing from archive: {relative}")
            actual_hash = sha256_bytes(archive.read(archive_name))
            if actual_hash != expected_hash:
                raise RuntimeError(f"Archived file differs from source: {relative}")
        for relative, expected_hash in scripts.items():
            archive_name = f"{bundle_name}/{relative}"
            if archive_name not in names:
                raise RuntimeError(f"Script missing from archive: {relative}")
            actual_hash = sha256_bytes(archive.read(archive_name))
            if actual_hash != expected_hash:
                raise RuntimeError(f"Archived script differs from source: {relative}")

    temporary.replace(output)
    checksum_path = output.with_name(output.name + ".sha256")
    checksum_path.write_text(
        f"{sha256_file(output)}  {output.name}\n", encoding="utf-8"
    )
    return len(files), len(scripts)


def main() -> None:
    args = parse_args()
    root = find_repository_root()
    output = args.output or (
        root / "downloads" / "GWTC5_all_figures_and_code_only.zip"
    )
    file_count, script_count = build_archive(root, output)
    print(f"Created: {output.resolve()}")
    print(f"Archive checksum: {output.resolve()}.sha256")
    print(f"Packaged and hash-verified files: {file_count}")
    print(f"Verified scripts and notebooks: {script_count}")


if __name__ == "__main__":
    main()
