"""Fetch specification archives from the public 3GPP FTP mirror.

3GPP encodes a spec version in the filename as three base-36 characters, e.g.
``24501-k00.zip`` is TS 24.501 v20.0.0. For Rel-8 onwards the major version
number equals the Release number, so ``k`` -> v20.x.y -> Release 20.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

BASE = "https://www.3gpp.org/ftp/Specs/archive"
FILE_RE = re.compile(r'href="[^"]*?/(\d{4,5})-([0-9a-z]{3})\.zip"', re.I)
USER_AGENT = "telcorag/1.0 (3GPP RAG corpus builder)"


def decode_version(code: str) -> tuple[int, int, int]:
    def val(ch: str) -> int:
        return int(ch, 36)

    if len(code) != 3:
        raise ValueError(f"bad 3GPP version code: {code!r}")
    return val(code[0]), val(code[1]), val(code[2])


def format_version(code: str) -> str:
    return "%d.%d.%d" % decode_version(code)


def encode_version(version: str) -> str:
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"bad version: {version!r}")
    return "".join(_BASE36[int(p)] for p in parts)


_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def archive_url(spec_id: str, version: str) -> str:
    return f"{BASE}/{series_of(spec_id)}/{spec_id}/{flat_id(spec_id)}-{encode_version(version)}.zip"


def series_of(spec_id: str) -> str:
    return f"{spec_id.split('.')[0]}_series"


def flat_id(spec_id: str) -> str:
    return spec_id.replace(".", "")


@dataclass
class SpecArchive:
    spec_id: str
    version_code: str
    version: str
    release: int
    url: str
    filename: str

    def as_dict(self) -> dict:
        return asdict(self)


def list_archives(spec_id: str, session: requests.Session | None = None) -> list[SpecArchive]:
    session = session or requests.Session()
    url = f"{BASE}/{series_of(spec_id)}/{spec_id}/"
    resp = session.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()

    wanted = flat_id(spec_id)
    seen: dict[str, SpecArchive] = {}
    for num, code in FILE_RE.findall(resp.text):
        if num != wanted:
            continue
        code = code.lower()
        major, _, _ = decode_version(code)
        name = f"{num}-{code}.zip"
        seen[code] = SpecArchive(
            spec_id=spec_id,
            version_code=code,
            version=format_version(code),
            release=major,
            url=f"{url}{name}",
            filename=name,
        )
    return sorted(seen.values(), key=lambda a: decode_version(a.version_code))


def pick(archives: list[SpecArchive], release: int | None = None) -> SpecArchive:
    if not archives:
        raise LookupError("no archives found")
    if release is None:
        return archives[-1]
    matching = [a for a in archives if a.release == release]
    if not matching:
        available = sorted({a.release for a in archives})
        raise LookupError(f"Release {release} unavailable; have {available}")
    return matching[-1]


def download(archive: SpecArchive, raw_dir: Path, session: requests.Session | None = None) -> Path:
    session = session or requests.Session()
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / archive.filename
    if target.exists() and target.stat().st_size > 0:
        return target

    tmp = target.with_suffix(".part")
    with session.get(archive.url, stream=True, timeout=300, headers={"User-Agent": USER_AGENT}) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for block in resp.iter_content(chunk_size=1 << 16):
                fh.write(block)
    tmp.replace(target)
    return target


def extract(zip_path: Path, corpus_dir: Path) -> list[Path]:
    """Unpack the archive, keeping only Word parts. Returns them in document order.

    3GPP splits large specs across several .docx parts whose names embed the
    clause range (``..._2_Main-Body_s05_s0504.docx``); the numeric prefix after
    the version code gives the correct reading order.
    """
    dest = corpus_dir / zip_path.stem
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name.lower().endswith((".docx", ".doc")):
                continue
            path = dest / name
            if not path.exists() or path.stat().st_size != info.file_size:
                path.write_bytes(zf.read(info))
            out.append(path)
    return sorted(out, key=_part_order)


def _part_order(path: Path) -> tuple[int, str]:
    m = re.search(r"-[0-9a-z]{3}_(\d+)_", path.name, re.I)
    return (int(m.group(1)) if m else 999, path.name.lower())
