"""Fetch a bounded research sample from the official VITON-HD ZIP using HTTP ranges.

No full archive or model download. Dataset remains CC BY-NC 4.0 research material.
"""
import argparse
import io
import json
import random
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath

URL = "https://drive.usercontent.google.com/download?id=1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w&export=download&confirm=t"


class RemoteZip(io.RawIOBase):
    def __init__(self, limit_mb=180):
        self.position = 0
        self.transferred = 0
        self.limit = limit_mb * 1024**2
        request = urllib.request.Request(URL, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=40) as response:
            if response.status != 206:
                raise RuntimeError("Server did not honor byte ranges; refusing full archive.")
            self.size = int(response.headers["Content-Range"].split("/")[-1])
            response.read(1)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count <= 0:
            return b""
        if count > 32 * 1024**2 or self.transferred + count > self.limit:
            raise RuntimeError("Download byte budget exceeded")
        begin = self.position
        for attempt in range(3):
            try:
                request = urllib.request.Request(URL, headers={"Range": f"bytes={begin}-{begin + count - 1}"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {begin}-"):
                        raise RuntimeError("Invalid range response")
                    data = response.read(count + 1)
                self.transferred += len(data)
                if len(data) != count:
                    raise RuntimeError("Incomplete range response")
                self.position += count
                return data
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data" / "viton")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--train", type=int, default=24)
    parser.add_argument("--test", type=int, default=8)
    args = parser.parse_args()
    if not (0 <= args.train <= 100 and 0 <= args.test <= 100):
        parser.error("Sample counts must be between 0 and 100")
    args.output.mkdir(parents=True, exist_ok=True)
    remote = RemoteZip()
    with zipfile.ZipFile(remote) as archive:
        entries = archive.infolist()
        index = [{"path": i.filename, "bytes": i.file_size, "compressed": i.compress_size, "crc32": i.CRC} for i in entries]
        (args.output / "archive-index.json").write_text(json.dumps(index), encoding="utf-8")
        names = set(archive.namelist())
        print(json.dumps({"archive_bytes": remote.size, "entries": len(index), "first_paths": sorted(names)[:25]}), flush=True)
        if args.index_only:
            return
        selected = []
        for split, count in (("train", args.train), ("test", args.test)):
            candidates = sorted(n for n in names if f"/{split}/image/" in "/" + n and n.endswith(".jpg"))
            for image in random.Random(20260926).sample(candidates, min(count, len(candidates))):
                stem = PurePosixPath(image).stem
                prefix = image.rsplit("/image/", 1)[0]
                paths = [image, f"{prefix}/cloth/{stem}.jpg", f"{prefix}/image-parse-v3/{stem}.png", f"{prefix}/agnostic-v3.2/{stem}.jpg", f"{prefix}/agnostic-mask/{stem}_mask.png"]
                if not all(p in names for p in paths):
                    raise RuntimeError("Expected paired files missing: " + str(paths))
                for name in paths:
                    member = archive.getinfo(name)
                    if member.file_size > 8 * 1024**2:
                        raise RuntimeError("Unexpectedly large sample member")
                    relative = PurePosixPath(name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise RuntimeError("Unsafe archive path")
                    target = args.output.joinpath(*relative.parts)
                    if target.exists() and zlib.crc32(target.read_bytes()) == member.CRC:
                        continue
                    content = archive.read(name)  # ZIP verifies CRC before returning.
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                selected.append({"official_split": split, "sample_id": stem, "files": paths})
                print(f"Verified {split}/{stem}", flush=True)
        (args.output / "download-record.json").write_text(json.dumps({"source": URL, "source_page": "https://github.com/shadow2496/VITON-HD", "license": "CC BY-NC 4.0; noncommercial research", "seed": 20260926, "transferred_bytes": remote.transferred, "samples": selected}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
