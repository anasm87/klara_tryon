"""Read selected members of the official VITON-HD ZIP using HTTP ranges."""

import io
import time
import urllib.request

URL = "https://drive.usercontent.google.com/download?id=1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w&export=download&confirm=t"


class RemoteZip(io.RawIOBase):

    def __init__(self, limit_mb=180):
        self.position = 0
        self.transferred = 0
        self.limit = limit_mb * 1024**2
        request = urllib.request.Request(URL, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=40) as response:
            if response.status != 206:
                raise RuntimeError(
                    "Server did not honor byte ranges; refusing full archive."
                )
            self.size = int(response.headers["Content-Range"].split("/")[-1])
            response.read(1)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (
            offset
            if whence == 0
            else self.position + offset if whence == 1 else self.size + offset
        )
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, count=-1):
        count = (
            self.size - self.position
            if count < 0
            else min(count, self.size - self.position)
        )
        if count <= 0:
            return b""
        if count > 32 * 1024**2 or self.transferred + count > self.limit:
            raise RuntimeError("Download byte budget exceeded")
        begin = self.position
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    URL, headers={"Range": f"bytes={begin}-{begin + count - 1}"}
                )
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206 or not response.headers.get(
                        "Content-Range", ""
                    ).startswith(f"bytes {begin}-"):
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
