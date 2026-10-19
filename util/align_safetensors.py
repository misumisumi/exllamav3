"""
Re-lay safetensors shards so every tensor starts on an N-byte file offset.

Zero-copy loading (EXL3_ATS_MMAP=1) aliases tensors straight out of a file mapping, and a mapping
cannot move a tensor that the writer packed at an odd offset: EXL3 trellis kernels need their int16
data on a 16-byte grid. This streams each shard that has an off-grid tensor into a new directory,
pads the header with spaces and fills gaps with small "__align_pad__.N" U8 tensors (so the file
stays a contiguous, standard safetensors buffer), verifies every tensor's bytes against the source,
and symlinks shards that need no change plus all other files.

    python util/align_safetensors.py SRC_DIR DST_DIR [--align 64] [--min-bytes 1048576]
        [--skip .engram.embed.] [--jobs 4]
"""

import argparse
import json
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

CHUNK = 64 << 20
PAD_PREFIX = "__align_pad__."
O_BINARY = getattr(os, "O_BINARY", 0)


def read_header(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def tensors_by_offset(header: dict):
    return sorted(
        ((k, v) for k, v in header.items() if k != "__metadata__"),
        key = lambda kv: kv[1]["data_offsets"][0]
    )


def needs_rewrite(header: dict, data0: int, align: int, min_bytes: int, skip: list[str]) -> bool:
    for k, v in tensors_by_offset(header):
        b, e = v["data_offsets"]
        if e - b >= min_bytes and not any(s in k for s in skip) and (data0 + b) % align:
            return True
    return False


def plan_layout(header: dict, align: int, tag: str):
    # Pad names carry the shard tag: loaders that merge shard headers would otherwise see the
    # same pad key in every shard
    new = {}
    if "__metadata__" in header:
        new["__metadata__"] = header["__metadata__"]
    plan, pos, pads = [], 0, 0
    for k, v in tensors_by_offset(header):
        if k.startswith(PAD_PREFIX):
            continue
        b, e = v["data_offsets"]
        at = -(-pos // align) * align
        if at > pos:
            new[f"{PAD_PREFIX}{tag}.{pads}"] = {"dtype": "U8", "shape": [at - pos], "data_offsets": [pos, at]}
            pads += 1
        new[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [at, at + e - b]}
        plan.append((k, b, at, e - b))
        pos = at + e - b
    blob = json.dumps(new, separators = (",", ":")).encode()
    blob += b" " * ((-(8 + len(blob))) % align)
    return blob, plan, pos, pads


def pread_all(fd: int, n: int, off: int) -> bytes:
    parts = []
    while n:
        if hasattr(os, "pread"):
            b = os.pread(fd, n, off)
        else:
            os.lseek(fd, off, os.SEEK_SET)
            b = os.read(fd, n)
        if not b:
            raise OSError(f"short read at {off}")
        parts.append(b)
        n -= len(b)
        off += len(b)
    return b"".join(parts)


def pwrite_all(fd: int, data: bytes, off: int):
    view = memoryview(data)
    while view:
        if hasattr(os, "pwrite"):
            w = os.pwrite(fd, view, off)
        else:
            os.lseek(fd, off, os.SEEK_SET)
            w = os.write(fd, view)
        view = view[w:]
        off += w


def copy_range(sfd: int, dfd: int, s_off: int, d_off: int, size: int):
    if hasattr(os, "copy_file_range"):
        while size:
            n = os.copy_file_range(sfd, dfd, min(size, 1 << 30), s_off, d_off)
            if n <= 0:
                raise OSError(f"copy_file_range stalled at {s_off}")
            s_off += n
            d_off += n
            size -= n
    else:
        while size:
            n = min(size, CHUNK)
            pwrite_all(dfd, pread_all(sfd, n, s_off), d_off)
            s_off += n
            d_off += n
            size -= n


def same_bytes(sfd: int, dfd: int, s_off: int, d_off: int, size: int) -> bool:
    while size:
        n = min(size, CHUNK)
        if pread_all(sfd, n, s_off) != pread_all(dfd, n, d_off):
            return False
        s_off += n
        d_off += n
        size -= n
    return True


def check_layout(path: str, src_header: dict, align: int):
    header, data0 = read_header(path)
    if data0 % align:
        raise RuntimeError(f"data start {data0} is off the {align}-byte grid")
    pos = 0
    for k, v in tensors_by_offset(header):
        b, e = v["data_offsets"]
        if b != pos:
            raise RuntimeError(f"{k} starts at {b}, expected contiguous {pos}")
        if not k.startswith(PAD_PREFIX):
            if (data0 + b) % align:
                raise RuntimeError(f"{k} is off the grid")
            s = src_header[k]
            if s["dtype"] != v["dtype"] or s["shape"] != v["shape"] or s["data_offsets"][1] - s["data_offsets"][0] != e - b:
                raise RuntimeError(f"{k} dtype/shape/size changed")
        pos = e
    if data0 + pos != os.path.getsize(path):
        raise RuntimeError(f"data ends at {data0 + pos}, file is {os.path.getsize(path)} bytes")
    missing = {k for k in src_header if k != "__metadata__" and not k.startswith(PAD_PREFIX)} - set(header)
    if missing:
        raise RuntimeError(f"{len(missing)} tensors missing, e.g. {sorted(missing)[0]}")


def process_shard(src_path: str, dst_path: str, args) -> str:
    name = os.path.basename(src_path)
    if os.path.lexists(dst_path):
        return f"{name}: exists, skipped"
    header, data0 = read_header(src_path)
    if not needs_rewrite(header, data0, args.align, args.min_bytes, args.skip):
        os.symlink(os.path.realpath(src_path), dst_path)
        return f"{name}: already on grid, linked"
    blob, plan, end, pads = plan_layout(header, args.align, name.removesuffix(".safetensors"))
    tmp = dst_path + ".tmp"
    t0 = time.time()
    sfd = os.open(src_path, os.O_RDONLY | O_BINARY)
    dfd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC | O_BINARY, 0o644)
    try:
        pwrite_all(dfd, struct.pack("<Q", len(blob)) + blob, 0)
        ndata0 = 8 + len(blob)
        for _, b, at, size in plan:
            copy_range(sfd, dfd, data0 + b, ndata0 + at, size)
        os.ftruncate(dfd, ndata0 + end)
        os.fsync(dfd)
        check_layout(tmp, header, args.align)
        for k, b, at, size in plan:
            if not same_bytes(sfd, dfd, data0 + b, ndata0 + at, size):
                raise RuntimeError(f"bytes differ for {k}")
    except Exception as e:
        raise RuntimeError(f"{name}: {e}") from e
    finally:
        os.close(sfd)
        os.close(dfd)
    os.replace(tmp, dst_path)
    return f"{name}: rewritten, {len(plan)} tensors, {pads} pads, {end / 2**30:.2f} GiB, {time.time() - t0:.0f}s, verified"


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help = "source model directory")
    ap.add_argument("dst", help = "output directory (created)")
    ap.add_argument("--align", type = int, default = 64, help = "tensor byte alignment, power of two")
    ap.add_argument("--min-bytes", type = int, default = 1 << 20, help = "only smaller tensors may stay off grid")
    ap.add_argument("--skip", action = "append", default = None,
                    help = "substring of tensor names that never force a rewrite (default .engram.embed.)")
    ap.add_argument("--jobs", type = int, default = 4, help = "shards processed in parallel")
    args = ap.parse_args()
    if args.skip is None:
        args.skip = [".engram.embed."]
    if args.align <= 0 or args.align & (args.align - 1):
        sys.exit("--align must be a power of two")

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    if src == dst:
        sys.exit("src and dst must differ")
    os.makedirs(dst, exist_ok = True)

    shards = []
    for name in sorted(os.listdir(src)):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if name.endswith(".safetensors"):
            shards.append((s, d))
        elif not os.path.lexists(d) and not name.endswith(".tmp"):
            os.symlink(os.path.realpath(s), d)

    failures = 0
    with ThreadPoolExecutor(max_workers = args.jobs) as pool:
        futures = [pool.submit(process_shard, s, d, args) for s, d in shards]
        for fut in futures:
            try:
                print(f" -- {fut.result()}", flush = True)
            except Exception as e:
                failures += 1
                print(f" !! {e}", flush = True)
    print(f"done: {len(shards) - failures}/{len(shards)} shards ok", flush = True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
