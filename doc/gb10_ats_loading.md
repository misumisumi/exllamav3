# Zero-copy loading on ATS systems (DGX Spark / GB10)

On a GB10 the GPU runs in ATS addressing mode: it shares the process page tables and can read any
host virtual address, including a file mapped with `mmap`, faulting pages in on its own. With
`EXL3_ATS_MMAP=1` the loader uses this to alias weight tensors straight out of the model's
safetensors files instead of copying them into CUDA allocations. The weights then live in the
page cache, which the kernel can reclaim and refill, so a model close to the size of system
memory can load in seconds and run without a second in-memory copy.

Check the mode first:

```
nvidia-smi -q | grep -i "addressing mode"
    Addressing Mode                   : ATS
```

## Why existing models usually need re-laying

A mapping can only alias a tensor whose bytes already sit where the kernels expect them. EXL3
trellis kernels need their int16 trellis data to start on a 16-byte boundary (any memory, CUDA
allocations included, faults with "misaligned address" otherwise), and every other dtype needs
its item size. Safetensors writers pack tensors back to back, so most shards of an existing EXL3
model have tensors on odd offsets. The loader copies those tensors instead of aliasing them, which
defeats the point: on a 110 GiB DeepSeek-V4.1 EXL3 model only 48.6 GiB could be aliased and
67.4 GiB were copied.

`util/align_safetensors.py` rewrites the shards so every tensor starts on an N-byte file offset.
After re-laying the same model at 64 bytes, all 116 GiB of text-model tensors aliased and nothing
was copied.

## Converting a model

```
python util/align_safetensors.py SRC_DIR DST_DIR [--align 64] [--min-bytes 1048576] \
    [--skip .engram.embed.] [--jobs 4]
```

- `--align` is the byte grid, a power of two. 16 is the minimum that satisfies the trellis
  kernels; 64 costs almost nothing extra and is what was measured.
- `--min-bytes` (default 1 MiB): smaller tensors never force a shard rewrite. The loader only
  aliases tensors of at least `EXL3_ATS_MMAP_MIN` bytes (same default), so small tensors are
  copied either way.
- `--skip` (default `.engram.embed.`): tensor name substrings that never force a rewrite. The
  Engram embedding tables are read by row, not aliased as whole tensors, and their shards are the
  largest in the model.
- `--jobs`: shards processed in parallel.

What it does:

- Shards that already have every qualifying tensor on the grid are symlinked, not copied.
- Shards that need it are streamed into `DST_DIR`: tensors are laid out in their original order,
  the header is padded with spaces to the grid, and gaps are filled with small
  `__align_pad__.<shard>.N` U8 tensors so the file stays a standard, contiguous safetensors buffer.
- Each rewritten shard is written to a `.tmp` file, fsynced, checked (grid, contiguity, file size,
  every tensor present with the same dtype, shape and size), compared byte for byte against the
  source, and only then renamed into place.
- All non-shard files (config, tokenizer, quantization metadata) are symlinked.

Practical notes:

- Keep the source directory. The output symlinks into it for every unchanged file.
- Free disk equal to the size of the rewritten shards is needed. On NVMe the 18 shards of the
  model above took about 20 minutes.
- Re-running is safe: finished shards in `DST_DIR` are skipped, and a shard that failed is redone
  from scratch.
- Intended for Linux (symlinks, `copy_file_range`). It has read/write fallbacks, but creating
  symlinks on Windows needs extra privileges.
- The loader on this branch skips `__align_pad__.*` tensors. Loading a re-laid model with other
  loaders has not been tested; the pad tensors are valid U8 tensors that no module references.

## Loading

```
EXL3_ATS_MMAP=1 python your_script.py
```

Tensors qualify for aliasing when they go to a CUDA device, are at least `EXL3_ATS_MMAP_MIN`
bytes, need no conversion on load (no transpose, padding or dtype change), and start on the grid
(`EXL3_ATS_MMAP_ALIGN`, default 16, for int16; the item size for other dtypes). Everything else
loads the normal way.

To confirm a model fully aliased, read the loader's counters after `model.load()`:

```python
aliased, copied = config.stc.ats_bytes
print(f"aliased {aliased / 2**30:.1f} GiB, copied off grid {copied / 2**30:.1f} GiB")
```

`copied` should be 0 for a re-laid model. A nonzero value names the amount that fell off the grid
and was copied into CUDA memory instead.

The mapping is shared and read-only. An earlier private mapping let GPU-touched pages turn into
copy-on-write anonymous memory that grew by tens of GiB during generation; the shared read-only
mapping keeps them as reclaimable page cache.

## Memory and warmup

- `torch.cuda.mem_get_info` on a GB10 reports free memory without counting reclaimable page
  cache. Watch `MemAvailable` in `/proc/meminfo` instead when judging headroom.
- The first tokens that touch cold weights fault them in from disk. Asking the kernel to read the
  weight ranges ahead of time removes most of that warmup:

```python
import os
stc = config.stc
fds = {}
for key, fn in stc.tensor_file_map.items():
    if ".engram.embed." in key:
        continue
    b, e = stc.file_headers[fn][key]["data_offsets"]
    fds.setdefault(fn, os.open(fn, os.O_RDONLY))
    os.posix_fadvise(fds[fn], stc.file_headers[fn]["_header_offset"] + b, e - b, os.POSIX_FADV_WILLNEED)
for fd in fds.values():
    os.close(fd)
```

- A model whose weights exceed the memory left for page cache still runs, but cold experts are
  read from disk as they are routed to, and decode slows down on prompts that touch them.

## Engram tables

With `EXL3_ATS_MMAP=1`, Engram layers also alias their row tables and gather rows on the GPU
(`EXL3_ENGRAM_ATS`, default on) instead of reading them with `pread` and copying. Because the GPU
faults cold pages one at a time, `EXL3_ENGRAM_PREFETCH` (default on) first reads the rows a
forward needs with a thread pool so their pages are already cached when the gather runs.
