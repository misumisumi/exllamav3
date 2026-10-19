from __future__ import annotations
from typing_extensions import override
import hashlib
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from .module import Module
from .linear import Linear
from ..model.config import Config
from ..util.tensor import get_for_device

"""
DeepSeek-V4.1 Engram: hashed n-gram embeddings added to the mHC residual streams before
layers `engram_layer_ids`. Each position hashes the compressed ids of the n = 2..max_ngram
tokens ending there, once per head (`n_hash_cols = (max_ngram - 1) * n_heads` rows of
`head_dim` fp8 values with one e8m0 scale per 32); `wkv` turns the concatenated rows into one
key per residual copy plus one shared value, and a normalized stream-vs-key dot product,
passed through a signed sqrt and a sigmoid, gates the value into every copy:

    streams <- streams + gate[copy] * value

Reference: DeepSeek inference/engram.py (hashing) and inference/model.py Engram (gate).
The tables stay on disk (two DiskTensorHandles per layer); rows are gathered per forward
with a thread pool, so NVMe or a network mount both work. Recurrent state per sequence:
the compressed ids of the previous max_ngram - 1 tokens (DEAD-filled at sequence start).
"""

DEAD = -1


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    if n % 3 == 0:
        return n == 3
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _next_prime(start: int, seen: set) -> int:
    c = start + 1
    while not _is_prime(c) or c in seen:
        c += 1
    return c


def _writable_dir(preferred: str) -> str:
    """Prefer `preferred`; fall back to ~/.cache if that path is missing or read-only
    (the usual case when tokenizer.json lives on an rclone/Hub mount)."""
    candidates = [
        preferred,
        os.path.join(os.path.expanduser("~/.cache/exllamav3"), "engram"),
        os.path.join(os.path.expanduser("~"), ".cache", "exllamav3", "engram"),
    ]
    for d in candidates:
        try:
            os.makedirs(d, exist_ok = True)
            probe = os.path.join(d, ".engram_write_probe")
            with open(probe, "wb") as f:
                f.write(b"")
            os.remove(probe)
            return d
        except OSError:
            continue
    return preferred


def build_compressed_token_map(tokenizer_json: str, cache_dir: str | None = None):
    """Compressed id per token id (tokens that normalize alike collapse), from tokenizer.json;
    cached as engram_token_map.<sha>.npz in cache_dir."""
    from tokenizers import Regex, Tokenizer, normalizers
    cache_dir = _writable_dir(cache_dir or os.path.dirname(os.path.abspath(tokenizer_json)))
    with open(tokenizer_json, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    cache = os.path.join(cache_dir, f"engram_token_map.{digest}.npz")
    if os.path.isfile(cache):
        z = np.load(cache)
        return z["token_map"], int(z["vocab_size"])
    backend = Tokenizer.from_file(tokenizer_json)
    sentinel = ""  # keeps a lone-space token alive through Strip()
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    n = backend.get_vocab_size(with_added_tokens = True)
    key_to_new = {}
    lookup = np.zeros(n, dtype = np.int64)
    for token_id in range(n):
        text = backend.decode([token_id], skip_special_tokens = False)
        if "�" in text:
            key = backend.id_to_token(token_id)  # partial UTF-8 byte token: raw form
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    try:
        os.makedirs(cache_dir, exist_ok = True)
        np.savez(cache, token_map = lookup, vocab_size = np.int64(len(key_to_new)))
    except OSError:
        pass
    return lookup, len(key_to_new)


class EngramHasher:
    """Hash ids of the n-grams ending at each position, shared by the engram layers of a model."""

    def __init__(self, config, tokenizer_json: str | None = None, cache_dir: str | None = None):
        self.layer_ids = tuple(config.engram_layer_ids)
        self.max_ngram = config.engram_max_ngram_size
        self.n_heads = config.engram_n_heads
        self.head_dim = config.engram_head_dim
        self.num_embeddings = tuple(config.engram_num_embeddings)
        self.context_len = self.max_ngram - 1
        tokenizer_json = tokenizer_json or os.path.join(config.directory, "tokenizer.json")
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~/.cache/exllamav3"), "engram")
        token_map, vocab_size = build_compressed_token_map(tokenizer_json, cache_dir)
        if vocab_size != config.engram_compressed_vocab_size:
            raise ValueError(f"engram: compressed vocab {vocab_size} != config {config.engram_compressed_vocab_size}")
        self.pad_id = int(token_map[config.engram_pad_token_id])
        primes, seen = [], set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram - 1):
                sizes, current = [], config.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    current = _next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(sizes)
            primes.append(per_ngram)
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
        for layer, sizes, rows in zip(self.layer_ids, flat, self.num_embeddings):
            if sum(sizes) != rows:
                raise ValueError(f"engram layer {layer}: primes sum to {sum(sizes)}, table has {rows} rows")
        bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
        mults = []
        for layer_id in self.layer_ids:
            g = np.random.default_rng(10007 * layer_id)
            v = g.integers(low = 0, high = bound, size = (self.max_ngram,), dtype = np.int64)
            mults.append(torch.tensor(v * 2 + 1))
        self.primes = torch.tensor(primes, dtype = torch.int64)                      # [layers, max_ngram-1, heads]
        self.offsets = torch.tensor(np.array([np.cumsum([0, *s[:-1]]) for s in flat]))  # [layers, n_cols]
        self.multipliers = torch.stack(mults)                                          # [layers, max_ngram]
        self.token_map = torch.tensor(token_map, dtype = torch.int64)
        self.n_cols = (self.max_ngram - 1) * self.n_heads

    def compress(self, ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        c = self.token_map[ids.to("cpu", torch.int64)]
        if token_mask is not None:
            c = torch.where(token_mask.cpu(), c, torch.full_like(c, DEAD))
        return c

    def hash_window(self, compressed: torch.Tensor, start_pos: int) -> torch.Tensor:
        """compressed: (bsz, context_len + seq) compressed ids (DEAD where no token exists);
        returns (bsz, seq, n_layers, n_cols) row ids."""
        h = self.context_len
        bsz, hl = compressed.shape
        seq = hl - h
        positions = torch.arange(start_pos, start_pos + seq).expand(bsz, seq)
        tokens, blocked = [], torch.zeros_like(positions, dtype = torch.bool)
        for shift in range(self.max_ngram):
            source = compressed[:, h - shift: h - shift + seq]
            blocked = blocked | (positions < shift) | (source == DEAD)
            tokens.append(torch.where(blocked, torch.full_like(source, self.pad_id), source))
        tokens = torch.stack(tokens, dim = -1)
        products = tokens.unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim = -1) + self.offsets


class EngramTable:
    """fp8 rows + e8m0 scales of one engram layer, gathered from disk on demand."""

    def __init__(self, stc, key: str, head_dim: int, threads: int = 32):
        self.stc = stc
        self.weight = stc.get_tensor_handle(f"{key}.weight")
        self.scale = stc.get_tensor_handle(f"{key}.scale")
        assert self.weight.dtype == torch.float8_e4m3fn and list(self.weight.row_shape) == [head_dim], \
            f"{key}.weight: expected fp8 rows of {head_dim}, got {self.weight.dtype} {self.weight.row_shape}"
        assert self.scale.dtype == torch.uint8 and self.scale.num_rows == self.weight.num_rows
        self.block = head_dim // int(self.scale.row_shape[0])
        self.head_dim = head_dim
        self.threads = threads
        self.pool = ThreadPoolExecutor(max_workers = threads) if threads > 1 else None

        # Load offline Engram rows file if EXL3_ENGRAM_ROWS env var is set
        self.rows_file_ids = None
        self.rows_file_weight = None
        self.rows_file_scale = None
        rows_file = os.environ.get("EXL3_ENGRAM_ROWS")
        if rows_file:
            match = re.search(r"layers\.(\d+)\.engram", key)
            if match:
                layer_id = match.group(1)
                # An explicitly requested rows file that cannot be read is an error: falling back to
                # the disk tables silently would hide a broken offline setup
                from safetensors import safe_open
                try:
                    with safe_open(rows_file, "pt") as f:
                        if f"layers.{layer_id}.ids" in f.keys():
                            self.rows_file_ids = f.get_tensor(f"layers.{layer_id}.ids").to(torch.int64)
                            self.rows_file_weight = f.get_tensor(f"layers.{layer_id}.weight").to(torch.uint8)
                            self.rows_file_scale = f.get_tensor(f"layers.{layer_id}.scale").to(torch.uint8)
                        else:
                            print(f" -- EXL3_ENGRAM_ROWS has no rows for layer {layer_id}: disk tables only")
                except (OSError, RuntimeError, KeyError) as e:
                    raise RuntimeError(f"EXL3_ENGRAM_ROWS={rows_file} unreadable for layer {layer_id}") from e

        # GPU-direct gather from the file-mapped tables when the collection aliases mappings (ATS)
        self.ats_off = not (getattr(stc, "ats_mmap", False) and self.rows_file_ids is None and
                            os.environ.get("EXL3_ENGRAM_ATS", "1") != "0")
        self.ats_tables = {}
        self.ats_prefetch = os.environ.get("EXL3_ENGRAM_PREFETCH", "1") != "0"

    def _gather(self, handle, ids: list, nbytes: int) -> bytes:
        fd = handle._ensure_open()
        base = handle.abs_offset

        def one(r):
            b = os.pread(fd, nbytes, base + r * nbytes)
            assert len(b) == nbytes
            return b
        if self.pool is None or len(ids) < 4:
            return b"".join(one(r) for r in ids)
        return b"".join(self.pool.map(one, ids, chunksize = 8))

    def _ats_aliases(self, device):
        """uint8 CUDA aliases [rows, row_bytes] of the weight and scale tables, created on first use
        per device; None when the GPU-direct path is off. Nothing is copied: the tables stay in
        reclaimable page cache and the GPU faults rows in on demand"""
        if self.ats_off:
            return None
        device = torch.device(device)
        idx = device.index if device.index is not None else torch.cuda.current_device()
        if idx not in self.ats_tables:
            self.ats_tables[idx] = tuple(
                self.stc._ats_alias(h.filename, h.abs_offset, h.num_rows * h.row_bytes, torch.uint8,
                                    [h.num_rows, h.row_bytes], device)
                for h in (self.weight, self.scale)
            )
        return self.ats_tables[idx]

    def rows(self, hash_ids: torch.Tensor, device) -> torch.Tensor:
        """hash_ids: any shape of row ids -> (*shape, head_dim) fp32 dequantized rows on device."""
        tables = self._ats_aliases(device)
        if tables is not None:
            w_u8, s_u8 = tables
            if self.ats_prefetch and self.pool is not None and hash_ids.device.type == "cpu":
                # GPU faults on cold rows are served one page at a time; reading the rows first with
                # the thread pool brings their pages into the page cache in parallel, so the gather
                # below finds them present (EXL3_ENGRAM_PREFETCH=0 skips this)
                needed = torch.unique(hash_ids.reshape(-1)).tolist()
                self._gather(self.weight, needed, self.weight.row_bytes)
                self._gather(self.scale, needed, self.scale.row_bytes)
            ids = hash_ids.reshape(-1).to(device, torch.int64)
            n = ids.shape[0]
            w = w_u8.index_select(0, ids).view(torch.float8_e4m3fn).float()
            s = s_u8.index_select(0, ids).float()
            vals = (w.view(n, -1, self.block) * torch.exp2(s - 127.0).unsqueeze(-1)).view(n, self.head_dim)
            return vals.view(*hash_ids.shape, self.head_dim)

        flat = hash_ids.reshape(-1).cpu().to(torch.int64)
        uniq, inverse = torch.unique(flat, return_inverse = True)

        # Hybrid gather: memory hits from rows file, disk fallback for missed ids
        if self.rows_file_ids is not None:
            n = len(self.rows_file_ids)
            pos = torch.searchsorted(self.rows_file_ids, uniq)
            hit = (pos < n) & (self.rows_file_ids[pos.clamp(max=n-1)] == uniq)

            # Gather hit rows from memory
            hit_vals = None
            if hit.any():
                hit_pos = pos[hit]
                w_hit = self.rows_file_weight[hit_pos].view(torch.float8_e4m3fn).to(device)
                s_hit = self.rows_file_scale[hit_pos].to(device)
                hit_vals = w_hit.float().view(hit_pos.shape[0], -1, self.block) * torch.exp2(s_hit.float() - 127.0).unsqueeze(-1)
                hit_vals = hit_vals.view(hit_pos.shape[0], self.head_dim)

            # Gather missed rows from disk
            miss_mask = ~hit
            if miss_mask.any():
                miss_ids = uniq[miss_mask].tolist()
                wb = self._gather(self.weight, miss_ids, self.weight.row_bytes)
                sb = self._gather(self.scale, miss_ids, self.scale.row_bytes)
                w_miss = torch.frombuffer(bytearray(wb), dtype = torch.float8_e4m3fn).view(len(miss_ids), self.head_dim).to(device)
                s_miss = torch.frombuffer(bytearray(sb), dtype = torch.uint8).view(len(miss_ids), -1).to(device)
                miss_vals = w_miss.float().view(len(miss_ids), -1, self.block) * torch.exp2(s_miss.float() - 127.0).unsqueeze(-1)
                miss_vals = miss_vals.view(len(miss_ids), self.head_dim)

            # Assemble results in uniq order
            vals = torch.empty(len(uniq), self.head_dim, device = device, dtype = torch.float32)
            if hit.any():
                vals[hit] = hit_vals
            if miss_mask.any():
                vals[miss_mask] = miss_vals
        else:
            # Disk-only gather (original path)
            ids = uniq.tolist()
            wb = self._gather(self.weight, ids, self.weight.row_bytes)
            sb = self._gather(self.scale, ids, self.scale.row_bytes)
            w = torch.frombuffer(bytearray(wb), dtype = torch.float8_e4m3fn).view(len(ids), self.head_dim).to(device)
            s = torch.frombuffer(bytearray(sb), dtype = torch.uint8).view(len(ids), -1).to(device)
            vals = w.float().view(len(ids), -1, self.block) * torch.exp2(s.float() - 127.0).unsqueeze(-1)
            vals = vals.view(len(ids), self.head_dim)

        return vals[inverse.to(device)].view(*hash_ids.shape, self.head_dim)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait = False)


class EngramLayerState:
    """Per-slot ring of compressed token ids indexed by absolute position, CPU-resident. The hash
    context of a forward starting at position p is the ids at p - ctx .. p - 1 (DEAD before the
    sequence start), so like the DSA layer states all bookkeeping is position-derived: rewinds,
    re-fed prefill chunks and accepted/rejected draft rounds need no per-layer action as long as
    the rewound distance stays inside the ring. (A front-copy + right-aligned history scheme went
    stale because DSV4State.rewind does not dispatch to layer states and the generator only rewinds
    when a draft token is rejected.)"""

    RING = 8192

    def __init__(self, module, max_batch_size: int, max_history: int, cache_id: int):
        self.module = module
        self.ctx = module.hasher.context_len
        self.ring_len = max(self.RING, max_history + self.ctx)
        self.id_state = torch.empty((max_batch_size, self.ring_len), dtype = torch.long, device = "meta")
        self.hi = torch.zeros(max_batch_size, dtype = torch.long)     # one past the highest position written
        self.device = None
        self.max_history = max_history
        self.max_batch_size = max_batch_size
        self.cache_id = cache_id

    def get_checkpoint_size(self):
        return self.ctx * 8

    def storage_size(self):
        return self.id_state.numel() * 8

    def alloc(self, device):
        self.id_state = torch.full_like(self.id_state, DEAD, device = "cpu")
        self.hi.zero_()
        self.device = device

    def free(self):
        self.id_state = torch.empty_like(self.id_state, device = "meta")
        self.device = None

    def clear(self, idx: int):
        if self.device is not None:
            self.id_state[idx].fill_(DEAD)
            self.hi[idx] = 0

    def get_state_tensors(self):
        return (self.id_state,)

    def context(self, slot: int, position: int) -> torch.Tensor:
        """Compressed ids at position - ctx .. position - 1, DEAD before the sequence start."""
        assert int(self.hi[slot]) - position <= self.ring_len - self.ctx, \
            f"EngramLayerState: position {position} rewound past the ring (written up to {int(self.hi[slot])})"
        idx = torch.arange(position - self.ctx, position)
        out = self.id_state[slot, idx % self.ring_len].clone()
        out[idx < 0] = DEAD
        return out

    def write(self, slot: int, position: int, ids: torch.Tensor):
        n = ids.shape[0]
        if n > self.ring_len:
            ids = ids[-self.ring_len:]
            position += n - self.ring_len
            n = self.ring_len
        self.id_state[slot, torch.arange(position, position + n) % self.ring_len] = ids
        self.hi[slot] = position + n

    def rewind(self, slot: int, last_history: int, num_tokens: int):
        pass  # position-indexed: the context is re-read from the ring at the new position

    def stash(self, slot, position: int = 0):
        return (self.context(slot, position),)

    def unstash(self, slot, stashed, position: int = 0):
        idx = torch.arange(position - self.ctx, position)
        keep = idx >= 0
        self.id_state[slot, idx[keep] % self.ring_len] = stashed[0][keep]
        self.hi[slot] = position

    def tp_export(self, plan):
        return {"cls": EngramLayerState, "args": {"cache_id": self.cache_id, "max_history": self.max_history,
                                                  "max_batch_size": self.max_batch_size}}


class EngramLayer(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        table_index: int,
        hasher: EngramHasher,
        hidden_size: int,
        hc_mult: int,
        rms_norm_eps: float,
        qmap: str | None = None,
        gather_threads: int = 32,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config = config, key = key, qmap = None)
        assert layer_idx < 0  # keeps the recurrent-state key distinct from the block at the same index
        self.layer_idx = layer_idx
        self.table_index = table_index
        self.hasher = hasher
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.eps = rms_norm_eps
        self.clamp_value = 1e-6
        self.gather_threads = gather_threads
        self.out_dtype = out_dtype
        self.n_hash_cols = hasher.n_cols
        self.wkv = Linear(
            config = config,
            key = f"{key}.wkv",
            in_features = self.n_hash_cols * hasher.head_dim,
            out_features = hidden_size * (hc_mult + 1),
            qmap = qmap,
            out_dtype = torch.half,
        )
        self.register_submodule(self.wkv)
        self.qk = None
        self.table = None
        self.caps.update({"recurrent_cache": True})
        self.layer_state_cls = EngramLayerState
        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}
        # Set True by tp_import. Mirrors DSV4Attention (dsv4.py:409 / :868): under TP
        # the recurrent states arrive as DSV4ExportedState and .cache is an opaque id.
        self.tp_mode = False

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        q = stc.get_tensor(f"{self.key}.q_weight", device, allow_bf16 = True, no_defer = True).float()
        k = stc.get_tensor(f"{self.key}.k_weight", device, allow_bf16 = True, no_defer = True).float()
        self.qk = (q * k).contiguous()                                   # (hc_mult, hidden), only ever used as a product
        self.table = EngramTable(stc, f"{self.key}.embed", self.hasher.head_dim, self.gather_threads)
        for rl in self.recurrent_layers:
            rl.alloc(device)

    @override
    def unload(self):
        for rl in self.recurrent_layers:
            rl.free()
        if self.table is not None:
            self.table.close()
        self.table = None
        self.qk = None
        super().unload()

    @override
    def get_tensors(self):
        return {}

    @override
    def weights_numel(self):
        return self.wkv.weights_numel() + 2 * self.hc_mult * self.hidden_size

    @override
    def optimizer_targets(self):
        return self.wkv.optimizer_targets()

    def _history(self, ids: torch.Tensor, params: dict):
        """(compressed (bsz, ctx + seq), state layer or None, slots or None)."""
        tm = params.get("token_mask")
        if tm is not None:
            tm = tm.cpu()
        comp = self.hasher.compress(ids, token_mask = tm)
        rsg = params.get("recurrent_states")
        if rsg:
            layer_instance = (self.layer_idx, params.get("layer_instance", 0))
            # Under TP the cache is shipped as an opaque id (model_tp.py:570-571), so the
            # rank resolves its own layer state through the id-keyed lookup. Same idiom as
            # short_conv.py:342, gated_delta_net.py:984, sliding_attn.py:853, mamba2.py:382.
            if self.tp_mode:
                rsl = self.tp_recurrent_lookup[rsg[0].cache]
            else:
                rsl = rsg[0].cache.get_recurrent_layer(layer_instance)
            slots = get_for_device(params, "recurrent_slots", "cpu").tolist()
            assert len(slots) == ids.shape[0] == len(rsg)
            prev = torch.stack([rsl.context(s, int(r.position)) for s, r in zip(slots, rsg)])
            return torch.cat((prev, comp), dim = 1), rsl, slots
        assert params.get("position", 0) == 0, "EngramLayer needs recurrent states for forwards past position 0"
        pad = torch.full((ids.shape[0], self.hasher.context_len), DEAD, dtype = torch.long)
        return torch.cat((pad, comp), dim = 1), None, None

    def forward_streams(self, streams: torch.Tensor, hash_ids: torch.Tensor, params: dict) -> torch.Tensor:
        """streams: (bsz, seq, hc_mult, hidden) fp32; hash_ids: (bsz, seq, n_hash_cols). Returns the
        delta to add to the streams."""
        bsz, seq, H, D = streams.shape
        rows = self.table.rows(hash_ids, streams.device)                       # (bsz, seq, cols, head_dim) fp32
        kv = self.wkv.forward(rows.view(bsz, seq, -1).half(), params)          # (bsz, seq, (H+1) * D) fp16
        key = kv[..., :H * D].float().view(bsz, seq, H, D)
        value = kv[..., H * D:].float()
        h = streams.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * self.qk * key).sum(-1) * rstd * (D ** -0.5)               # (bsz, seq, H)
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        return gate.unsqueeze(-1) * value.unsqueeze(-2)

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        bsz, seq = x.shape[:2]
        ids = params.get("input_ids")
        if ids is None:
            ids = torch.zeros((bsz, seq), dtype = torch.long)               # measuring forward
        ids = ids.to("cpu", torch.int64)
        rsg = params.get("recurrent_states")
        if rsg:
            # Cached forwards carry the absolute position in the recurrent state, not params["position"].
            # Hashing a chunk from position 0 would block every n-gram of its first context_len tokens
            # (positions < shift), so single-token decode hashed unigram-only rows. With a batch of
            # differently advanced sequences the DEAD-filled history before each sequence start blocks
            # the same n-grams, so a start position past the context is exact for all of them
            pos = int(rsg[0].position) if len(rsg) == 1 else self.hasher.context_len
        else:
            pos = int(params.get("position", 0))
        window, rsl, slots = self._history(ids, params)
        hash_ids = self.hasher.hash_window(window, pos)[:, :, self.table_index, :]
        delta = self.forward_streams(x, hash_ids, params)
        tm = params.get("token_mask")
        if tm is not None:
            delta = delta * tm.to(device = delta.device, dtype = delta.dtype).unsqueeze(-1).unsqueeze(-1)
        if rsl is not None:
            ctx = self.hasher.context_len
            for i, (s, r) in enumerate(zip(slots, rsg)):
                rsl.write(s, int(r.position), window[i, ctx:])
        return x + delta

    # ---- tensor-parallel support -------------------------------------------------------
    # Replicated per rank, matching the KV-side replication the attention allocation uses.
    # qk is small and is sent. The EngramTable is NOT: it is the disk-backed n-gram store
    # (~189 GiB) and cannot cross a shared-memory boundary. A TP worker's local_context has
    # no stc/config/path (measured), so the model directory is carried in the export and each
    # rank rebuilds its own collection with Config.from_directory, which reads config.json
    # and safetensors headers only and loads no weights.

    @override
    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."
        _dir = None
        for _o in (self.config, getattr(self.config, "stc", None)):
            if _o is None:
                continue
            for _a in ("directory", "model_dir", "path", "dir", "model_directory"):
                _v = getattr(_o, _a, None)
                if isinstance(_v, str) and _v:
                    _dir = _v
                    break
            if _dir:
                break
        return {
            "cls": EngramLayer,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "table_index": self.table_index,
                "hidden_size": self.hidden_size,
                "hc_mult": self.hc_mult,
                "rms_norm_eps": self.eps,
                "gather_threads": self.gather_threads,
                "out_dtype": self.out_dtype,
            },
            "hasher": self.hasher,
            "wkv": self.wkv.tp_export(plan, producer),
            "qk": producer.send(self.qk),
            "recurrent_layers": [rl.tp_export(plan) for rl in self.recurrent_layers],
            "engram_dir": _dir,
            "config_attrs": sorted(a for a in dir(self.config) if not a.startswith("_"))[:60],
            "stc_attrs": sorted(a for a in dir(getattr(self.config, "stc", object())) if not a.startswith("_"))[:60],
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        device = local_context["device"]
        consumer = local_context["consumer"]
        module = EngramLayer(
            config = None,
            hasher = exported["hasher"],
            **exported["kwargs"],
        )
        module.device = device
        wkv = exported["wkv"]["cls"].tp_import(local_context, exported["wkv"], plan)
        module.wkv = wkv
        module.modules = [wkv]
        module.qk = consumer.recv(exported["qk"], cuda = True)
        _dir = exported.get("engram_dir")
        if not _dir:
            raise NotImplementedError(
                "EngramLayer.tp_import: could not resolve the model directory from the "
                "parent config, so the disk-backed table cannot be opened per rank. "
                "config attrs = %s ; stc attrs = %s"
                % (exported.get("config_attrs"), exported.get("stc_attrs"))
            )
        _cfg = Config.from_directory(_dir)
        module.table = EngramTable(
            _cfg.stc, "%s.embed" % module.key, module.hasher.head_dim, module.gather_threads)
        module._tp_cfg_ref = _cfg   # keep the collection alive for the table's handles
        for rl in exported["recurrent_layers"]:
            rli = rl["cls"](module, **rl["args"])
            # A worker module is built by tp_import and load() is never called on it, so
            # the per-rank state EngramLayer.load() would allocate (:467-468) must be
            # allocated here: id_state starts on "meta" (:347) and only alloc() gives it
            # real storage (:360-363). The peer modules get this from the
            # module.load_local(device) at the end of their tp_import (dsv4.py:898,
            # gated_delta_net.py:1334, sliding_attn.py:1256, mamba2.py:719); EngramLayer
            # has no load_local, so it is explicit.
            rli.alloc(device)
            module.recurrent_layers.append(rli)
            module.tp_recurrent_lookup[rl["args"]["cache_id"]] = rli
        module.tp_mode = True
        return module
