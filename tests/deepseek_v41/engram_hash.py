"""DeepSeek-V4.1 Engram n-gram hashing, engine-side re-implementation.

Mirrors the reference (inference/engram.py): a tokenizer-derived compressed vocab, one
prime-sized bucket range per (engram layer, n-gram size, head) drawn as the first
primes above engram_vocab_size - 1 and never reused, odd per-(layer, lookback)
multipliers from numpy's default_rng(10007 * layer_id), and the XOR-of-products hash of
the n = 2..max_ngram_size tokens ending at each position. Needs only `tokenizers`,
numpy and torch, so an engine can build it from tokenizer.json alone. The compressed
token map is cached on disk next to the tokenizer because building it decodes every
token once (~129k decodes).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

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


def find_next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer_json: str | Path, *, cache_dir: str | Path | None = None) -> tuple[np.ndarray, int]:
    """Compressed ids per token id (tokens that normalize alike collapse), from tokenizer.json only."""
    from tokenizers import Regex, Tokenizer, normalizers

    tokenizer_json = Path(tokenizer_json)
    cache_dir = Path(cache_dir) if cache_dir is not None else tokenizer_json.parent
    digest = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()[:16]
    cache = cache_dir / f"engram_token_map.{digest}.npz"
    if cache.is_file():
        z = np.load(cache)
        return z["token_map"], int(z["vocab_size"])
    backend = Tokenizer.from_file(str(tokenizer_json))
    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    n = backend.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = np.zeros(n, dtype=np.int64)
    for token_id in range(n):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            # a partial UTF-8 byte token: nothing to normalize, key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    try:
        np.savez(cache, token_map=lookup, vocab_size=np.int64(len(key_to_new)))
    except OSError:
        pass
    return lookup, len(key_to_new)


def compute_hash_multipliers(layer_ids, max_ngram_size: int, compressed_vocab_size: int) -> torch.Tensor:
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        g = np.random.default_rng(10007 * layer_id)
        values = g.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


def engram_primes(layer_ids, max_ngram_size: int, n_heads: int, engram_vocab_size: int) -> list[list[list[int]]]:
    primes, seen = [], set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, current = [], engram_vocab_size - 1
            for _ in range(n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    return primes


class EngramHasher:
    """Stateless hash of the n-grams ending at each position, given the preceding tokens.

    `forward(ids)` takes [B, L] compressed-or-raw token ids of a contiguous window whose
    first `max_ngram_size - 1` entries are the history (pad_id where the sequence starts,
    DEAD for image tokens) and returns hash ids [B, L - history, n_layers, n_cols].
    """

    def __init__(self, text_config: dict, tokenizer_json: str | Path, *, cache_dir=None, device="cpu"):
        c = text_config
        self.layer_ids = tuple(c["engram_layer_ids"])
        self.max_ngram = int(c["engram_max_ngram_size"])
        self.n_heads = int(c["engram_n_heads"])
        self.head_dim = int(c["engram_head_dim"])
        self.num_embeddings = tuple(c["engram_num_embeddings"])
        token_map, vocab_size = build_compressed_token_map(tokenizer_json, cache_dir=cache_dir)
        if vocab_size != int(c["engram_compressed_vocab_size"]):
            raise ValueError(f"compressed vocab {vocab_size} != config {c['engram_compressed_vocab_size']}")
        self.pad_id = int(token_map[int(c["engram_pad_token_id"])])
        primes = engram_primes(self.layer_ids, self.max_ngram, self.n_heads, int(c["engram_vocab_size"]))
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
        for layer, sizes, rows in zip(self.layer_ids, flat, self.num_embeddings):
            if sum(sizes) != rows:
                raise ValueError(f"engram layer {layer}: primes sum to {sum(sizes)}, table has {rows} rows")
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.primes = torch.tensor(primes, dtype=torch.int64, device=device)          # [n_layers, max_ngram-1, n_heads]
        self.offsets = torch.tensor(np.array(offsets), dtype=torch.int64, device=device)  # [n_layers, n_cols]
        self.multipliers = compute_hash_multipliers(self.layer_ids, self.max_ngram, vocab_size).to(device)
        self.token_map = torch.tensor(token_map, dtype=torch.int64, device=device)
        self.n_cols = (self.max_ngram - 1) * self.n_heads

    def compress(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, torch.full_like(compressed, DEAD))
        return compressed

    def hash_window(self, compressed: torch.Tensor, start_pos: int) -> torch.Tensor:
        """compressed: [B, H + L] where H = max_ngram - 1 history slots (DEAD or any value when
        start_pos - H + j < 0, they are replaced by pad); returns [B, L, n_layers, n_cols]."""
        h = self.max_ngram - 1
        B, HL = compressed.shape
        L = HL - h
        positions = torch.arange(start_pos, start_pos + L, device=compressed.device).expand(B, L)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.max_ngram):
            # slot for position p - shift lives at window index (p - start_pos) + h - shift
            source = compressed[:, h - shift: h - shift + L]
            blocked = blocked | (positions < shift) | (source == DEAD)
            tokens.append(torch.where(blocked, torch.full_like(source, self.pad_id), source))
        tokens = torch.stack(tokens, dim=-1)                       # [B, L, max_ngram]
        products = tokens.unsqueeze(2) * self.multipliers           # [B, L, n_layers, max_ngram]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


def reference_check(text_config: dict, tokenizer_json: str, tokenizer_dir: str, reference_dir: str, n_seq=3, L=257, seed=0):
    """Compare against the reference NgramHashState on random sequences; returns max abs diff."""
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, reference_dir)
    from engram import EngramLayout, NgramHashState  # reference
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    c = text_config
    args = SimpleNamespace(engram_layer_ids=c["engram_layer_ids"], engram_max_ngram_size=c["engram_max_ngram_size"],
                           engram_n_heads=c["engram_n_heads"], engram_vocab_size=c["engram_vocab_size"],
                           engram_num_embeddings=c["engram_num_embeddings"], engram_head_dim=c["engram_head_dim"],
                           engram_compressed_vocab_size=c["engram_compressed_vocab_size"], engram_pad_id=c["engram_pad_token_id"],
                           max_batch_size=n_seq, max_seq_len=L + 64)
    ref = NgramHashState(args, EngramLayout.from_args(args), tok)
    mine = EngramHasher(c, tokenizer_json)
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, int(c["vocab_size"]), (n_seq, L), generator=g)
    ids[1, 40:60] = int(c["engram_pad_token_id"])
    # prefill: whole sequence from position 0
    r_full = ref(ids, 0)
    m_full = mine.hash_window(torch.cat([torch.full((n_seq, mine.max_ngram - 1), DEAD, dtype=torch.int64), mine.compress(ids)], 1), 0)
    d_full = (r_full - m_full).abs().max().item()
    # prefill 200 then decode one token at a time with history windows
    ref2 = NgramHashState(args, EngramLayout.from_args(args), tok)
    r_pre = ref2(ids[:, :200], 0)
    comp = mine.compress(ids)
    m_pre = mine.hash_window(torch.cat([torch.full((n_seq, mine.max_ngram - 1), DEAD, dtype=torch.int64), comp[:, :200]], 1), 0)
    d_pre = (r_pre - m_pre).abs().max().item()
    d_dec = 0.0
    for p in range(200, L):
        r_step = ref2(ids[:, p:p + 1], p)
        m_step = mine.hash_window(comp[:, p - (mine.max_ngram - 1): p + 1], p)
        d_dec = max(d_dec, (r_step - m_step).abs().max().item())
    # image-token dead spans
    mask = torch.ones_like(ids, dtype=torch.bool); mask[0, 100:110] = False
    r_mask = ref(ids, 0, mask)
    m_mask = mine.hash_window(torch.cat([torch.full((n_seq, mine.max_ngram - 1), DEAD, dtype=torch.int64), mine.compress(ids, mask)], 1), 0)
    d_mask = (r_mask - m_mask).abs().max().item()
    return {"full_prefill": d_full, "prefill_200": d_pre, "decode_steps": d_dec, "dead_span": d_mask,
            "pad_id": mine.pad_id, "ref_pad_id": int(ref.pad_id), "shape": list(m_full.shape),
            "max_hash": int(m_full.max()), "rows": mine.num_embeddings}


if __name__ == "__main__":
    import sys
    cfg = json.load(open(sys.argv[1]))["text_config"]
    print(json.dumps(reference_check(cfg, sys.argv[2], sys.argv[3], sys.argv[4]), indent=1))
