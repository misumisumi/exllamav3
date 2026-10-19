#!/usr/bin/env python3
"""
Test GPU-direct ATS path for Engram tables.

Run on GB10 with a DeepSeek-V4.1 model:
    EXL3_ATS_MMAP=1 python engram_ats_check.py /path/to/model

The script loads an engram table twice: once via disk path (EXL3_ENGRAM_ATS=0)
and once via GPU-direct ATS alias (EXL3_ENGRAM_ATS=1), and verifies the results match.
"""

import os
import sys
import torch

# Must set EXL3_ATS_MMAP before importing exllamav3
os.environ.setdefault("EXL3_ATS_MMAP", "1")

from exllamav3.model.config import Config
from exllamav3.modules.engram import EngramTable


def find_engram_table_key(config):
    """Find the first engram table key in the tensor file map."""
    for key in config.stc.tensor_file_map.keys():
        if ".engram.embed.weight" in key:
            return key
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: engram_ats_check.py <model_directory>", file = sys.stderr)
        sys.exit(1)

    model_dir = sys.argv[1]
    config = Config.from_directory(model_dir)

    # Find engram table key
    weight_key = find_engram_table_key(config)
    if weight_key is None:
        print("ENGRAM_ATS_SKIP: no engram tables found")
        return

    # Extract layer index and get head_dim
    table_key = weight_key.replace(".weight", "")
    head_dim = config.stc.get_tensor_handle(weight_key).row_shape[0] * \
               config.stc.get_tensor_handle(weight_key).row_shape[1] if len(config.stc.get_tensor_handle(weight_key).row_shape) > 1 \
               else config.stc.get_tensor_handle(weight_key).row_shape[0]

    # Create seeded random row ids: [2, 2048]
    g = torch.Generator()
    g.manual_seed(42)
    num_rows = config.stc.get_tensor_handle(weight_key).num_rows
    ids_base = torch.randint(0, num_rows, (4096,), generator = g)
    # Include boundary cases
    ids_base[0] = 0
    ids_base[1] = num_rows - 1
    ids = ids_base.view(2, 2048)

    device = torch.device("cuda:0")

    # Disk path (EXL3_ENGRAM_ATS=0)
    os.environ["EXL3_ENGRAM_ATS"] = "0"
    table_disk = EngramTable(config.stc, table_key, head_dim)
    result_disk = table_disk.rows(ids, device)
    table_disk.close()

    # GPU ATS path (EXL3_ENGRAM_ATS=1)
    os.environ["EXL3_ENGRAM_ATS"] = "1"
    table_ats = EngramTable(config.stc, table_key, head_dim)
    result_ats = table_ats.rows(ids, device)
    table_ats.close()

    # Compare
    if torch.equal(result_disk, result_ats):
        print("ENGRAM_ATS_OK")
    else:
        diff = (result_disk - result_ats).abs().max().item()
        print(f"max_abs_diff={diff:.2e} ENGRAM_ATS_FAIL")


if __name__ == "__main__":
    main()
