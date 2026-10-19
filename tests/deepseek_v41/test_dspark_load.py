#!/usr/bin/env python3
"""
Test loading V4.1 model with drafter.
"""

import sys
import os

# Add the repo to path
sys.path.insert(0, os.path.expanduser("~/tp1/src/exl3-dspark"))

def main():
    model_path = os.path.expanduser("~/tp1/v41port/model")

    # Set environment before import
    os.environ.setdefault("EXL3_MOE_CPU_SPLIT", "256")
    os.environ.setdefault("EXL3_MOE_CPU_THREADS", "16")
    os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
    os.environ.setdefault("EXL3_MOE_STREAM_MIN_ROWS", "1000000000")

    import torch
    from exllamav3 import Config, Model, Tokenizer

    print(f"Loading model from {model_path}...")
    config = Config.from_directory(model_path)

    print(f"Config arch: {config.arch_string}")
    print(f"Model classes: {list(config.model_classes.keys())}")
    print(f"DSpark block size: {config.dspark_block_size}")
    print(f"DSpark target layer IDs: {config.dspark_target_layer_ids}")
    print(f"Num MTP layers: {config.num_mtp_layers}")

    model = Model.from_config(config, component="text")
    print(f"Model loaded: {model.__class__.__name__}")

    # Load drafter if available
    if "mtp" in config.model_classes:
        print("✓ Loading drafter model...")
        drafter = Model.from_config(config, component="mtp")
        print(f"  Drafter class: {drafter.__class__.__name__}")
        print(f"  Drafter blocks: {drafter.num_mtp_layers}")
        print(f"  Block size: {drafter.input_layer.block_size}")
        if hasattr(drafter, 'markov_embed'):
            print(f"  Markov embed module: {drafter.markov_embed.__class__.__name__}")
    else:
        print("✗ No drafter model in config")

if __name__ == "__main__":
    main()
