"""Reported experimental settings for the Secure IDS study.

The executable pipeline exposes these settings as command-line options. This
file provides a compact, human-readable record of the reported configuration.
"""

SEEDS = [42, 43, 44, 45, 46]
TEST_SIZE = 0.30
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

ATTACK_MAX_SAMPLES = 20_000
ATTACK_EPOCHS = 10
SHADOW_MODELS = 2

DATASET_HINTS = {
    "CICIDS2017": "A filename containing 'cicids2017' activates CICIDS safeguards.",
    "IoT Network Intrusion": "Retain Label, Cat, and Sub_Cat when available.",
    "NSL/NNSL-KDD": "A filename containing 'nsl-kdd' or 'nnsl-kdd' activates NSL-KDD safeguards.",
    "train_test_network": "Retain the binary target column and any detailed type column.",
}
