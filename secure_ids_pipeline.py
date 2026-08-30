# ============================================================
# Reviewer-revision Secure IDS experimental pipeline
# Google Colab + GitHub-ready, multi-dataset reviewer-experiment version
# ============================================================
# IMPORTANT:
# 1) Put all CSV datasets in one folder (Colab /content, repository data/,
#    or a directory supplied with --data-dir).
# 2) Every supported CSV in that folder is detected automatically.
# 3) It does NOT invent or overwrite experimental results.
# 4) It saves all outputs in results/.
# 5) CNN-BiLSTM + MLP require TensorFlow (preinstalled in Google Colab).
# 6) The original IDS result path is unchanged. Reconstruction, shadow-model
#    membership inference, attribute inference, and five-case ablation are
#    added after the original per-seed metrics are computed.
#
# Tested layouts include CICIDS2017, Bot-IoT/IoT Network Intrusion,
# NSL/NNSL-KDD numeric columns, and train_test_network.
# ============================================================

# ---------- 1. Install / import ----------
# Packages are already available in this Colab environment

import os, gc, json, time, hashlib, shutil, random, warnings, subprocess, getpass, argparse
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import psutil
except ImportError:
    psutil = None
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_IMPORT_ERROR = None
except ImportError as exc:
    tf = keras = layers = None
    TENSORFLOW_IMPORT_ERROR = exc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

try:
    from IPython.display import display
except ImportError:
    display = print

warnings.filterwarnings("ignore")

# Reproducibility
SEEDS = [42, 43, 44, 45, 46]
TEST_SIZE = 0.30
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

# Google Drive persistence. All intermediate/final outputs are written here.
DRIVE_PROJECT_NAME = "Secure_IDS_Reviewer_Attacks_v5"

# GitHub settings. Change the repository name if desired.
ENABLE_GITHUB_UPLOAD = False
GITHUB_REPO = "secure-ids-all-datasets-results"
GITHUB_PRIVATE = True

LABEL_CANDIDATES = [
    "label", "Label", "LABEL",
    "class", "Class", "CLASS",
    "target", "Target",
    "binary_label", "Binary_Label",
]

# Columns whose names commonly identify a target.  The lookup is normalized,
# so variants such as ``Attack Type`` and ``attack_type`` are both accepted.
NORMALIZED_LABEL_NAMES = {
    "label", "labels", "class", "target", "binarylabel", "attacklabel",
    "trafficlabel", "attacktype", "attackbinary", "binaryattack",
    "outcome", "intrusion", "isattack",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Secure IDS pipeline on every supported CSV dataset."
    )
    parser.add_argument(
        "--data-dir", default=os.environ.get("SECURE_IDS_DATA_DIR"),
        help="Directory containing CSV datasets (default: auto-detect)."
    )
    parser.add_argument(
        "--results-dir", default=os.environ.get("SECURE_IDS_RESULTS_DIR"),
        help="Output directory. In Colab, an already-mounted Drive is used automatically."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--seeds", default=",".join(map(str, SEEDS)),
        help="Comma-separated random seeds, e.g. 42,43,44,45,46."
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Optional stratified row limit per dataset for a quick/test run."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only detect, load and audit datasets; do not train models."
    )
    parser.add_argument(
        "--copy-datasets-to-drive", action="store_true",
        help="Copy CSV files from --data-dir into the project's Google Drive datasets folder."
    )
    parser.add_argument(
        "--github-upload", action="store_true",
        help="After completion, create/update the configured GitHub repository."
    )
    parser.add_argument(
        "--skip-reviewer-experiments", action="store_true",
        help="Run only the unchanged original pipeline and skip the four added reviewer experiments."
    )
    parser.add_argument(
        "--attack-max-samples", type=int, default=20000,
        help="Maximum held-out samples used by each privacy attacker (default: 20000)."
    )
    parser.add_argument(
        "--attack-epochs", type=int, default=10,
        help="Epochs for reconstruction and shadow-model attackers (default: 10)."
    )
    parser.add_argument(
        "--shadow-models", type=int, default=2,
        help="Number of independently split shadow MLP models (default: 2)."
    )
    parser.add_argument(
        "--attribute-column", default=None,
        help=(
            "Optional exact sensitive attribute column. By default the code selects "
            "a detailed attack category, subtype, protocol, service, or device field "
            "and records the selected column."
        )
    )
    return parser.parse_args()

def is_nsl_kdd_numeric_file(filename):
    name = Path(filename).name.lower().replace("_", "-")
    return "nsl-kdd" in name or "nnsl-kdd" in name

def _normalize_header_name(name):
    """Aggressively normalize odd CSV headers (BOM, spaces, punctuation)."""
    import re
    s = str(name).replace("\ufeff", "").replace("ï»¿", "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", s)

def _cicids_semantic_label_candidates(df):
    """Find CICIDS target columns from values when the header itself is corrupted/nonstandard."""
    benign_tokens = {"benign", "normal", "normaltraffic", "normal_traffic"}
    candidates = []
    for c in df.columns:
        s = pd.Series(df[c]).dropna()
        if s.empty:
            continue
        # Target columns should have comparatively few classes; this avoids scanning IDs/continuous data.
        n_unique = s.nunique(dropna=True)
        if n_unique < 2 or n_unique > 100:
            continue
        vals = s.astype(str).str.strip().str.lower().unique().tolist()
        compact = {v.replace(" ", "").replace("-", "").replace("_", "") for v in vals}
        has_benign = any(v in {"benign", "normal", "normaltraffic"} for v in compact)
        if has_benign and len(compact) >= 2:
            candidates.append(c)
    return candidates

def resolve_label_column(df, filename):
    """Resolve a target safely across common IDS CSV layouts."""
    fname = Path(filename).name.lower()

    # Supplied NSL-KDD numeric layout: 41=attack type, 42=difficulty.
    if is_nsl_kdd_numeric_file(filename) and "41" in df.columns:
        return "41"

    named = [c for c in df.columns if _normalize_header_name(c) in NORMALIZED_LABEL_NAMES]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        # Prefer an actually binary column (e.g. Bot-IoT has Label, Cat, Sub_Cat).
        binary = [c for c in named if pd.Series(df[c]).dropna().nunique() == 2]
        if len(binary) == 1:
            return binary[0]
        raise ValueError(
            f"{filename}: ambiguous target columns {named}. "
            "Keep one target column or rename the intended one to 'label'."
        )

    # CICIDS2017 fallback: detect the target by BENIGN/normal semantics.
    if "cicids2017" in fname:
        # Fallback: detect the target by BENIGN/normal semantics in the column values.
        semantic = _cicids_semantic_label_candidates(df)
        if len(semantic) == 1:
            print(f"CICIDS2017 target resolved from values: {repr(semantic[0])}")
            return semantic[0]
        if len(semantic) > 1:
            raise ValueError(
                f"{filename}: multiple CICIDS2017 target-like columns found by values: {semantic}. "
                "Refusing to guess."
            )

        # Diagnostic details make malformed/preprocessed files obvious.
        tail = list(df.columns[-15:])
        low_card = []
        for c in df.columns:
            try:
                nu = df[c].nunique(dropna=True)
                if 2 <= nu <= 20:
                    low_card.append((str(c), int(nu), df[c].dropna().astype(str).unique()[:6].tolist()))
            except Exception:
                pass
        raise ValueError(
            f"{filename}: no safe CICIDS2017 label column found. "
            f"Total columns={len(df.columns)}. Last columns={tail}. "
            f"Low-cardinality candidates={low_card[:15]}. "
            "The supplied CSV may have been exported without its target column."
        )

    # Generic named labels: prefer truly binary columns.
    for candidate in LABEL_CANDIDATES:
        if candidate in df.columns:
            vals = pd.Series(df[candidate]).dropna().unique()
            if len(vals) == 2:
                return candidate

    wanted = NORMALIZED_LABEL_NAMES
    for c in df.columns:
        if _normalize_header_name(c) in wanted:
            vals = pd.Series(df[c]).dropna().unique()
            if len(vals) == 2:
                return c

    raise ValueError(
        f"{filename}: could not safely resolve a label column. "
        f"Available columns include: {list(df.columns)[:50]}"
    )

def _find_first_case_insensitive(directory, exact_name):
    directory = Path(directory)
    exact_lower = exact_name.lower()
    for p in directory.glob("*.csv"):
        if p.name.lower() == exact_lower:
            return p
    return None

def _default_data_directory():
    """Choose a dataset folder usable in Colab, GitHub Actions, and local runs."""
    candidates = []
    drive_datasets = Path("/content/drive/MyDrive") / DRIVE_PROJECT_NAME / "datasets"
    if drive_datasets.is_dir():
        candidates.append(drive_datasets)
    if Path("/content").is_dir():
        candidates.append(Path("/content"))
    candidates.extend([Path.cwd() / "data", Path.cwd() / "datasets", Path.cwd()])
    try:
        script_parent = Path(__file__).resolve().parent
        candidates.extend([script_parent / "data", script_parent / "datasets", script_parent])
    except NameError:
        pass
    seen = set()
    for directory in candidates:
        key = str(directory.resolve()) if directory.exists() else str(directory)
        if key not in seen and directory.is_dir() and any(directory.glob("*.csv")):
            return directory
        seen.add(key)
    return Path.cwd()

def copy_csv_datasets_to_drive(source_dir):
    """Persist uploaded Colab CSV files in the project's Google Drive folder."""
    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.is_dir():
        raise RuntimeError(
            "Google Drive is not mounted. Run drive.mount('/content/drive') "
            "in a Colab notebook cell before starting this script."
        )
    source = Path(source_dir).expanduser().resolve() if source_dir else _default_data_directory()
    files = sorted(source.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found to copy from {source}")
    destination = drive_root / DRIVE_PROJECT_NAME / "datasets"
    destination.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = destination / src.name
        if src.resolve() != dst.resolve():
            print(f"Copying dataset to Google Drive: {src.name}")
            shutil.copy2(src, dst)
    print(f"Datasets stored in Google Drive: {destination}")
    return destination

def build_dataset_config(data_dir=None):
    """Discover every top-level CSV dataset instead of requiring fixed filenames."""
    directory = Path(data_dir).expanduser().resolve() if data_dir else _default_data_directory()
    csv_files = sorted(directory.glob("*.csv"), key=lambda p: p.name.lower())
    # Do not accidentally retrain on outputs if data and result paths overlap.
    output_names = {
        "dataset_audit.csv", "all_runs_raw.csv", "confusion_matrices.csv",
        "training_history.csv", "results_mean.csv", "results_std.csv",
        "results_mean_plus_std.csv", "ablation_reconstruction.csv",
        "reconstruction_attack_results.csv",
        "membership_inference_results.csv",
        "attribute_inference_results.csv",
        "ablation_study_results.csv",
    }
    csv_files = [p for p in csv_files if p.name not in output_names and "_seed_" not in p.name]
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV datasets found in {directory}. Use --data-dir /path/to/csv/files."
        )

    configs = {}
    used_names = set()
    for p in csv_files:
        stem = p.stem
        # Remove browser/Colab copy suffixes such as (1), (2), ... for display only.
        import re
        display_name = re.sub(r"\s*\(\d+\)$", "", stem).strip() or stem
        base_name = display_name
        counter = 2
        while display_name.lower() in used_names:
            display_name = f"{base_name}_{counter}"
            counter += 1
        used_names.add(display_name.lower())
        cfg = {"file": str(p)}
        if is_nsl_kdd_numeric_file(p.name):
            cfg["format"] = "nsl_kdd_numeric"
        elif "cicids2017" in p.name.lower():
            cfg["format"] = "cicids2017"
        configs[display_name] = cfg

    print(f"\nResolved {len(configs)} dataset(s) in {directory}:")
    for name, cfg in configs.items():
        print(f"  {name}: {cfg['file']}")
    return configs

def initialize_persistent_storage(explicit_results_dir=None):
    """Select storage without calling Colab's interactive mount from a subprocess."""
    if explicit_results_dir:
        results_dir = Path(explicit_results_dir).expanduser().resolve()
        project_dir = results_dir.parent
    else:
        drive_root = Path("/content/drive/MyDrive")
        if drive_root.is_dir():
            project_dir = drive_root / DRIVE_PROJECT_NAME
            results_dir = project_dir / "results"
            print(f"Using mounted Google Drive: {project_dir}")
        else:
            project_dir = Path.cwd() / DRIVE_PROJECT_NAME
            results_dir = project_dir / "results"
            if os.environ.get("COLAB_RELEASE_TAG"):
                print(
                    "NOTE: Google Drive is not mounted. Results will be saved locally. "
                    "To persist them, run drive.mount('/content/drive') in a notebook cell first."
                )
            print(f"Using project directory: {project_dir}")

    backups_dir = project_dir / "backups"
    code_dir = project_dir / "code"
    results_dir.mkdir(parents=True, exist_ok=True)
    backups_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    return project_dir, results_dir, backups_dir, code_dir

PROJECT_DIR = RESULTS_DIR = BACKUPS_DIR = CODE_DIR = None

def backup_running_code():
    """Keep a copy of this exact executable script in Google Drive."""
    try:
        source = Path(__file__).resolve()
        destination = CODE_DIR / source.name
        shutil.copy2(source, destination)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy2(source, BACKUPS_DIR / f"{source.stem}_{stamp}{source.suffix}")
        print(f"Code backup saved: {destination}")
    except Exception as exc:
        print(f"WARNING: code backup failed: {exc}")

# ---------- 2. Upload datasets ----------

# ---------- 3. Dataset-specific configuration ----------

# ---------- 4. Utilities ----------


# =============================================================================
# STEP 79 - LEAKAGE-SAFE GROUP-AWARE SPLITTING
# =============================================================================

def leakage_safe_group_split(
    df,
    label_col,
    test_size=0.30,
    random_state=42,
):
    """
    Leakage-safe group split.

    Identical complete feature vectors are assigned to the same split.
    Returns ONLY positional NumPy int64 indices in the range [0, len(df)-1].
    """

    from sklearn.model_selection import GroupShuffleSplit

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in dataframe."
        )

    feature_cols = [c for c in df.columns if c != label_col]

    if not feature_cols:
        raise ValueError("No feature columns available.")

    feature_frame = df[feature_cols].copy().reset_index(drop=True)

    feature_frame = feature_frame.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Stable duplicate-group identifiers.
    group_hash = pd.util.hash_pandas_object(
        feature_frame,
        index=False
    ).to_numpy(dtype=np.uint64)

    # Convert hash values to compact integer group IDs.
    _, group_codes = np.unique(
        group_hash,
        return_inverse=True
    )

    group_codes = np.asarray(
        group_codes,
        dtype=np.int64
    )

    n_rows = len(feature_frame)

    if len(group_codes) != n_rows:
        raise RuntimeError(
            "GROUP ALIGNMENT FAILURE: group count does not match dataframe rows."
        )

    row_positions = np.arange(
        n_rows,
        dtype=np.int64
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, test_idx = next(
        splitter.split(
            row_positions,
            groups=group_codes
        )
    )

    train_idx = np.asarray(
        train_idx,
        dtype=np.int64
    ).reshape(-1)

    test_idx = np.asarray(
        test_idx,
        dtype=np.int64
    ).reshape(-1)

    if train_idx.size == 0 or test_idx.size == 0:
        raise RuntimeError(
            "GROUP SPLIT FAILURE: empty train or test split."
        )

    if train_idx.min() < 0 or test_idx.min() < 0:
        raise RuntimeError(
            "INDEX RANGE FAILURE: negative index detected."
        )

    if train_idx.max() >= n_rows or test_idx.max() >= n_rows:
        raise RuntimeError(
            f"INDEX RANGE FAILURE: n_rows={n_rows}, "
            f"train_max={train_idx.max()}, "
            f"test_max={test_idx.max()}"
        )

    train_groups = set(
        group_codes[train_idx].tolist()
    )

    test_groups = set(
        group_codes[test_idx].tolist()
    )

    crossing_groups = train_groups.intersection(
        test_groups
    )

    print("\n===== LEAKAGE-SAFE SPLIT =====")
    print(f"Seed              : {random_state}")
    print(f"Original rows     : {n_rows:,}")
    print(f"Train rows        : {len(train_idx):,}")
    print(f"Test rows         : {len(test_idx):,}")
    print(
        f"Train percentage  : "
        f"{100.0 * len(train_idx) / n_rows:.4f}%"
    )
    print(
        f"Test percentage   : "
        f"{100.0 * len(test_idx) / n_rows:.4f}%"
    )
    print(f"Crossing groups   : {len(crossing_groups)}")

    if crossing_groups:
        raise RuntimeError(
            "LEAKAGE SAFETY FAILURE: duplicate feature groups crossed "
            "Train/Test boundary."
        )

    print("OK: NO duplicate feature group crosses Train/Test.")

    print("\nIndex dtype verification:")
    print(f"train_idx dtype   : {train_idx.dtype}")
    print(f"test_idx dtype    : {test_idx.dtype}")
    print(f"train_idx type    : {type(train_idx).__name__}")
    print(f"test_idx type     : {type(test_idx).__name__}")

    return train_idx, test_idx


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)

def require_training_dependencies():
    if tf is None:
        raise ImportError(
            "TensorFlow is required for training. Install dependencies with: "
            "pip install tensorflow pandas numpy scikit-learn psutil cryptography "
            "matplotlib seaborn"
        ) from TENSORFLOW_IMPORT_ERROR

def clean_dataframe(df):
    df = df.copy()

    # Normalize column names to strings
    df.columns = [str(c).strip() for c in df.columns]

    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop fully empty rows
    df = df.dropna(how="all")

    # Remove exact duplicate records
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before - len(df)

    return df, duplicates_removed

def encode_binary_label(series):
    """
    Convert a binary label into 0/1 without assuming that
    the lexical order represents the attack/benign meaning.
    The mapping is saved for auditability.
    """
    s = series.copy()

    # Numeric binary labels
    if pd.api.types.is_numeric_dtype(s):
        vals = sorted(pd.Series(s.dropna()).unique().tolist())
        if len(vals) != 2:
            raise ValueError(f"Expected binary numeric label, found: {vals}")
        mapping = {vals[0]: 0, vals[1]: 1}
        y = s.map(mapping)
        return y.astype(int), mapping

    # String labels
    s = s.astype(str).str.strip()

    # Common benign labels
    benign_tokens = {
        "0", "normal", "benign", "normal traffic", "normal_traffic",
        "non-attack", "non_attack", "nonattack", "background"
    }

    vals = sorted(s.dropna().unique().tolist())

    # Prefer semantic mapping when possible
    lower = {v: v.lower() for v in vals}
    benign = [v for v in vals if lower[v] in benign_tokens]

    if len(benign) == 1:
        benign_val = benign[0]
        mapping = {v: (0 if v == benign_val else 1) for v in vals}
        y = s.map(mapping)
        return y.astype(int), mapping
    if len(vals) != 2:
        raise ValueError(
            f"Expected a binary target, or a multiclass attack target containing a "
            f"normal/benign class; found {len(vals)} classes: {vals[:20]}"
        )
    else:
        # Deterministic fallback; mapping is recorded in results.
        benign_val, attack_val = vals[0], vals[1]

    mapping = {benign_val: 0, attack_val: 1}
    y = s.map(mapping)

    if y.isna().any():
        raise ValueError("Label encoding produced missing values.")

    return y.astype(int), mapping

def audit_direct_target_leakage(df, label_col, max_unique=100):
    """
    Detect suspicious low/medium-cardinality columns that deterministically
    map to the binary target. This is an audit safeguard; detected columns are
    reported so they can be excluded before model training.
    """
    if label_col not in df.columns:
        return []

    y = df[label_col]
    suspicious = []

    for c in df.columns:
        if c == label_col:
            continue

        s = df[c]
        nunique = s.nunique(dropna=True)
        if nunique == 0 or nunique > max_unique:
            continue

        try:
            tmp = pd.DataFrame({"x": s, "y": y}).dropna()
            if tmp.empty:
                continue

            # If every feature value maps to exactly one target class, the
            # column can reveal the target directly (classic target leakage).
            mapping_cardinality = tmp.groupby("x", dropna=False)["y"].nunique()
            if len(mapping_cardinality) > 0 and int(mapping_cardinality.max()) == 1:
                suspicious.append(c)
        except Exception:
            pass

    return suspicious


def numeric_features(df, label_col):
    """
    Build model features without target leakage.

    For NNSL-KDD:
    - Keep numerical features.
    - One-hot encode categorical features.
    - Exclude only the target column.
    - Do not use attack-type labels or target-derived categories.
    """

    # Exclude target-derived columns and identifiers that can create leakage
    # or let the model memorize hosts/flows instead of learning traffic behavior.
    # Matching is case-insensitive so dataset naming variants are handled safely.
    leakage_cols = audit_direct_target_leakage(df, label_col)
    if leakage_cols:
        print(f"WARNING: direct target-leakage columns detected and excluded: {leakage_cols}")

    excluded_names_ci = {
        str(label_col).strip().lower(),
        "type",
        "attack_type",
        "attack",
        "class",
        "category",
        "cat",
        "sub_cat",
        "subcategory",
        "timestamp",
        "flow_id",
        "src_ip",
        "dst_ip",
        "source_ip",
        "destination_ip",
    }

    excluded_cols = {
        c for c in df.columns
        if str(c).strip().lower() in excluded_names_ci
    }
    excluded_cols.update(leakage_cols)

    if excluded_cols:
        print("Excluded feature columns:", sorted(map(str, excluded_cols)))

    feature_cols = [
        c for c in df.columns
        if c not in excluded_cols
    ]

    X = df[feature_cols].copy()

    # Identify categorical columns.
    object_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Drop high-cardinality text instead of creating an unbounded one-hot matrix
    # (IP addresses, URIs, certificates and identifiers can exhaust Colab RAM).
    high_cardinality = [
        c for c in object_cols
        if X[c].nunique(dropna=True) > 100
        or X[c].nunique(dropna=True) > max(20, int(0.02 * len(X)))
    ]
    if high_cardinality:
        print("High-cardinality text columns excluded:", list(map(str, high_cardinality)))
        X = X.drop(columns=high_cardinality)
        object_cols = [c for c in object_cols if c not in high_cardinality]

    # One-hot encode only bounded categorical features.
    if object_cols:
        X = pd.get_dummies(
            X,
            columns=object_cols,
            dummy_na=True
        )

    # Convert everything to numeric.
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace infinite values.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove columns that are completely missing.
    all_nan_cols = X.columns[X.isna().all()].tolist()

    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    # Float32 reduces RAM usage.
    X = X.astype(np.float32)

    constant_cols = X.columns[X.nunique(dropna=False) <= 1].tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)

    return X, all_nan_cols + constant_cols

def build_cnn_bilstm(input_shape):
    """
    CNN-BiLSTM feature extractor built with the Keras Functional API.
    Functional construction avoids graph-tracking issues when extracting
    the intermediate deep_features representation.
    """
    inputs = keras.Input(shape=input_shape, name="input_features")

    x = layers.Conv1D(
        32, 3, padding="same", activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True)
    )(x)

    x = layers.GlobalAveragePooling1D()(x)

    deep_features = layers.Dense(
        64,
        activation="relu",
        name="deep_features"
    )(x)

    outputs = layers.Dropout(0.2)(deep_features)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_BiLSTM_Extractor"
    )

def build_mlp(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ], name="MLP_Classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def metrics_from_predictions(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()

    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "test_samples": int(len(y_true))
    }


# =============================================================================
# ADD-ON REVIEWER EXPERIMENTS
# These functions do not replace or alter the original IDS pipeline. They run
# after the original per-seed metrics have been computed and use the same split.
# =============================================================================

def _attribute_equivalent_to_binary_label(values, binary_y):
    values = pd.Series(values).astype(str).reset_index(drop=True)
    binary_y = pd.Series(binary_y).astype(int).reset_index(drop=True)
    if values.nunique(dropna=False) != 2:
        return False
    table = pd.DataFrame({"attribute": values, "y": binary_y})
    return bool(table.groupby("attribute")["y"].nunique().max() == 1)


def select_attribute_inference_target(
    df, label_col, raw_target, binary_y, filename, requested=None
):
    """Choose a real secondary categorical attribute without inventing a label."""
    normalized_columns = {
        _normalize_header_name(column): column
        for column in df.columns
        if column != label_col
    }

    candidates = []
    if requested:
        key = _normalize_header_name(requested)
        if key not in normalized_columns:
            raise ValueError(
                f"Requested attribute column '{requested}' was not found in {filename}."
            )
        candidates.append((normalized_columns[key], "explicit_column"))
    else:
        raw_target = pd.Series(raw_target).astype(str).str.strip()
        raw_classes = int(raw_target.nunique(dropna=False))
        # CICIDS2017 and NSL-KDD derive the binary target from a detailed attack
        # category. Preserve that pre-binarization category as the sensitive target.
        if 3 <= raw_classes <= 100:
            candidates.append(("__RAW_DETAILED_TARGET__", "pre_binary_attack_category"))

        preferred = [
            "attacksubcategory", "subcategory", "subcat", "attacktype",
            "attackcategory", "category", "cat", "type", "trafficcategory",
            "protocol", "proto", "protocoltype", "service", "devicetype",
            "device", "applicationprotocol",
        ]
        for key in preferred:
            if key in normalized_columns:
                candidates.append((normalized_columns[key], "dataset_column"))

        if is_nsl_kdd_numeric_file(filename) and "1" in df.columns:
            candidates.append(("1", "nsl_kdd_protocol_type"))

    seen = set()
    for candidate, source in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate == "__RAW_DETAILED_TARGET__":
            values = pd.Series(raw_target).astype(str).str.strip()
            display_name = f"{label_col} (detailed pre-binary category)"
        else:
            values = df[candidate].fillna("__MISSING__").astype(str).str.strip()
            display_name = str(candidate)
        cardinality = int(values.nunique(dropna=False))
        if not (2 <= cardinality <= 100):
            continue
        if _attribute_equivalent_to_binary_label(values, binary_y):
            continue
        return values.reset_index(drop=True), display_name, source

    return None, None, "not_available"


def stratified_cap_indices(labels, max_samples, seed):
    labels = np.asarray(labels)
    if max_samples is None or len(labels) <= int(max_samples):
        return np.arange(len(labels), dtype=np.int64)
    selected, _ = train_test_split(
        np.arange(len(labels), dtype=np.int64),
        train_size=int(max_samples),
        random_state=int(seed),
        stratify=labels,
    )
    return np.sort(np.asarray(selected, dtype=np.int64))


def build_reconstruction_attacker(input_dim, output_dim):
    attacker = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.10),
        layers.Dense(128, activation="relu"),
        layers.Dense(output_dim, activation="linear"),
    ], name="Reviewer_Reconstruction_Attacker")
    attacker.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return attacker


def run_reconstruction_condition(
    condition, attacker_input, original_target, class_labels,
    seed, attack_epochs, batch_size, max_samples,
):
    """Train/test the attacker only on held-out target-model nonmembers."""
    pool_idx = stratified_cap_indices(class_labels, max_samples, seed + 700)
    x_pool = np.asarray(attacker_input[pool_idx], dtype=np.float32)
    z_pool = np.asarray(original_target[pool_idx], dtype=np.float32)
    y_pool = np.asarray(class_labels[pool_idx], dtype=np.int32)
    attacker_train, attacker_test = train_test_split(
        np.arange(len(pool_idx), dtype=np.int64),
        test_size=0.40,
        random_state=seed,
        stratify=y_pool,
    )
    set_seed(seed)
    attacker = build_reconstruction_attacker(x_pool.shape[1], z_pool.shape[1])
    attacker.fit(
        x_pool[attacker_train], z_pool[attacker_train],
        validation_split=0.10,
        epochs=max(1, int(attack_epochs)),
        batch_size=max(16, int(batch_size)),
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )],
        verbose=0,
    )
    reconstructed = attacker.predict(
        x_pool[attacker_test], batch_size=batch_size, verbose=0
    )
    target = z_pool[attacker_test]
    mse = float(np.mean(np.square(reconstructed - target)))
    numerator = np.sum(reconstructed * target, axis=1)
    denominator = (
        np.linalg.norm(reconstructed, axis=1)
        * np.linalg.norm(target, axis=1)
    )
    cosine_values = numerator / np.maximum(denominator, 1e-12)
    cosine = float(np.mean(np.clip(cosine_values, -1.0, 1.0)))
    result = {
        "condition": condition,
        "attacker_data_origin": "held_out_target_test_only",
        "attacker_train_samples": int(len(attacker_train)),
        "attacker_test_samples": int(len(attacker_test)),
        "mse": mse,
        "cosine_similarity": cosine,
        "status": "evaluated",
    }
    del attacker, reconstructed
    gc.collect()
    return result


def _confidence_vector(probabilities):
    probabilities = np.clip(
        np.asarray(probabilities, dtype=np.float64).reshape(-1), 1e-7, 1 - 1e-7
    )
    return np.column_stack([1.0 - probabilities, probabilities])


def run_shadow_membership_attack(
    F_train, y_train, prob_train,
    F_test, y_test, prob_test,
    seed, shadow_models, attack_epochs, batch_size, max_samples,
):
    """Standard shadow-model membership attack with target-model holdout evaluation."""
    heldout_idx = stratified_cap_indices(y_test, max_samples, seed + 900)
    shadow_local_idx, target_nonmember_local_idx = train_test_split(
        np.arange(len(heldout_idx), dtype=np.int64),
        test_size=0.30,
        random_state=seed + 901,
        stratify=np.asarray(y_test)[heldout_idx],
    )
    shadow_pool_idx = heldout_idx[shadow_local_idx]
    target_nonmember_pool_idx = heldout_idx[target_nonmember_local_idx]
    shadow_x = np.asarray(F_test[shadow_pool_idx], dtype=np.float32)
    shadow_y = np.asarray(y_test[shadow_pool_idx], dtype=np.int32)
    attack_x_parts = []
    attack_y_parts = []

    for shadow_id in range(max(1, int(shadow_models))):
        shadow_seed = seed + 10000 + shadow_id
        shadow_member, shadow_nonmember = train_test_split(
            np.arange(len(shadow_x), dtype=np.int64),
            test_size=0.50,
            random_state=shadow_seed,
            stratify=shadow_y,
        )
        set_seed(shadow_seed)
        shadow_model = build_mlp(shadow_x.shape[1])
        shadow_model.fit(
            shadow_x[shadow_member], shadow_y[shadow_member],
            validation_split=0.10,
            epochs=max(1, int(attack_epochs)),
            batch_size=max(16, int(batch_size)),
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )],
            verbose=0,
        )
        p_member = shadow_model.predict(
            shadow_x[shadow_member], batch_size=batch_size, verbose=0
        ).ravel()
        p_nonmember = shadow_model.predict(
            shadow_x[shadow_nonmember], batch_size=batch_size, verbose=0
        ).ravel()
        attack_x_parts.extend([
            _confidence_vector(p_member),
            _confidence_vector(p_nonmember),
        ])
        attack_y_parts.extend([
            np.ones(len(p_member), dtype=np.int32),
            np.zeros(len(p_nonmember), dtype=np.int32),
        ])
        del shadow_model
        gc.collect()

    attack_x_train = np.vstack(attack_x_parts)
    attack_y_train = np.concatenate(attack_y_parts)
    attack_classifier = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=seed
    )
    attack_classifier.fit(attack_x_train, attack_y_train)

    target_per_group = min(
        len(y_train), len(target_nonmember_pool_idx),
        max(50, int(max_samples) // 2)
    )
    target_member_idx = stratified_cap_indices(y_train, target_per_group, seed + 1200)
    target_nonmember_local = stratified_cap_indices(
        np.asarray(y_test)[target_nonmember_pool_idx],
        target_per_group,
        seed + 1300,
    )
    target_nonmember_idx = target_nonmember_pool_idx[target_nonmember_local]
    target_attack_x = np.vstack([
        _confidence_vector(np.asarray(prob_train)[target_member_idx]),
        _confidence_vector(np.asarray(prob_test)[target_nonmember_idx]),
    ])
    target_attack_y = np.concatenate([
        np.ones(len(target_member_idx), dtype=np.int32),
        np.zeros(len(target_nonmember_idx), dtype=np.int32),
    ])
    attack_probability = attack_classifier.predict_proba(target_attack_x)[:, 1]
    attack_prediction = (attack_probability >= 0.5).astype(np.int32)
    return {
        "shadow_models": int(max(1, shadow_models)),
        "shadow_data_origin": (
            "heldout_shadow_pool_disjoint_from_target_training_and_"
            "target_nonmember_evaluation"
        ),
        "attack_classifier": "logistic_regression_on_binary_confidence_vector",
        "attack_train_samples": int(len(attack_y_train)),
        "attack_test_samples": int(len(target_attack_y)),
        "attack_accuracy": float(accuracy_score(target_attack_y, attack_prediction)),
        "attack_auc": float(roc_auc_score(target_attack_y, attack_probability)),
        "status": "evaluated",
    }


def run_attribute_inference_attack(
    permuted_test_features, sensitive_test,
    seed, max_samples,
):
    sensitive_test = np.asarray(sensitive_test, dtype=np.int32)
    counts = pd.Series(sensitive_test).value_counts()
    eligible = counts[counts >= 4].index.to_numpy(dtype=np.int32)
    keep = np.isin(sensitive_test, eligible)
    x = np.asarray(permuted_test_features[keep], dtype=np.float32)
    a = sensitive_test[keep]
    if len(np.unique(a)) < 2:
        raise ValueError(
            "Fewer than two sensitive classes have enough held-out samples."
        )
    pool_idx = stratified_cap_indices(a, max_samples, seed + 1500)
    x = x[pool_idx]
    a = a[pool_idx]
    attacker_train, attacker_test = train_test_split(
        np.arange(len(a), dtype=np.int64),
        test_size=0.40,
        random_state=seed,
        stratify=a,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[attacker_train])
    x_test = scaler.transform(x[attacker_test])
    classifier = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=seed
    )
    classifier.fit(x_train, a[attacker_train])
    prediction = classifier.predict(x_test)
    n_classes = int(len(np.unique(a)))
    return {
        "attacker_input": "permuted_deep_feature_vector_only",
        "attacker_data_origin": "held_out_target_test_only",
        "attacker_train_samples": int(len(attacker_train)),
        "attacker_test_samples": int(len(attacker_test)),
        "sensitive_classes": n_classes,
        "attack_accuracy": float(accuracy_score(a[attacker_test], prediction)),
        "attack_macro_f1": float(
            f1_score(a[attacker_test], prediction, average="macro", zero_division=0)
        ),
        "random_guessing_accuracy": float(1.0 / n_classes),
        "majority_baseline_accuracy": float(
            pd.Series(a[attacker_test]).value_counts(normalize=True).max()
        ),
        "status": "evaluated",
    }


def _train_ablation_mlp(train_features, y_train, seed, epochs, batch_size):
    set_seed(seed)
    model = build_mlp(train_features.shape[1])
    model.fit(
        train_features, y_train,
        validation_split=0.10,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    return model


def _aes_roundtrip_rows(rows, aes):
    recovered = []
    for row in np.asarray(rows, dtype=np.float32):
        nonce = os.urandom(12)
        ciphertext = aes.encrypt(nonce, row.tobytes(), None)
        plaintext = aes.decrypt(nonce, ciphertext, None)
        recovered.append(np.frombuffer(plaintext, dtype=np.float32))
    return np.vstack(recovered)


def run_five_case_ablation(
    mlp, base_metrics, F_train, F_test, NF_train, NF_test,
    PF_train, PF_test, recovered_full_test, aes, ledger_hashes,
    y_train, y_test, seed, epochs, batch_size,
):
    rows = [{
        "case": "1_cnn_bilstm_mlp_no_protection",
        **{k: base_metrics[k] for k in ["accuracy", "precision", "recall", "f1"]},
        "status": "evaluated",
    }]

    normalized_model = _train_ablation_mlp(
        NF_train, y_train, seed, epochs, batch_size
    )
    normalized_prob = normalized_model.predict(
        NF_test, batch_size=batch_size, verbose=0
    ).ravel()
    normalized_metrics = metrics_from_predictions(y_test, normalized_prob)
    rows.append({
        "case": "2_normalization_only",
        **{k: normalized_metrics[k] for k in ["accuracy", "precision", "recall", "f1"]},
        "status": "evaluated",
    })

    permuted_model = _train_ablation_mlp(
        PF_train, y_train, seed, epochs, batch_size
    )
    permuted_prob = permuted_model.predict(
        PF_test, batch_size=batch_size, verbose=0
    ).ravel()
    permuted_metrics = metrics_from_predictions(y_test, permuted_prob)
    rows.append({
        "case": "3_normalization_plus_permutation_no_encryption",
        **{k: permuted_metrics[k] for k in ["accuracy", "precision", "recall", "f1"]},
        "status": "evaluated",
    })

    aes_only_test = _aes_roundtrip_rows(F_test, aes)
    aes_only_prob = mlp.predict(
        aes_only_test, batch_size=batch_size, verbose=0
    ).ravel()
    aes_only_metrics = metrics_from_predictions(y_test, aes_only_prob)
    rows.append({
        "case": "4_aes_gcm_only_no_permutation",
        **{k: aes_only_metrics[k] for k in ["accuracy", "precision", "recall", "f1"]},
        "authorized_roundtrip_mae": float(np.mean(np.abs(aes_only_test - F_test))),
        "status": "evaluated",
    })

    if len(ledger_hashes) != len(recovered_full_test):
        raise RuntimeError("Ledger record count does not match protected test vectors.")
    full_prob = normalized_model.predict(
        recovered_full_test, batch_size=batch_size, verbose=0
    ).ravel()
    full_metrics = metrics_from_predictions(y_test, full_prob)
    rows.append({
        "case": "5_full_normalization_permutation_aes_gcm_local_ledger",
        **{k: full_metrics[k] for k in ["accuracy", "precision", "recall", "f1"]},
        "authorized_roundtrip_mae": float(
            np.mean(np.abs(recovered_full_test - NF_test))
        ),
        "status": "evaluated",
    })
    del normalized_model, permuted_model
    gc.collect()
    return rows


def _write_seed_experiment(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"REVIEWER EXPERIMENT AUTOSAVE: {path}")


def run_and_save_reviewer_experiments(
    results_dir, dataset_name, dataset_slug, seed,
    F_train, F_test, NF_train, NF_test, PF_train, PF_test,
    recovered_test, aes, ledger_hashes, mlp, base_metrics,
    y_train, y_test, sensitive_codes, sensitive_name, sensitive_source,
    train_idx, test_idx, args,
):
    experiment_dir = Path(results_dir) / "reviewer_experiments"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    reconstruction_rows = []
    for condition, attacker_input in [
        ("raw_deep_features_baseline", NF_test),
        ("permuted_deep_features", PF_test),
    ]:
        try:
            result = run_reconstruction_condition(
                condition, attacker_input, NF_test, y_test,
                seed, args.attack_epochs, BATCH_SIZE, args.attack_max_samples,
            )
            reconstruction_rows.append({
                "dataset": dataset_name, "seed": seed, **result
            })
        except Exception as exc:
            reconstruction_rows.append({
                "dataset": dataset_name, "seed": seed,
                "condition": condition, "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            })
    _write_seed_experiment(
        reconstruction_rows,
        experiment_dir / f"{dataset_slug}_seed_{seed}_reconstruction_attack.csv",
    )

    try:
        prob_train = mlp.predict(
            F_train, batch_size=BATCH_SIZE, verbose=0
        ).ravel()
        prob_test = mlp.predict(
            F_test, batch_size=BATCH_SIZE, verbose=0
        ).ravel()
        membership = run_shadow_membership_attack(
            F_train, y_train, prob_train,
            F_test, y_test, prob_test,
            seed, args.shadow_models, args.attack_epochs,
            BATCH_SIZE, args.attack_max_samples,
        )
        membership_rows = [{
            "dataset": dataset_name, "seed": seed, **membership
        }]
    except Exception as exc:
        membership_rows = [{
            "dataset": dataset_name, "seed": seed, "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }]
    _write_seed_experiment(
        membership_rows,
        experiment_dir / f"{dataset_slug}_seed_{seed}_membership_inference.csv",
    )

    if sensitive_codes is None:
        attribute_rows = [{
            "dataset": dataset_name, "seed": seed,
            "sensitive_attribute": "NOT_AVAILABLE",
            "sensitive_attribute_source": sensitive_source,
            "status": "not_evaluated",
            "reason": (
                "No non-binary-equivalent detailed category or categorical "
                "attribute was available; no surrogate label was invented."
            ),
        }]
    else:
        try:
            sensitive_test = np.asarray(sensitive_codes)[test_idx]
            attribute = run_attribute_inference_attack(
                PF_test, sensitive_test, seed, args.attack_max_samples
            )
            attribute_rows = [{
                "dataset": dataset_name, "seed": seed,
                "sensitive_attribute": sensitive_name,
                "sensitive_attribute_source": sensitive_source,
                **attribute,
            }]
        except Exception as exc:
            attribute_rows = [{
                "dataset": dataset_name, "seed": seed,
                "sensitive_attribute": sensitive_name,
                "sensitive_attribute_source": sensitive_source,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }]
    _write_seed_experiment(
        attribute_rows,
        experiment_dir / f"{dataset_slug}_seed_{seed}_attribute_inference.csv",
    )

    try:
        ablation_rows = run_five_case_ablation(
            mlp, base_metrics, F_train, F_test, NF_train, NF_test,
            PF_train, PF_test, recovered_test, aes, ledger_hashes,
            y_train, y_test, seed, EPOCHS, BATCH_SIZE,
        )
        ablation_rows = [
            {"dataset": dataset_name, "seed": seed, **row}
            for row in ablation_rows
        ]
    except Exception as exc:
        ablation_rows = [{
            "dataset": dataset_name, "seed": seed,
            "case": "ablation_suite", "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }]
    _write_seed_experiment(
        ablation_rows,
        experiment_dir / f"{dataset_slug}_seed_{seed}_ablation_study.csv",
    )


def combine_reviewer_experiment_csvs(results_dir):
    experiment_dir = Path(results_dir) / "reviewer_experiments"
    mappings = {
        "*_reconstruction_attack.csv": "reconstruction_attack_results.csv",
        "*_membership_inference.csv": "membership_inference_results.csv",
        "*_attribute_inference.csv": "attribute_inference_results.csv",
        "*_ablation_study.csv": "ablation_study_results.csv",
    }
    for pattern, output_name in mappings.items():
        frames = []
        for path in sorted(experiment_dir.glob(pattern)):
            try:
                frame = pd.read_csv(path)
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                print(f"WARNING: could not aggregate {path.name}: {exc}")
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(Path(results_dir) / output_name, index=False)
            print(f"REVIEWER TABLE READY: {Path(results_dir) / output_name}")

def safe_dataset_slug(name):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))


# ---------- 5. Load and audit datasets ----------
audit_rows = []
loaded_data = {}



def main(args=None):
    global PROJECT_DIR, RESULTS_DIR, BACKUPS_DIR, CODE_DIR
    global EPOCHS, BATCH_SIZE, SEEDS, ENABLE_GITHUB_UPLOAD
    args = args or parse_args()
    EPOCHS = max(1, int(args.epochs))
    BATCH_SIZE = max(1, int(args.batch_size))
    try:
        SEEDS = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must be comma-separated integers") from exc
    if not SEEDS:
        raise ValueError("At least one seed is required.")
    if args.attack_max_samples < 100:
        raise ValueError("--attack-max-samples must be at least 100.")
    if args.attack_epochs < 1:
        raise ValueError("--attack-epochs must be at least 1.")
    if args.shadow_models < 1:
        raise ValueError("--shadow-models must be at least 1.")
    ENABLE_GITHUB_UPLOAD = bool(args.github_upload)

    PROJECT_DIR, RESULTS_DIR, BACKUPS_DIR, CODE_DIR = initialize_persistent_storage(
        args.results_dir
    )
    backup_running_code()
    data_dir = args.data_dir
    if args.copy_datasets_to_drive:
        data_dir = str(copy_csv_datasets_to_drive(data_dir))
    DATASETS = build_dataset_config(data_dir)
    for name, cfg in DATASETS.items():
        print(f"\n===== Loading {name} =====")
        df = pd.read_csv(cfg["file"], low_memory=False)

        label_col = str(resolve_label_column(df, cfg["file"]))
        print("Resolved label column:", label_col)

        original_rows = len(df)
        original_cols = len(df.columns)

        # Dataset-specific safe preprocessing for the supplied numeric-column NSL-KDD.
        # 41 = attack-type target, 42 = difficulty level, Unnamed: 0 = exported row index.
        is_nsl = cfg.get("format") == "nsl_kdd_numeric"
        if is_nsl:
            drop_cols = [c for c in ["Unnamed: 0", "42"] if c in df.columns]
            if drop_cols:
                print("NSL-KDD non-feature columns excluded:", drop_cols)
                df = df.drop(columns=drop_cols)

        df, duplicates_removed = clean_dataframe(df)

        # Remove missing labels first
        before_label_drop = len(df)
        df = df.dropna(subset=[label_col]).reset_index(drop=True)
        missing_labels_removed = before_label_drop - len(df)
        raw_target_before_binary = df[label_col].copy().reset_index(drop=True)

        is_cicids = cfg.get("format") == "cicids2017"

        if is_nsl:
            # Standard binary NSL-KDD interpretation: normal=0, every named attack=1.
            raw_label = df[label_col].astype(str).str.strip().str.lower()
            y = (raw_label != "normal").astype(int)
            label_mapping = {"normal": 0, "all_non_normal_attack_types": 1}

            print("NSL-KDD attack types found:", df[label_col].nunique(dropna=True))
            print("Binary class counts:", y.value_counts().sort_index().to_dict())
            df[label_col] = y

        elif is_cicids:
            # CICIDS2017 binary task: BENIGN=0, every attack label=1.
            raw_label = df[label_col].astype(str).str.strip().str.lower()
            benign_tokens = {"benign", "normal", "normal traffic", "normal_traffic", "0"}
            y = (~raw_label.isin(benign_tokens)).astype(int)
            label_mapping = {"BENIGN/normal": 0, "all_attack_labels": 1}

            print("CICIDS2017 raw label classes found:", df[label_col].nunique(dropna=True))
            print("CICIDS2017 binary class counts:", y.value_counts().sort_index().to_dict())
            if y.nunique() != 2:
                raise ValueError(
                    f"CICIDS2017 binary conversion did not produce two classes. "
                    f"Raw labels sample: {df[label_col].dropna().astype(str).unique()[:20].tolist()}"
                )
            # Replace raw multiclass target for downstream leakage auditing.
            df[label_col] = y

        else:
            y, label_mapping = encode_binary_label(df[label_col])

        sensitive_values, sensitive_name, sensitive_source = (
            select_attribute_inference_target(
                df=df,
                label_col=label_col,
                raw_target=raw_target_before_binary,
                binary_y=y,
                filename=cfg["file"],
                requested=args.attribute_column,
            )
        )
        if sensitive_values is not None:
            print(
                f"Attribute-inference target: {sensitive_name} "
                f"[{sensitive_source}], classes="
                f"{sensitive_values.nunique(dropna=False)}"
            )
        else:
            print(
                "Attribute-inference target: NOT AVAILABLE. "
                "No second categorical label will be invented."
            )

        X, all_nan_cols = numeric_features(df, label_col)

        # Keep identical row index after conversion
        valid_mask = ~X.isna().all(axis=1)
        X = X.loc[valid_mask].reset_index(drop=True)
        y = y.loc[valid_mask].reset_index(drop=True)
        if sensitive_values is not None:
            sensitive_values = sensitive_values.loc[valid_mask].reset_index(drop=True)

        if args.max_rows and len(X) > args.max_rows:
            if args.max_rows < 20:
                raise ValueError("--max-rows must be at least 20.")
            keep, _ = train_test_split(
                np.arange(len(X)), train_size=args.max_rows,
                random_state=SEEDS[0], stratify=y
            )
            keep = np.sort(keep)
            X = X.iloc[keep].reset_index(drop=True)
            y = y.iloc[keep].reset_index(drop=True)
            if sensitive_values is not None:
                sensitive_values = sensitive_values.iloc[keep].reset_index(drop=True)
            print(f"Stratified row limit applied: {len(X):,}")

        if sensitive_values is not None:
            sensitive_codes, sensitive_levels = pd.factorize(
                sensitive_values.astype(str), sort=True
            )
            sensitive_codes = np.asarray(sensitive_codes, dtype=np.int32)
            sensitive_class_count = int(len(sensitive_levels))
        else:
            sensitive_codes = None
            sensitive_levels = np.asarray([], dtype=object)
            sensitive_class_count = 0

        audit_rows.append({
            "dataset": name,
            "original_rows": original_rows,
            "original_columns": original_cols,
            "after_duplicate_removal": len(df),
            "duplicates_removed": duplicates_removed,
            "missing_labels_removed": missing_labels_removed,
            "final_rows_for_model": len(X),
            "feature_columns_before_scaling": X.shape[1],
            "all_nan_columns_removed": len(all_nan_cols),
            "attribute_inference_target": sensitive_name or "NOT_AVAILABLE",
            "attribute_inference_source": sensitive_source,
            "attribute_inference_classes": sensitive_class_count,
            "label_mapping": json.dumps({str(k): int(v) for k,v in label_mapping.items()})
        })

        loaded_data[name] = (
            X, y, label_col,
            sensitive_codes, sensitive_name, sensitive_source,
        )

        print("Rows:", original_rows)
        print("Rows after cleaning:", len(X))
        print("Features:", X.shape[1])
        print("Label mapping:", label_mapping)
        print("Class counts:", y.value_counts().sort_index().to_dict())

    pd.DataFrame(audit_rows).to_csv(
        RESULTS_DIR / "dataset_audit.csv", index=False
    )

    if args.dry_run:
        print("\nDRY RUN COMPLETE: dataset detection and preprocessing succeeded.")
        print(pd.DataFrame(audit_rows).to_string(index=False))
        return

    require_training_dependencies()

    # ---------- 6. Main experiment ----------
    all_runs = []
    all_confusions = []
    all_histories = []

    # Load previously completed seeds so final summaries
    # include them after a resumed run.
    for old_file in sorted(RESULTS_DIR.glob("*_seed_*_results.csv")):
        try:
            old_df = pd.read_csv(old_file)
            if not old_df.empty:
                all_runs.extend(old_df.to_dict("records"))
                print(f"RESUME: loaded {old_file.name}")
        except Exception as e:
            print(f"WARNING: could not load {old_file.name}: {e}")

    for dataset_name, (
        X_df, y, label_col,
        sensitive_codes, sensitive_name, sensitive_source,
    ) in loaded_data.items():

        print(f"\n\n############################")
        print(f"DATASET: {dataset_name}")
        print(f"############################")

        X_np = X_df.to_numpy(dtype=np.float32)
        y_np = y.to_numpy(dtype=np.int32)

        dataset_slug = safe_dataset_slug(dataset_name)
        for seed in [s for s in SEEDS
                       if not (RESULTS_DIR / f"{dataset_slug}_seed_{s}_results.csv").exists()]:
            print(f"\n--- {dataset_name}: seed {seed} ---")
            set_seed(seed)
            tf.keras.backend.clear_session()
            gc.collect()

            # Leakage-safe 70:30 group split.
            # Identical feature vectors are forced into the same split.
            train_idx, test_idx = leakage_safe_group_split(
                pd.concat(
                    [
                        X_df.reset_index(drop=True),
                        pd.Series(y_np, name=label_col)
                    ],
                    axis=1
                ),
                label_col,
                test_size=TEST_SIZE,
                random_state=seed
            )

            train_idx = np.asarray(train_idx, dtype=np.int64)
            test_idx = np.asarray(test_idx, dtype=np.int64)

            if len(X_np) != len(y_np):
                raise RuntimeError(
                    f"ROW ALIGNMENT FAILURE: X_np={len(X_np)}, y_np={len(y_np)}"
                )

            if train_idx.max() >= len(X_np) or test_idx.max() >= len(X_np):
                raise RuntimeError(
                    f"INDEX RANGE FAILURE: X_np={len(X_np)}, "
                    f"train_max={train_idx.max()}, test_max={test_idx.max()}"
                )

            X_train_raw = X_np[train_idx]
            X_test_raw = X_np[test_idx]
            y_train = y_np[train_idx]
            y_test = y_np[test_idx]

            # Median imputation FIT ONLY on training
            train_medians = np.nanmedian(X_train_raw, axis=0)
            train_medians = np.where(
                np.isfinite(train_medians), train_medians, 0.0
            )

            X_train_raw = np.where(
                np.isnan(X_train_raw), train_medians, X_train_raw
            )
            X_test_raw = np.where(
                np.isnan(X_test_raw), train_medians, X_test_raw
            )

            # StandardScaler FIT ONLY on training
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
            X_test = scaler.transform(X_test_raw).astype(np.float32)

            # Current supplied implementation uses ordered feature vectors,
            # not packet-level temporal reconstruction.
            X_train_seq = X_train[..., np.newaxis]
            X_test_seq = X_test[..., np.newaxis]

            # ---------- Baseline deep feature extractor ----------
            t0 = time.perf_counter()

            extractor = build_cnn_bilstm(X_train_seq.shape[1:])
            extractor.compile(
                optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            # Train extractor as an IDS classifier head temporarily.
            clf_head = keras.Sequential([
                extractor,
                layers.Dense(1, activation="sigmoid")
            ])

            clf_head.compile(
                optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            history = clf_head.fit(
                X_train_seq, y_train,
                validation_split=0.10,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            )

            # Extract deep features
            feature_model = keras.Model(
                inputs=extractor.input,
                outputs=extractor.get_layer("deep_features").output
            )

            t_feature = time.perf_counter()
            F_train = feature_model.predict(X_train_seq, batch_size=BATCH_SIZE, verbose=0)
            F_test = feature_model.predict(X_test_seq, batch_size=BATCH_SIZE, verbose=0)
            t_extract = time.perf_counter()

            # MLP
            mlp = build_mlp(F_train.shape[1])

            t_mlp0 = time.perf_counter()
            mlp.fit(
                F_train, y_train,
                validation_split=0.10,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            )
            t_mlp_train = time.perf_counter() - t_mlp0

            t_inf0 = time.perf_counter()
            prob = mlp.predict(F_test, batch_size=BATCH_SIZE, verbose=0).ravel()
            t_inference = time.perf_counter() - t_inf0

            baseline_runtime = (
                t_feature - t0
                + t_mlp_train
                + t_inference
            )

            base_metrics = metrics_from_predictions(y_test, prob)

            # ---------- Security timing ----------
            # Normalize deep features using training statistics only
            sec0 = time.perf_counter()

            f_scaler = StandardScaler()
            NF_train = f_scaler.fit_transform(F_train).astype(np.float32)
            NF_test = f_scaler.transform(F_test).astype(np.float32)
            normalization_time = time.perf_counter() - sec0

            # Secret permutation
            rng = np.random.default_rng(seed)
            perm = rng.permutation(NF_train.shape[1])

            p0 = time.perf_counter()
            PF_train = NF_train[:, perm]
            PF_test = NF_test[:, perm]
            permutation_time = time.perf_counter() - p0

            # AES-GCM encryption
            # Encrypt each test vector for measurable security overhead.
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            key = AESGCM.generate_key(bit_length=256)
            aes = AESGCM(key)

            e0 = time.perf_counter()
            encrypted_test = []
            for row in PF_test:
                nonce = os.urandom(12)
                ct = aes.encrypt(nonce, row.tobytes(), None)
                encrypted_test.append((nonce, ct))
            encryption_time = time.perf_counter() - e0

            # Conventional authenticated storage: SHA-256 digest
            c0 = time.perf_counter()
            conventional_hashes = [
                hashlib.sha256(ct).hexdigest()
                for nonce, ct in encrypted_test
            ]
            conventional_storage_time = time.perf_counter() - c0

            # Lightweight permissioned-ledger simulation
            b0 = time.perf_counter()
            prev_hash = "0" * 64
            ledger_hashes = []
            for i, (nonce, ct) in enumerate(encrypted_test):
                payload_hash = hashlib.sha256(ct).hexdigest()
                block_string = f"{i}|{payload_hash}|{prev_hash}"
                block_hash = hashlib.sha256(block_string.encode()).hexdigest()
                ledger_hashes.append(block_hash)
                prev_hash = block_hash
            blockchain_time = time.perf_counter() - b0

            # Decrypt
            d0 = time.perf_counter()
            decrypted = []
            for nonce, ct in encrypted_test:
                plain = aes.decrypt(nonce, ct, None)
                decrypted.append(
                    np.frombuffer(plain, dtype=np.float32)
                )
            decrypted = np.vstack(decrypted)
            decryption_time = time.perf_counter() - d0

            # Unshuffle
            u0 = time.perf_counter()
            inverse_perm = np.argsort(perm)
            recovered_test = decrypted[:, inverse_perm]
            unshuffle_time = time.perf_counter() - u0

            security_extra_time = (
                normalization_time
                + permutation_time
                + encryption_time
                + conventional_storage_time
                + blockchain_time
                + decryption_time
                + unshuffle_time
            )

            secure_total_runtime = baseline_runtime + security_extra_time
            overhead_pct = (
                100.0 * security_extra_time / baseline_runtime
                if baseline_runtime > 0 else np.nan
            )
            throughput = (
                len(y_test) / secure_total_runtime
                if secure_total_runtime > 0 else np.nan
            )

            # Check that encryption/decryption recovers the protected vector
            reconstruction_error = float(
                np.mean(np.abs(recovered_test - NF_test))
            )

            # Memory snapshot (process-level; recorded as an execution diagnostic)
            if psutil is not None:
                memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            else:
                import resource
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                memory_mb = rss / 1024.0  # Linux reports KiB.

            row = {
                "dataset": dataset_name,
                "seed": seed,
                **base_metrics,
                "train_samples": len(y_train),
                "test_samples": len(y_test),
                "n_features": X_train.shape[1],
                "deep_feature_dim": F_train.shape[1],
                "baseline_runtime_s": baseline_runtime,
                "normalization_s": normalization_time,
                "permutation_s": permutation_time,
                "encryption_s": encryption_time,
                "conventional_storage_s": conventional_storage_time,
                "blockchain_s": blockchain_time,
                "decryption_s": decryption_time,
                "unshuffle_s": unshuffle_time,
                "security_extra_time_s": security_extra_time,
                "secure_total_runtime_s": secure_total_runtime,
                "overhead_pct": overhead_pct,
                "throughput_samples_per_s": throughput,
                "process_memory_mb": memory_mb,
                "reconstruction_mae": reconstruction_error,
            }

            # Reviewer experiments are add-ons. The original pipeline above,
            # its model, predictions, timing variables, and published row are
            # left unchanged. Each add-on uses the same split and seed.
            if not args.skip_reviewer_experiments:
                run_and_save_reviewer_experiments(
                    results_dir=RESULTS_DIR,
                    dataset_name=dataset_name,
                    dataset_slug=dataset_slug,
                    seed=seed,
                    F_train=F_train,
                    F_test=F_test,
                    NF_train=NF_train,
                    NF_test=NF_test,
                    PF_train=PF_train,
                    PF_test=PF_test,
                    recovered_test=recovered_test,
                    aes=aes,
                    ledger_hashes=ledger_hashes,
                    mlp=mlp,
                    base_metrics=base_metrics,
                    y_train=y_train,
                    y_test=y_test,
                    sensitive_codes=sensitive_codes,
                    sensitive_name=sensitive_name,
                    sensitive_source=sensitive_source,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    args=args,
                )

            all_runs.append(row)
    # AUTOSAVE: save this seed immediately to Google Drive
            seed_result_path = RESULTS_DIR / f"{dataset_slug}_seed_{seed}_results.csv"
            pd.DataFrame([row]).to_csv(
                seed_result_path,
                index=False
            )
            print(f"\nAUTOSAVE COMPLETE: {seed_result_path}")

            cm = confusion_matrix(
                y_test,
                (prob >= 0.5).astype(int),
                labels=[0,1]
            )

            all_confusions.append({
                "dataset": dataset_name,
                "seed": seed,
                "TN": int(cm[0,0]),
                "FP": int(cm[0,1]),
                "FN": int(cm[1,0]),
                "TP": int(cm[1,1]),
            })

            # Save confusion matrix immediately
            pd.DataFrame(all_confusions).to_csv(
                RESULTS_DIR / "confusion_matrices.csv",
                index=False
            )

            for epoch, (loss, acc) in enumerate(
                zip(history.history["loss"], history.history["accuracy"]), start=1
            ):
                all_histories.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "epoch": epoch,
                    "loss": loss,
                    "accuracy": acc
                })

            # Save one confusion matrix image per run
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(5,4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Attack"],
                yticklabels=["Benign", "Attack"]
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"{dataset_name} — Seed {seed}")
            plt.tight_layout()
            plt.savefig(
                RESULTS_DIR / f"confusion_{dataset_name.replace(' ','_')}_seed{seed}.png",
                dpi=300
            )
            plt.close()

            del extractor, clf_head, feature_model, mlp
            gc.collect()
            tf.keras.backend.clear_session()

    # ---------- 7. Save raw results ----------
    runs_df = pd.DataFrame(all_runs)
    cm_df = pd.DataFrame(all_confusions)
    hist_df = pd.DataFrame(all_histories)

    runs_df.to_csv(RESULTS_DIR / "all_runs_raw.csv", index=False)
    cm_df.to_csv(RESULTS_DIR / "confusion_matrices.csv", index=False)
    hist_df.to_csv(RESULTS_DIR / "training_history.csv", index=False)

    # ---------- 8. Aggregate 5-run results ----------
    metric_cols = [
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "baseline_runtime_s", "normalization_s", "permutation_s",
        "encryption_s", "conventional_storage_s", "blockchain_s",
        "decryption_s", "unshuffle_s", "security_extra_time_s",
        "secure_total_runtime_s", "overhead_pct",
        "throughput_samples_per_s", "process_memory_mb",
        "reconstruction_mae"
    ]

    summary_mean = runs_df.groupby("dataset")[metric_cols].mean().reset_index()
    summary_std = runs_df.groupby("dataset")[metric_cols].std(ddof=1).reset_index()

    summary_mean.to_csv(RESULTS_DIR / "results_mean.csv", index=False)
    summary_std.to_csv(RESULTS_DIR / "results_std.csv", index=False)

    # Combined mean ± std table
    combined = summary_mean.copy()
    for m in metric_cols:
        std_map = summary_std.set_index("dataset")[m]
        combined[m] = combined.apply(
            lambda r: f"{r[m]:.6f} ± {std_map.loc[r['dataset']]:.6f}",
            axis=1
        )

    combined.to_csv(RESULTS_DIR / "results_mean_plus_std.csv", index=False)

    # ---------- 9. Ablation study ----------
    # The ablation below evaluates feature-protection transformations using
    # the same trained deep-feature representation. It does NOT claim that
    # these transformations improve classification accuracy; it measures
    # whether protected representations can be recovered consistently and
    # records storage/processing cost.

    ablation_rows = []

    for dataset_name, (
        X_df, y, label_col,
        _sensitive_codes, _sensitive_name, _sensitive_source,
    ) in loaded_data.items():
        X_np = X_df.to_numpy(dtype=np.float32)
        y_np = y.to_numpy(dtype=np.int32)

        # Use a fixed seed for controlled ablation.
        seed = 42
        set_seed(seed)

        # Leakage-safe group split.
        # Identical feature vectors are forced into the same split.
        train_idx, test_idx = leakage_safe_group_split(
            pd.concat(
                [
                    X_df.reset_index(drop=True),
                    pd.Series(y_np, name=label_col)
                ],
                axis=1
            ),
            label_col,
            test_size=TEST_SIZE,
            random_state=seed
        )

        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)

        if len(X_np) != len(y_np):
            raise RuntimeError(
                f"ROW ALIGNMENT FAILURE: X_np={len(X_np)}, y_np={len(y_np)}"
            )

        if train_idx.max() >= len(X_np) or test_idx.max() >= len(X_np):
            raise RuntimeError(
                f"INDEX RANGE FAILURE: X_np={len(X_np)}, "
                f"train_max={train_idx.max()}, test_max={test_idx.max()}"
            )

        X_train_raw = X_np[train_idx]
        X_test_raw = X_np[test_idx]
        y_train = y_np[train_idx]
        y_test = y_np[test_idx]

        train_medians = np.nanmedian(X_train_raw, axis=0)
        train_medians = np.where(np.isfinite(train_medians), train_medians, 0.0)

        X_train_raw = np.where(np.isnan(X_train_raw), train_medians, X_train_raw)
        X_test_raw = np.where(np.isnan(X_test_raw), train_medians, X_test_raw)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test = scaler.transform(X_test_raw).astype(np.float32)

        X_train_seq = X_train[..., np.newaxis]
        X_test_seq = X_test[..., np.newaxis]

        # Train one representation model for controlled component analysis
        extractor = build_cnn_bilstm(X_train_seq.shape[1:])
        clf_head = keras.Sequential([
            extractor,
            layers.Dense(1, activation="sigmoid")
        ])
        clf_head.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        clf_head.fit(
            X_train_seq, y_train,
            validation_split=0.10,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        feature_model = keras.Model(
            extractor.input,
            extractor.get_layer("deep_features").output
        )

        F_train = feature_model.predict(X_train_seq, batch_size=BATCH_SIZE, verbose=0)
        F_test = feature_model.predict(X_test_seq, batch_size=BATCH_SIZE, verbose=0)

        f_scaler = StandardScaler()
        NF_train = f_scaler.fit_transform(F_train).astype(np.float32)
        NF_test = f_scaler.transform(F_test).astype(np.float32)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(NF_train.shape[1])
        inverse_perm = np.argsort(perm)

        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        key = AESGCM.generate_key(bit_length=256)
        aes = AESGCM(key)

        # Define controlled variants.
        variants = {}

        # 1. Normalization only
        variants["normalization_only"] = NF_test

        # 2. Normalization + permutation
        variants["normalization_permutation"] = NF_test[:, perm]

        # 3. AES-GCM without permutation
        enc_no_perm = []
        for row in NF_test:
            nonce = os.urandom(12)
            ct = aes.encrypt(nonce, row.tobytes(), None)
            enc_no_perm.append((nonce, ct))
        dec_no_perm = [
            np.frombuffer(aes.decrypt(n, c, None), dtype=np.float32)
            for n, c in enc_no_perm
        ]
        variants["aes_gcm_no_permutation"] = np.vstack(dec_no_perm)

        # 4. AES-GCM with permutation
        enc_perm = []
        for row in NF_test[:, perm]:
            nonce = os.urandom(12)
            ct = aes.encrypt(nonce, row.tobytes(), None)
            enc_perm.append((nonce, ct))
        dec_perm = [
            np.frombuffer(aes.decrypt(n, c, None), dtype=np.float32)
            for n, c in enc_perm
        ]
        variants["aes_gcm_with_permutation"] = np.vstack(dec_perm)[:, inverse_perm]

        for variant, recovered in variants.items():
            mae = float(np.mean(np.abs(recovered - NF_test)))
            max_abs = float(np.max(np.abs(recovered - NF_test)))

            ablation_rows.append({
                "dataset": dataset_name,
                "variant": variant,
                "reconstruction_mae_vs_normalized_features": mae,
                "max_absolute_error": max_abs,
            })

        del extractor, clf_head, feature_model
        gc.collect()
        tf.keras.backend.clear_session()

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(RESULTS_DIR / "ablation_reconstruction.csv", index=False)

    if not args.skip_reviewer_experiments:
        combine_reviewer_experiment_csvs(RESULTS_DIR)

    # ---------- 10. Package results ----------
    # Create persistent ZIP in Google Drive without placing the ZIP inside itself.
    zip_base = PROJECT_DIR / "reviewer_results_all_datasets"
    zip_path = Path(str(zip_base) + ".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_base), "zip", root_dir=RESULTS_DIR)

    print("\n==============================================")
    print("EXPERIMENT COMPLETED")
    print("Results directory:", RESULTS_DIR)
    print("ZIP:", zip_path)
    print("==============================================")

    print("\nMain results:")
    display(combined)

    print("\nConfusion matrices:")
    display(cm_df)

    print("\nDataset audit:")
    display(pd.DataFrame(audit_rows))

    print("\nDownload the ZIP from the Colab file panel:")
    print(zip_path)

def _run_git(args, cwd, env=None, check=True):
    cmd = ["git"] + list(args)
    print("GIT:", " ".join(args))
    return subprocess.run(cmd, cwd=str(cwd), env=env, check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def _get_github_token():
    # Preferred: Colab Secrets named GITHUB_TOKEN.
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
        if token:
            return token.strip()
    except Exception:
        pass
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token
    print("\nGitHub upload requires a Personal Access Token.")
    print("The token is NOT written into the project files.")
    return getpass.getpass("GitHub token (leave blank to skip GitHub upload): ").strip()

def sync_project_to_github():
    """Create/update a GitHub repo and push code + results, never raw datasets."""
    if not ENABLE_GITHUB_UPLOAD:
        print("GitHub upload disabled by configuration.")
        return

    token = _get_github_token()
    if not token:
        print("GitHub upload skipped: no token supplied.")
        return

    try:
        import requests
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        me = requests.get("https://api.github.com/user", headers=headers, timeout=30)
        me.raise_for_status()
        username = me.json()["login"]

        repo_check = requests.get(
            f"https://api.github.com/repos/{username}/{GITHUB_REPO}",
            headers=headers, timeout=30
        )
        repo_exists = repo_check.status_code == 200
        repo_is_empty = False
        default_branch = "main"
        if repo_exists:
            repo_metadata = repo_check.json()
            repo_is_empty = int(repo_metadata.get("size", 0)) == 0
            default_branch = repo_metadata.get("default_branch") or "main"
        if repo_check.status_code == 404:
            create = requests.post(
                "https://api.github.com/user/repos",
                headers=headers,
                json={
                    "name": GITHUB_REPO,
                    "private": bool(GITHUB_PRIVATE),
                    "description": "Leakage-safe multi-dataset IDS experiments and reproducible results",
                    "auto_init": False,
                }, timeout=30
            )
            create.raise_for_status()
            print(f"Created GitHub repository: {username}/{GITHUB_REPO}")
        elif repo_check.status_code >= 400:
            repo_check.raise_for_status()

        staging = Path("/content/github_secure_ids_stage") if Path("/content").exists() else Path.cwd() / "github_secure_ids_stage"
        if staging.exists():
            shutil.rmtree(staging)

        # Askpass keeps the token out of command arguments and .git/config.
        askpass = staging.parent / ".secure_ids_git_askpass.sh"
        askpass.write_text(
            '#!/bin/sh\ncase "$1" in\n*Username*) echo "x-access-token" ;;\n*Password*) echo "$GITHUB_TOKEN" ;;\nesac\n',
            encoding="utf-8"
        )
        askpass.chmod(0o700)
        env = os.environ.copy()
        env["GITHUB_TOKEN"] = token
        env["GIT_ASKPASS"] = str(askpass)
        env["GIT_TERMINAL_PROMPT"] = "0"
        remote = f"https://github.com/{username}/{GITHUB_REPO}.git"

        if repo_exists and not repo_is_empty:
            _run_git(
                ["clone", "--branch", default_branch, remote, str(staging)],
                staging.parent,
                env=env,
            )
            _run_git(["checkout", "-B", "main"], staging)
        else:
            staging.mkdir(parents=True)
            _run_git(["init"], staging)
            _run_git(["checkout", "-B", "main"], staging)
            _run_git(["remote", "add", "origin", remote], staging)

        # Copy executable code and complete results. Raw CSV datasets are never copied.
        script_path = Path(__file__).resolve()
        shutil.copy2(script_path, staging / script_path.name)
        shutil.copytree(RESULTS_DIR, staging / "results", dirs_exist_ok=True)
        if (PROJECT_DIR / "reviewer_results_all_datasets.zip").exists():
            shutil.copy2(PROJECT_DIR / "reviewer_results_all_datasets.zip", staging / "reviewer_results_all_datasets.zip")

        readme = f"""# Secure IDS – Multi-Dataset Experiments

Leakage-safe CNN-BiLSTM + MLP intrusion-detection experiments with separate
reviewer-requested privacy attacks and utility ablation.

## Datasets
Every supported CSV placed in the selected data directory is processed.
Raw datasets are not redistributed in this repository.

## Reproducibility
- Seeds: {SEEDS}
- Test size: {TEST_SIZE}
- Epochs: {EPOCHS}
- Batch size: {BATCH_SIZE}
- Learning rate: {LEARNING_RATE}
- Group-aware train/test split
- Train-only imputation and scaling
- Automated direct target-leakage audit
- Reconstruction attack: unprotected vs. permuted deep features
- Shadow-model membership inference with target-model evaluation
- Attribute inference from permuted deep features only
- Five-case classification ablation on identical per-seed splits

The `results/` directory contains the unchanged original outputs plus four
separate reviewer tables: reconstruction, membership inference, attribute
inference, and five-case ablation.
Raw datasets are intentionally excluded.
"""
        (staging / "README.md").write_text(readme, encoding="utf-8")
        (staging / ".gitignore").write_text(
            "*.csv\n!results/*.csv\n*.parquet\n*.zip\n!reviewer_results_all_datasets.zip\n"
            "__pycache__/\n*.pyc\n.ipynb_checkpoints/\n",
            encoding="utf-8"
        )

        _run_git(["config", "user.name", username], staging)
        noreply = f"{username}@users.noreply.github.com"
        _run_git(["config", "user.email", noreply], staging)
        _run_git(["add", "."], staging)
        commit = _run_git(["commit", "-m", "Update leakage-safe multi-dataset experiment results"], staging, check=False)
        if commit.stdout:
            print(commit.stdout.strip())

        push = _run_git(["push", "-u", "origin", "main"], staging, env=env)
        if push.stdout:
            print(push.stdout.strip())
        askpass.unlink(missing_ok=True)
        print(f"\nGitHub upload complete: https://github.com/{username}/{GITHUB_REPO}")
    except Exception as exc:
        print(f"WARNING: GitHub upload failed: {exc}")
        print("All results are still safely stored in Google Drive.")

if __name__ == "__main__":
    main()
    sync_project_to_github()
