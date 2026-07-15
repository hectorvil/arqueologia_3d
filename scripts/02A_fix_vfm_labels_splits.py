from pathlib import Path
import re
import json

import numpy as np
import pandas as pd


PROJECT_DIR = Path("/Users/hectorperalta/Documents/arqueologia_3d")
MODEL_READY_DIR = PROJECT_DIR / "data" / "model_ready"

INDEX_PATH = MODEL_READY_DIR / "index.csv"
BACKUP_PATH = MODEL_READY_DIR / "index_backup_before_vfm_fix.csv"

LABEL_MAP_PATH = MODEL_READY_DIR / "label_map.json"
CLASS_WEIGHTS_PATH = MODEL_READY_DIR / "class_weights.json"
SPLIT_SUMMARY_PATH = MODEL_READY_DIR / "split_summary.csv"
BALANCED_INDEX_PATH = MODEL_READY_DIR / "index_balanced_first_model.csv"

RANDOM_SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Para el primer modelo. Puedes subirlo después.
MAX_TRAIN_PER_CLASS = 1000
MAX_VAL_PER_CLASS = 300
MAX_TEST_PER_CLASS = 300


FRAGMENT_RE = re.compile(
    r"^(?P<vessel_id>.+)_"
    r"(?P<n_fragments>\d+)f_"
    r"(?P<resolution>\d+)_"
    r"(?P<iteration>\d+)it_"
    r"(?P<fragment_id>\d+)_"
    r"1024p\.ply$"
)

FULL_RE = re.compile(
    r"^(?P<vessel_id>.+)_1024p\.ply$"
)


def is_valid(x):
    if pd.isna(x):
        return False
    s = str(x).strip()
    return s not in {"", "nan", "None", "null"}


def parse_vfm_file(source_path: str) -> dict:
    path = Path(str(source_path))
    name = path.name
    vessel_folder = path.parent.name

    complete_name = f"{vessel_folder}_1024p.ply"

    if name == complete_name:
        return {
            "artifact_kind": "iberian_vessel_complete",
            "label_original": "iberian_vessel_complete",
            "ml_label": "iberian_vessel_complete",
            "is_fragment": False,
            "vessel_id": vessel_folder,
            "n_fragments": None,
            "resolution": None,
            "iteration": None,
            "fragment_id": None,
        }

    if name.endswith("_1024p.ply"):
        return {
            "artifact_kind": "iberian_vessel_fragment",
            "label_original": "iberian_vessel_fragment",
            "ml_label": "iberian_vessel_fragment",
            "is_fragment": True,
            "vessel_id": vessel_folder,
            "n_fragments": None,
            "resolution": None,
            "iteration": None,
            "fragment_id": None,
        }

    return {
        "artifact_kind": "unknown_vfm_pointcloud",
        "label_original": "unknown_vfm_pointcloud",
        "ml_label": "unknown_vfm_pointcloud",
        "is_fragment": None,
        "vessel_id": vessel_folder,
        "n_fragments": None,
        "resolution": None,
        "iteration": None,
        "fragment_id": None,
    }


def build_group_id(row: pd.Series) -> str:
    dataset = str(row.get("dataset", "unknown_dataset"))

    if dataset == "vfm_iberian_vessels":
        if is_valid(row.get("vessel_id")):
            return f"{dataset}::{row.get('vessel_id')}"
        return f"{dataset}::{Path(str(row.get('source_path'))).stem}"

    # Para CeramicNet cada archivo es muestra independiente.
    if is_valid(row.get("sample_id")):
        return f"{dataset}::{row.get('sample_id')}"

    return f"{dataset}::{Path(str(row.get('source_path'))).stem}"


def majority_label(labels: pd.Series) -> str:
    return labels.value_counts().index[0]


def make_grouped_split(index: pd.DataFrame) -> dict:
    rng = np.random.default_rng(RANDOM_SEED)

    group_table = (
        index
        .groupby("group_id")
        .agg(
            group_label=("ml_label", majority_label),
            n_samples=("ml_label", "size"),
        )
        .reset_index()
    )

    split_map = {}

    for label, gdf in group_table.groupby("group_label"):
        groups = gdf["group_id"].tolist()
        rng.shuffle(groups)

        n = len(groups)

        if n < 3:
            for g in groups:
                split_map[g] = "train"
            continue

        n_test = max(1, int(round(n * TEST_FRAC)))
        n_val = max(1, int(round(n * VAL_FRAC)))

        while n - n_test - n_val < 1:
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                break

        test_groups = groups[:n_test]
        val_groups = groups[n_test:n_test + n_val]
        train_groups = groups[n_test + n_val:]

        for g in train_groups:
            split_map[g] = "train"

        for g in val_groups:
            split_map[g] = "val"

        for g in test_groups:
            split_map[g] = "test"

    return split_map


def make_label_map(index: pd.DataFrame) -> dict:
    labels = sorted(index["ml_label"].dropna().unique())
    return {label: i for i, label in enumerate(labels)}


def make_class_weights(index: pd.DataFrame) -> dict:
    train = index[index["split"] == "train"].copy()

    counts = train["label_id"].value_counts().sort_index()
    total = counts.sum()
    n_classes = index["label_id"].nunique()

    weights = {}

    for label_id in sorted(index["label_id"].unique()):
        count = counts.get(label_id, 0)
        if count == 0:
            weights[int(label_id)] = 0.0
        else:
            weights[int(label_id)] = float(total / (n_classes * count))

    return weights


def make_balanced_index(index: pd.DataFrame) -> pd.DataFrame:
    parts = []

    limits = {
        "train": MAX_TRAIN_PER_CLASS,
        "val": MAX_VAL_PER_CLASS,
        "test": MAX_TEST_PER_CLASS,
    }

    for split, sdf in index.groupby("split"):
        max_n = limits.get(split, 1000)

        for label, ldf in sdf.groupby("ml_label"):
            if len(ldf) > max_n:
                sampled = ldf.sample(n=max_n, random_state=RANDOM_SEED)
            else:
                sampled = ldf

            parts.append(sampled)

    balanced = pd.concat(parts, ignore_index=True)
    return balanced


def main():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"No encontré: {INDEX_PATH}")

    index = pd.read_csv(INDEX_PATH)

    index.to_csv(BACKUP_PATH, index=False)
    print("Backup guardado en:")
    print(BACKUP_PATH)

    # ========================================================
    # 1. Corregir VoxelFragmentML
    # ========================================================

    mask_vfm = index["dataset"] == "vfm_iberian_vessels"

    parsed = index.loc[mask_vfm, "source_path"].apply(parse_vfm_file)
    parsed_df = pd.DataFrame(parsed.tolist(), index=index.loc[mask_vfm].index)

    for col in parsed_df.columns:
        index.loc[mask_vfm, col] = parsed_df[col]

    # ========================================================
    # 2. Asegurar ml_label para todos
    # ========================================================

    if "ml_label" not in index.columns:
        index["ml_label"] = index["artifact_kind"]

    index["ml_label"] = index["ml_label"].fillna(index["artifact_kind"])
    index["ml_label"] = index["ml_label"].astype(str).str.strip()

    # ========================================================
    # 3. Recalcular group_id para evitar fuga de información
    # ========================================================

    index["group_id"] = index.apply(build_group_id, axis=1)

    # ========================================================
    # 4. Rehacer split agrupado
    # ========================================================

    split_map = make_grouped_split(index)
    index["split"] = index["group_id"].map(split_map)

    if index["split"].isna().any():
        missing = index[index["split"].isna()]
        raise RuntimeError(f"Hay muestras sin split: {len(missing)}")

    # ========================================================
    # 5. Rehacer label_id y label_map
    # ========================================================

    label_to_id = make_label_map(index)
    id_to_label = {v: k for k, v in label_to_id.items()}

    index["label_id"] = index["ml_label"].map(label_to_id).astype(int)

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "label_column_used": "ml_label",
                "note": "VFM corrected: complete vessels vs fragments. Splits rebuilt by group_id.",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # 6. Class weights
    # ========================================================

    class_weights = make_class_weights(index)

    with open(CLASS_WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(class_weights, f, indent=2)

    # ========================================================
    # 7. Guardar index corregido
    # ========================================================

    index.to_csv(INDEX_PATH, index=False)

    # ========================================================
    # 8. Resumen split x etiqueta
    # ========================================================

    split_summary = (
        index
        .groupby(["split", "ml_label"])
        .size()
        .reset_index(name="n_samples")
        .sort_values(["split", "ml_label"])
    )

    split_summary.to_csv(SPLIT_SUMMARY_PATH, index=False)

    # ========================================================
    # 9. Crear index balanceado para primer modelo
    # ========================================================

    balanced = make_balanced_index(index)
    balanced.to_csv(BALANCED_INDEX_PATH, index=False)

    # ========================================================
    # 10. Reporte
    # ========================================================

    print("\nCorrección terminada.")

    print("\nConteo por etiqueta corregida:")
    print(index["ml_label"].value_counts())

    print("\nConteo por split:")
    print(index["split"].value_counts())

    print("\nResumen split x etiqueta:")
    print(split_summary)

    print("\nNúmero de group_id:")
    print(index["group_id"].nunique())

    print("\nConteo balanceado para primer modelo:")
    print(balanced["ml_label"].value_counts())

    print("\nArchivos guardados:")
    print("Index corregido:", INDEX_PATH)
    print("Backup:", BACKUP_PATH)
    print("Label map:", LABEL_MAP_PATH)
    print("Class weights:", CLASS_WEIGHTS_PATH)
    print("Split summary:", SPLIT_SUMMARY_PATH)
    print("Index balanceado:", BALANCED_INDEX_PATH)


if __name__ == "__main__":
    main()