from pathlib import Path
from typing import Dict, List, Tuple
import json
import hashlib
import re

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_DIR = Path("/Users/hectorperalta/Documents/arqueologia_3d")

INPUT_METADATA = PROJECT_DIR / "data" / "processed" / "metadata.csv"
INPUT_POINTCLOUD_DIR = PROJECT_DIR / "data" / "processed" / "pointclouds"

OUT_DIR = PROJECT_DIR / "data" / "model_ready"
OUT_POINTCLOUD_DIR = OUT_DIR / "pointclouds"

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_POINTCLOUD_DIR.mkdir(parents=True, exist_ok=True)

N_POINTS = 1024

# Para un primer modelo supervisado, artifact_kind es la opción más directa.
# Ejemplos esperados:
# bowl, plate, dish_body, dish_cap, vessel_fragment, iberian_vessel, etc.
LABEL_COLUMN = "artifact_kind"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

RANDOM_SEED = 42

# Para entrenamiento supervisado conviene eliminar clases con poquísimos grupos.
# Si no quieres eliminar nada, cambia esto a False.
DROP_CLASSES_WITH_FEW_GROUPS = True
MIN_GROUPS_PER_CLASS = 3

# Si el dataset es muy grande, deja esto en False.
# Si lo pones True, además de los .npy separados crea un único .npz.
CREATE_COMPRESSED_NPZ = False


# ============================================================
# UTILIDADES
# ============================================================

def stable_seed(text: str, base_seed: int = 42) -> int:
    """
    Crea una semilla estable a partir de un texto.
    Así el remuestreo de cada muestra es reproducible.
    """
    key = f"{base_seed}_{text}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    return int(digest[:8], 16)


def sanitize_id(value: object) -> str:
    """
    Convierte un sample_id en un nombre seguro para archivo.
    """
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value


def is_valid_value(x: object) -> bool:
    if pd.isna(x):
        return False

    s = str(x).strip()

    if s == "":
        return False

    if s.lower() in {"nan", "none", "null"}:
        return False

    return True


def resolve_pointcloud_path(path_value: object) -> Path:
    """
    Resuelve rutas absolutas o relativas.
    Sirve por si metadata.csv tiene processed_path absoluto o relativo.
    """
    if not is_valid_value(path_value):
        raise ValueError("Ruta vacía o inválida.")

    path = Path(str(path_value))

    if path.exists():
        return path

    candidate = PROJECT_DIR / path
    if candidate.exists():
        return candidate

    candidate = INPUT_POINTCLOUD_DIR / path.name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"No encontré el archivo: {path_value}")


def clean_points(points: np.ndarray) -> np.ndarray:
    """
    Asegura matriz Nx3 y elimina puntos no finitos.
    """
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2:
        raise ValueError(f"La nube debe ser 2D, pero llegó con shape {points.shape}")

    if points.shape[1] < 3:
        raise ValueError(f"La nube debe tener al menos 3 columnas, pero tiene {points.shape[1]}")

    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]

    if len(points) == 0:
        raise ValueError("La nube quedó vacía después de limpiar NaN/inf.")

    return points.astype(np.float32)


def compute_features(points: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Features simples que nos van a servir para EDA.
    """
    points = clean_points(points)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    widths = maxs - mins

    centroid = points.mean(axis=0)
    radii = np.linalg.norm(points - centroid, axis=1)

    covariance = np.cov(points.T)
    eigvals = np.linalg.eigvalsh(covariance)
    eigvals = np.sort(eigvals)[::-1]

    return {
        f"{prefix}_n_points": int(len(points)),

        f"{prefix}_min_x": float(mins[0]),
        f"{prefix}_min_y": float(mins[1]),
        f"{prefix}_min_z": float(mins[2]),

        f"{prefix}_max_x": float(maxs[0]),
        f"{prefix}_max_y": float(maxs[1]),
        f"{prefix}_max_z": float(maxs[2]),

        f"{prefix}_width_x": float(widths[0]),
        f"{prefix}_width_y": float(widths[1]),
        f"{prefix}_width_z": float(widths[2]),

        f"{prefix}_bbox_volume": float(widths[0] * widths[1] * widths[2]),

        f"{prefix}_mean_radius": float(radii.mean()),
        f"{prefix}_std_radius": float(radii.std()),
        f"{prefix}_max_radius": float(radii.max()),

        f"{prefix}_pca_lambda_1": float(eigvals[0]),
        f"{prefix}_pca_lambda_2": float(eigvals[1]),
        f"{prefix}_pca_lambda_3": float(eigvals[2]),
    }


def standardize_pointcloud(
    points: np.ndarray,
    sample_id: str,
    n_points: int = 1024,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Transformación determinística aplicada a TODAS las muestras:

    1. limpiar
    2. centrar
    3. escalar a esfera unitaria
    4. remuestrear a 1024 puntos
    5. barajar puntos de forma reproducible

    Devuelve:
    - points_final: matriz 1024 x 3
    - info: centroid y escala usados
    """
    points = clean_points(points)

    centroid = points.mean(axis=0)
    centered = points - centroid

    scale = np.max(np.linalg.norm(centered, axis=1))

    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered

    normalized = normalized.astype(np.float32)

    seed = stable_seed(sample_id, RANDOM_SEED)
    rng = np.random.default_rng(seed)

    n = len(normalized)

    if n > n_points:
        idx = rng.choice(n, size=n_points, replace=False)
        points_fixed = normalized[idx]
    elif n < n_points:
        idx = rng.choice(n, size=n_points, replace=True)
        points_fixed = normalized[idx]
    else:
        points_fixed = normalized

    # El orden de puntos no debería importar.
    # Aun así, lo barajamos de forma reproducible para evitar patrones de orden.
    perm = rng.permutation(n_points)
    points_fixed = points_fixed[perm]

    info = {
        "standard_centroid_x": float(centroid[0]),
        "standard_centroid_y": float(centroid[1]),
        "standard_centroid_z": float(centroid[2]),
        "standard_scale": float(scale),
    }

    return points_fixed.astype(np.float32), info


def build_group_id(row: pd.Series) -> str:
    """
    Agrupa muestras que no deben separarse entre train/val/test.

    Para VoxelFragmentML, varios fragmentos pueden venir de la misma vasija.
    Por eso usamos vessel_id cuando existe.

    Para CeramicNet, normalmente cada archivo es una muestra independiente.
    """
    dataset = str(row.get("dataset", "unknown_dataset"))

    if "vessel_id" in row and is_valid_value(row.get("vessel_id")):
        return f"{dataset}::{row.get('vessel_id')}"

    if "source_path" in row and is_valid_value(row.get("source_path")):
        stem = Path(str(row.get("source_path"))).stem
        return f"{dataset}::{stem}"

    return f"{dataset}::{row.get('sample_id')}"


def majority_label(labels: pd.Series) -> str:
    """
    Etiqueta mayoritaria de un grupo.
    """
    return labels.value_counts().index[0]


def make_grouped_split(
    metadata: pd.DataFrame,
    label_col: str = "ml_label",
    group_col: str = "group_id",
) -> Dict[str, str]:
    """
    Crea splits por grupo para evitar leakage.

    Todas las muestras con el mismo group_id quedan en el mismo split.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    group_table = (
        metadata
        .groupby(group_col)
        .agg(
            group_label=(label_col, majority_label),
            n_samples=(label_col, "size"),
        )
        .reset_index()
    )

    split_map = {}

    for label, label_groups in group_table.groupby("group_label"):
        groups = label_groups[group_col].tolist()
        rng.shuffle(groups)

        n = len(groups)

        if n < 3:
            # No hay suficientes grupos para train/val/test.
            # Los dejamos en train.
            for g in groups:
                split_map[g] = "train"
            continue

        n_test = int(round(n * TEST_FRAC))
        n_val = int(round(n * VAL_FRAC))

        n_test = max(1, n_test)
        n_val = max(1, n_val)

        # Asegurar que quede al menos un grupo para train.
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


def make_class_weights(index: pd.DataFrame) -> Dict[str, float]:
    """
    Pesos inversos por clase, calculados con train.
    Útiles después para modelos con clases desbalanceadas.
    """
    train = index[index["split"] == "train"].copy()

    counts = train["label_id"].value_counts().sort_index()
    total = counts.sum()
    n_classes = index["label_id"].nunique()

    weights_by_id = {}

    for label_id in sorted(index["label_id"].unique()):
        count = counts.get(label_id, 0)

        if count == 0:
            weights_by_id[int(label_id)] = 0.0
        else:
            weights_by_id[int(label_id)] = float(total / (n_classes * count))

    return weights_by_id


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("\nLeyendo metadata:")
    print(INPUT_METADATA)

    if not INPUT_METADATA.exists():
        raise FileNotFoundError(
            f"No encontré metadata.csv en {INPUT_METADATA}. "
            "Primero corre el script 01_unificar_datos.py"
        )

    metadata = pd.read_csv(INPUT_METADATA)

    if "processed_path" in metadata.columns:
        input_path_col = "processed_path"
    elif "npy_path" in metadata.columns:
        input_path_col = "npy_path"
    else:
        raise ValueError(
            "No encontré columna processed_path ni npy_path en metadata.csv"
        )

    if LABEL_COLUMN not in metadata.columns:
        raise ValueError(
            f"No encontré la columna {LABEL_COLUMN}. "
            f"Columnas disponibles: {metadata.columns.tolist()}"
        )

    # Crear etiqueta ML.
    metadata["ml_label"] = (
        metadata[LABEL_COLUMN]
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )

    # Crear sample_id si no existiera.
    if "sample_id" not in metadata.columns:
        metadata["sample_id"] = [
            f"sample_{i:06d}" for i in range(len(metadata))
        ]

    metadata["sample_id"] = metadata["sample_id"].apply(sanitize_id)

    # Crear group_id para split sin leakage.
    metadata["group_id"] = metadata.apply(build_group_id, axis=1)

    # Opcional: quitar clases con muy pocos grupos.
    if DROP_CLASSES_WITH_FEW_GROUPS:
        group_counts_per_label = (
            metadata[["ml_label", "group_id"]]
            .drop_duplicates()
            .groupby("ml_label")
            .size()
            .sort_values(ascending=False)
        )

        valid_labels = group_counts_per_label[
            group_counts_per_label >= MIN_GROUPS_PER_CLASS
        ].index.tolist()

        dropped_labels = sorted(set(metadata["ml_label"]) - set(valid_labels))

        if dropped_labels:
            print("\nClases eliminadas por tener pocos grupos:")
            for label in dropped_labels:
                print(" -", label)

        metadata = metadata[metadata["ml_label"].isin(valid_labels)].copy()

    print("\nMuestras después de filtrar:", len(metadata))
    print("\nConteo por etiqueta:")
    print(metadata["ml_label"].value_counts())

    records: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []

    print("\nEstandarizando y copiando nubes a data/model_ready/pointclouds...")

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        sample_id = sanitize_id(row["sample_id"])

        try:
            input_path = resolve_pointcloud_path(row[input_path_col])
            points_input = np.load(input_path)

            input_features = compute_features(points_input, prefix="input")

            points_ready, transform_info = standardize_pointcloud(
                points=points_input,
                sample_id=sample_id,
                n_points=N_POINTS,
            )

            ready_features = compute_features(points_ready, prefix="ready")

            output_path = OUT_POINTCLOUD_DIR / f"{sample_id}.npy"
            np.save(output_path, points_ready)

            record = row.to_dict()

            record.update({
                "model_ready_path": str(output_path),
                "model_n_points": int(points_ready.shape[0]),
                "model_n_dimensions": int(points_ready.shape[1]),
                "model_format": "npy_float32_1024x3",
            })

            record.update(transform_info)
            record.update(input_features)
            record.update(ready_features)

            records.append(record)

        except Exception as e:
            errors.append({
                "sample_id": sample_id,
                "error": str(e),
                "input_path": row.get(input_path_col),
            })

    index = pd.DataFrame(records)

    if len(index) == 0:
        raise RuntimeError("No se pudo procesar ninguna nube de puntos.")

    # Crear label map.
    labels = sorted(index["ml_label"].unique())
    label_map = {label: i for i, label in enumerate(labels)}
    inverse_label_map = {i: label for label, i in label_map.items()}

    index["label_id"] = index["ml_label"].map(label_map).astype(int)

    # Crear split group-aware.
    split_map = make_grouped_split(index)
    index["split"] = index["group_id"].map(split_map)

    # Guardar index.
    index_path = OUT_DIR / "index.csv"
    index.to_csv(index_path, index=False)

    # Guardar label map.
    label_map_path = OUT_DIR / "label_map.json"

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_to_id": label_map,
                "id_to_label": inverse_label_map,
                "label_column_used": LABEL_COLUMN,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Guardar pesos de clase.
    class_weights = make_class_weights(index)
    class_weights_path = OUT_DIR / "class_weights.json"

    with open(class_weights_path, "w", encoding="utf-8") as f:
        json.dump(class_weights, f, indent=2)

    # Guardar resumen de splits.
    split_summary = (
        index
        .groupby(["split", "ml_label"])
        .size()
        .reset_index(name="n_samples")
        .sort_values(["split", "ml_label"])
    )

    split_summary_path = OUT_DIR / "split_summary.csv"
    split_summary.to_csv(split_summary_path, index=False)

    # Guardar errores, si hubo.
    if errors:
        errors_path = OUT_DIR / "processing_errors.csv"
        pd.DataFrame(errors).to_csv(errors_path, index=False)
    else:
        errors_path = None

    # Opcional: crear NPZ único.
    if CREATE_COMPRESSED_NPZ:
        print("\nCreando archivo comprimido NPZ. Esto puede tardar...")

        X = []
        y = []
        split = []
        sample_ids = []

        for _, row in tqdm(index.iterrows(), total=len(index)):
            X.append(np.load(row["model_ready_path"]))
            y.append(int(row["label_id"]))
            split.append(row["split"])
            sample_ids.append(row["sample_id"])

        X = np.stack(X).astype(np.float32)
        y = np.array(y, dtype=np.int64)
        split = np.array(split)
        sample_ids = np.array(sample_ids)

        npz_path = OUT_DIR / "dataset_model_ready.npz"

        np.savez_compressed(
            npz_path,
            X=X,
            y=y,
            split=split,
            sample_id=sample_ids,
        )

        print("NPZ guardado en:", npz_path)

    print("\n===================================================")
    print("DATASET MODEL-READY CREADO")
    print("===================================================")
    print("Index:", index_path)
    print("Pointclouds:", OUT_POINTCLOUD_DIR)
    print("Label map:", label_map_path)
    print("Class weights:", class_weights_path)
    print("Split summary:", split_summary_path)

    if errors_path:
        print("Errores:", errors_path)

    print("\nTotal muestras:", len(index))

    print("\nConteo por split:")
    print(index["split"].value_counts())

    print("\nConteo por etiqueta:")
    print(index["ml_label"].value_counts())

    print("\nResumen split x etiqueta:")
    print(split_summary)

    example = np.load(index.iloc[0]["model_ready_path"])
    print("\nEjemplo:")
    print("Archivo:", index.iloc[0]["model_ready_path"])
    print("Shape:", example.shape)
    print("Dtype:", example.dtype)


if __name__ == "__main__":
    main()