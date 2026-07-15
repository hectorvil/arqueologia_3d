from pathlib import Path
import json
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# RUTAS
# ============================================================

PROJECT_DIR = Path("/Users/hectorperalta/Documents/arqueologia_3d")

MODEL_READY_DIR = PROJECT_DIR / "data" / "model_ready"
INDEX_PATH = MODEL_READY_DIR / "index.csv"
BALANCED_INDEX_PATH = MODEL_READY_DIR / "index_balanced_first_model.csv"

REPORT_DIR = PROJECT_DIR / "reports" / "eda_inicial"
PLOTS_DIR = REPORT_DIR / "plots"
TABLES_DIR = REPORT_DIR / "tables"
EXAMPLES_DIR = REPORT_DIR / "pointcloud_examples"

for d in [REPORT_DIR, PLOTS_DIR, TABLES_DIR, EXAMPLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


RANDOM_SEED = 42
MAX_ROWS_FOR_HIST = 50000
MAX_ROWS_FOR_PCA = 15000


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLES_DIR / name
    df.to_csv(path, index=False)
    return path


def save_series(series: pd.Series, name: str, index_name: str, value_name: str) -> Path:
    df = series.reset_index()
    df.columns = [index_name, value_name]
    return save_table(df, name)


def safe_bool_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "nan": np.nan,
            "none": np.nan,
        })
    )


def resolve_path(path_value) -> Path:
    path = Path(str(path_value))

    if path.exists():
        return path

    candidate = PROJECT_DIR / path
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"No encontré archivo: {path_value}")


def plot_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, out_name: str, top_n=None):
    if top_n is not None:
        series = series.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))

    series.sort_values().plot(kind="barh", ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    out_path = PLOTS_DIR / out_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return out_path


def plot_hist(df: pd.DataFrame, column: str, title: str, out_name: str, log_x=False):
    values = df[column].dropna()

    if len(values) == 0:
        return None

    if len(values) > MAX_ROWS_FOR_HIST:
        values = values.sample(MAX_ROWS_FOR_HIST, random_state=RANDOM_SEED)

    if log_x:
        values = values[values > 0]
        if len(values) == 0:
            return None

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(values, bins=60)

    if log_x:
        ax.set_xscale("log")

    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frecuencia")

    fig.tight_layout()
    out_path = PLOTS_DIR / out_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return out_path


def set_axes_equal_3d(ax, points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_pointcloud(points: np.ndarray, title: str, out_path: Path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=3)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    set_axes_equal_3d(ax, points)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_pca_plot(index: pd.DataFrame):
    candidate_features = [
        "ready_width_x",
        "ready_width_y",
        "ready_width_z",
        "ready_bbox_volume",
        "ready_mean_radius",
        "ready_std_radius",
        "ready_max_radius",
        "ready_pca_lambda_1",
        "ready_pca_lambda_2",
        "ready_pca_lambda_3",
        "input_width_x",
        "input_width_y",
        "input_width_z",
        "input_bbox_volume",
        "input_mean_radius",
        "input_std_radius",
        "input_max_radius",
        "input_pca_lambda_1",
        "input_pca_lambda_2",
        "input_pca_lambda_3",
    ]

    feature_cols = [c for c in candidate_features if c in index.columns]

    if len(feature_cols) < 2:
        print("No hay suficientes features para PCA.")
        return None

    df = index[["sample_id", "ml_label"] + feature_cols].dropna().copy()

    if len(df) < 3:
        print("Muy pocas filas para PCA.")
        return None

    if len(df) > MAX_ROWS_FOR_PCA:
        df = df.sample(MAX_ROWS_FOR_PCA, random_state=RANDOM_SEED)

    X = df[feature_cols].to_numpy(dtype=np.float64)

    # estandarizar
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0

    Xz = (X - X_mean) / X_std

    # PCA con SVD
    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    pcs = U[:, :2] * S[:2]

    df["pc1"] = pcs[:, 0]
    df["pc2"] = pcs[:, 1]

    explained = (S ** 2) / np.sum(S ** 2)
    ev1 = explained[0] * 100
    ev2 = explained[1] * 100

    fig, ax = plt.subplots(figsize=(9, 7))

    for label, g in df.groupby("ml_label"):
        ax.scatter(g["pc1"], g["pc2"], s=8, alpha=0.5, label=label)

    ax.set_title("PCA inicial con features geométricas")
    ax.set_xlabel(f"PC1 ({ev1:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({ev2:.1f}% var.)")

    ax.legend(markerscale=2, fontsize=8, loc="best")

    fig.tight_layout()
    out_path = PLOTS_DIR / "pca_features_geométricas.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    pca_table = df[["sample_id", "ml_label", "pc1", "pc2"]]
    save_table(pca_table, "pca_coordinates.csv")

    return out_path


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nLeyendo index completo:")
    print(INDEX_PATH)

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"No encontré {INDEX_PATH}")

    index = pd.read_csv(INDEX_PATH)

    if "is_fragment" in index.columns:
        index["is_fragment_clean"] = safe_bool_series(index["is_fragment"])
    else:
        index["is_fragment_clean"] = np.nan

    print("\nShape index:")
    print(index.shape)

    print("\nColumnas disponibles:")
    print(index.columns.tolist())

    # ========================================================
    # TABLAS BÁSICAS
    # ========================================================

    counts_dataset = index["dataset"].value_counts()
    counts_label = index["ml_label"].value_counts()
    counts_split = index["split"].value_counts()

    save_series(counts_dataset, "counts_by_dataset.csv", "dataset", "n_samples")
    save_series(counts_label, "counts_by_label.csv", "ml_label", "n_samples")
    save_series(counts_split, "counts_by_split.csv", "split", "n_samples")

    split_label = (
        index
        .groupby(["split", "ml_label"])
        .size()
        .reset_index(name="n_samples")
        .sort_values(["split", "ml_label"])
    )
    save_table(split_label, "counts_by_split_and_label.csv")

    dataset_label = (
        index
        .groupby(["dataset", "ml_label"])
        .size()
        .reset_index(name="n_samples")
        .sort_values(["dataset", "ml_label"])
    )
    save_table(dataset_label, "counts_by_dataset_and_label.csv")

    if "is_fragment_clean" in index.columns:
        fragment_counts = index["is_fragment_clean"].value_counts(dropna=False)
        save_series(fragment_counts, "counts_by_fragment_status.csv", "is_fragment", "n_samples")

    if "group_id" in index.columns:
        group_counts = (
            index
            .groupby(["dataset", "ml_label"])
            .agg(
                n_samples=("sample_id", "size"),
                n_groups=("group_id", "nunique"),
            )
            .reset_index()
            .sort_values(["dataset", "ml_label"])
        )
        save_table(group_counts, "samples_and_groups_by_dataset_label.csv")

    if "vessel_id" in index.columns:
        vfm = index[index["dataset"] == "vfm_iberian_vessels"].copy()

        if len(vfm) > 0:
            vfm_vessel_counts = (
                vfm
                .groupby("vessel_id")
                .size()
                .reset_index(name="n_samples")
                .sort_values("n_samples", ascending=False)
            )
            save_table(vfm_vessel_counts, "vfm_samples_by_vessel_id.csv")

    # ========================================================
    # GRÁFICAS DE CONTEOS
    # ========================================================

    plot_bar(
        counts_dataset,
        title="Número de muestras por dataset",
        xlabel="Muestras",
        ylabel="Dataset",
        out_name="counts_by_dataset.png",
    )

    plot_bar(
        counts_label,
        title="Número de muestras por etiqueta",
        xlabel="Muestras",
        ylabel="Etiqueta",
        out_name="counts_by_label.png",
    )

    plot_bar(
        counts_split,
        title="Número de muestras por split",
        xlabel="Muestras",
        ylabel="Split",
        out_name="counts_by_split.png",
    )

    # Split x label
    split_label_pivot = pd.crosstab(index["ml_label"], index["split"])

    fig, ax = plt.subplots(figsize=(10, 6))
    split_label_pivot.plot(kind="barh", stacked=True, ax=ax)
    ax.set_title("Distribución split x etiqueta")
    ax.set_xlabel("Muestras")
    ax.set_ylabel("Etiqueta")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "split_by_label_stacked.png", dpi=180)
    plt.close(fig)

    # ========================================================
    # HISTOGRAMAS DE FEATURES
    # ========================================================

    hist_cols = [
        "ready_width_x",
        "ready_width_y",
        "ready_width_z",
        "ready_bbox_volume",
        "ready_mean_radius",
        "ready_std_radius",
        "ready_max_radius",
        "input_width_x",
        "input_width_y",
        "input_width_z",
        "input_bbox_volume",
        "input_mean_radius",
        "input_std_radius",
        "input_max_radius",
    ]

    for col in hist_cols:
        if col in index.columns:
            log_x = "volume" in col
            plot_hist(
                index,
                column=col,
                title=f"Distribución de {col}",
                out_name=f"hist_{col}.png",
                log_x=log_x,
            )

    # ========================================================
    # PCA CON FEATURES GEOMÉTRICAS
    # ========================================================

    pca_path = make_pca_plot(index)

    # ========================================================
    # EJEMPLOS VISUALES DE NUBES DE PUNTOS
    # ========================================================

    path_col = None

    if "model_ready_path" in index.columns:
        path_col = "model_ready_path"
    elif "processed_path" in index.columns:
        path_col = "processed_path"

    example_paths = []

    if path_col is not None:
        examples = (
            index
            .groupby("ml_label", group_keys=False)
            .apply(lambda g: g.sample(1, random_state=RANDOM_SEED))
            .reset_index(drop=True)
        )

        for _, row in examples.iterrows():
            label = str(row["ml_label"])
            sample_id = str(row["sample_id"])

            try:
                pc_path = resolve_path(row[path_col])
                points = np.load(pc_path)

                out_path = EXAMPLES_DIR / f"example_{label}_{sample_id}.png"
                plot_pointcloud(
                    points,
                    title=f"{label}\n{sample_id}",
                    out_path=out_path,
                )

                example_paths.append(out_path)

            except Exception as e:
                print(f"No pude graficar {sample_id}: {e}")

    # ========================================================
    # REPORTE MARKDOWN
    # ========================================================

    total_samples = len(index)
    n_labels = index["ml_label"].nunique()
    n_datasets = index["dataset"].nunique()

    n_groups = index["group_id"].nunique() if "group_id" in index.columns else None

    imbalance_ratio = None
    if len(counts_label) > 1 and counts_label.min() > 0:
        imbalance_ratio = counts_label.max() / counts_label.min()

    balanced_info = ""

    if BALANCED_INDEX_PATH.exists():
        balanced = pd.read_csv(BALANCED_INDEX_PATH)
        balanced_counts = balanced["ml_label"].value_counts().to_string()
        balanced_shape = str(balanced.shape)

        balanced_info = (
            "\n## Dataset balanceado / capado\n\n"
            "También existe:\n\n"
            f"{BALANCED_INDEX_PATH}\n\n"
            "Shape:\n\n"
            f"{balanced_shape}\n\n"
            "Conteo por etiqueta en el índice balanceado:\n\n"
            f"{balanced_counts}\n"
        )

        (
            balanced["ml_label"]
            .value_counts()
            .rename_axis("ml_label")
            .reset_index(name="n_samples")
            .to_csv(TABLES_DIR / "balanced_counts_by_label.csv", index=False)
        )

if __name__ == "__main__":
    main()