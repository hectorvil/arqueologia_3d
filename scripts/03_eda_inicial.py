from pathlib import Path
import re
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

REPORT_DIR = PROJECT_DIR / "reports" / "eda_objetos_originales"
PLOTS_DIR = REPORT_DIR / "plots"
TABLES_DIR = REPORT_DIR / "tables"
EXAMPLES_DIR = REPORT_DIR / "pointcloud_examples"
FRAGMENTS_DIR = REPORT_DIR / "fragments_anexo"

for d in [REPORT_DIR, PLOTS_DIR, TABLES_DIR, EXAMPLES_DIR, FRAGMENTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


RANDOM_SEED = 42
MAX_ROWS_FOR_PCA = 15000


# ============================================================
# FUNCIONES
# ============================================================

def clean_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text


def save_table(df: pd.DataFrame, filename: str) -> Path:
    path = TABLES_DIR / filename
    df.to_csv(path, index=False)
    return path


def save_series(series: pd.Series, filename: str, index_name: str, value_name: str) -> Path:
    df = series.reset_index()
    df.columns = [index_name, value_name]
    return save_table(df, filename)


def resolve_path(path_value) -> Path:
    path = Path(str(path_value))

    if path.exists():
        return path

    candidate = PROJECT_DIR / path
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"No encontré archivo: {path_value}")


def get_pointcloud_path_column(df: pd.DataFrame) -> str:
    for col in ["model_ready_path", "processed_path", "npy_path"]:
        if col in df.columns:
            return col

    raise ValueError(
        "No encontré columna de ruta de nubes. "
        "Esperaba model_ready_path, processed_path o npy_path."
    )


def filter_original_objects(index: pd.DataFrame) -> pd.DataFrame:
    """
    EDA principal:
    - Todo CeramicNet cuenta como objeto base.
    - De VoxelFragmentML solo usamos vasijas completas.
    - Excluimos fragmentos derivados.
    """
    mask_ceramicnet = index["dataset"] == "ceramicnet_sue_ware"

    mask_vfm_complete = (
        (index["dataset"] == "vfm_iberian_vessels")
        & (index["ml_label"] == "iberian_vessel_complete")
    )

    objects = index[mask_ceramicnet | mask_vfm_complete].copy()

    return objects


def filter_fragments(index: pd.DataFrame) -> pd.DataFrame:
    fragments = index[index["ml_label"] == "iberian_vessel_fragment"].copy()
    return fragments


def plot_barh(series: pd.Series, title: str, xlabel: str, ylabel: str, filename: str):
    if len(series) == 0:
        return None

    fig_height = max(5, 0.55 * len(series))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    series.sort_values().plot(kind="barh", ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()

    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return out_path


def plot_stacked_split_by_label(df: pd.DataFrame, filename: str):
    if len(df) == 0:
        return None

    pivot = pd.crosstab(df["ml_label"], df["split"])

    fig_height = max(5, 0.6 * len(pivot))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    pivot.plot(kind="barh", stacked=True, ax=ax)

    ax.set_title("Distribución de split por etiqueta — objetos originales")
    ax.set_xlabel("Número de objetos")
    ax.set_ylabel("Etiqueta")

    fig.tight_layout()

    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return out_path


def plot_hist(df: pd.DataFrame, column: str, title: str, filename: str, log_x: bool = False):
    if column not in df.columns:
        return None

    values = df[column].dropna()

    if len(values) == 0:
        return None

    if log_x:
        values = values[values > 0]

    if len(values) == 0:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(values, bins=40)

    if log_x:
        ax.set_xscale("log")

    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frecuencia")

    fig.tight_layout()

    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return out_path


def set_axes_equal_3d(ax, points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    centers = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2

    if radius == 0:
        radius = 1.0

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


def make_examples(df: pd.DataFrame, path_col: str):
    examples = (
        df
        .groupby("ml_label", group_keys=False)
        .apply(lambda g: g.sample(1, random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )

    example_records = []

    for _, row in examples.iterrows():
        label = str(row["ml_label"])
        sample_id = str(row["sample_id"])

        try:
            pc_path = resolve_path(row[path_col])
            points = np.load(pc_path)

            filename = f"example_{clean_filename(label)}_{clean_filename(sample_id)}.png"
            out_path = EXAMPLES_DIR / filename

            plot_pointcloud(
                points=points,
                title=f"{label}\n{sample_id}",
                out_path=out_path,
            )

            example_records.append({
                "sample_id": sample_id,
                "ml_label": label,
                "image_path": str(out_path),
                "pointcloud_path": str(pc_path),
            })

        except Exception as e:
            print(f"No pude graficar {sample_id}: {e}")

    if example_records:
        save_table(pd.DataFrame(example_records), "pointcloud_examples.csv")


def make_pca_plot(df: pd.DataFrame):
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

    feature_cols = [c for c in candidate_features if c in df.columns]

    if len(feature_cols) < 2:
        print("No hay suficientes features para PCA.")
        return None

    pca_df = df[["sample_id", "ml_label"] + feature_cols].dropna().copy()

    if len(pca_df) < 3:
        print("Muy pocas filas para PCA.")
        return None

    if len(pca_df) > MAX_ROWS_FOR_PCA:
        pca_df = pca_df.sample(MAX_ROWS_FOR_PCA, random_state=RANDOM_SEED)

    X = pca_df[feature_cols].to_numpy(dtype=np.float64)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0

    Xz = (X - X_mean) / X_std

    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    pcs = U[:, :2] * S[:2]

    explained = (S ** 2) / np.sum(S ** 2)

    pca_df["pc1"] = pcs[:, 0]
    pca_df["pc2"] = pcs[:, 1]

    save_table(pca_df[["sample_id", "ml_label", "pc1", "pc2"]], "pca_original_objects_coordinates.csv")

    fig, ax = plt.subplots(figsize=(9, 7))

    for label, g in pca_df.groupby("ml_label"):
        ax.scatter(g["pc1"], g["pc2"], s=18, alpha=0.65, label=label)

    ax.set_title("PCA inicial con features geométricas — objetos originales")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var.)")

    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()

    out_path = PLOTS_DIR / "pca_original_objects_features.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return out_path


def make_fragments_anexo(fragments: pd.DataFrame):
    """
    Reporte secundario de fragmentos derivados.
    No domina el EDA principal.
    """
    if len(fragments) == 0:
        return

    fragments_counts_split = fragments["split"].value_counts()
    fragments_counts_dataset = fragments["dataset"].value_counts()

    fragments_counts_split.reset_index().to_csv(
        FRAGMENTS_DIR / "fragment_counts_by_split.csv",
        index=False,
    )

    fragments_counts_dataset.reset_index().to_csv(
        FRAGMENTS_DIR / "fragment_counts_by_dataset.csv",
        index=False,
    )

    if "vessel_id" in fragments.columns:
        fragments_by_vessel = (
            fragments
            .groupby("vessel_id")
            .size()
            .reset_index(name="n_fragment_samples")
            .sort_values("n_fragment_samples", ascending=False)
        )

        fragments_by_vessel.to_csv(
            FRAGMENTS_DIR / "fragments_by_vessel_id.csv",
            index=False,
        )

        desc = fragments_by_vessel["n_fragment_samples"].describe().reset_index()
        desc.columns = ["statistic", "value"]
        desc.to_csv(FRAGMENTS_DIR / "fragments_by_vessel_id_describe.csv", index=False)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(fragments_by_vessel["n_fragment_samples"], bins=30)
        ax.set_title("Fragmentos derivados por vasija base")
        ax.set_xlabel("Número de fragmentos/nubes derivadas")
        ax.set_ylabel("Frecuencia de vasijas base")
        fig.tight_layout()
        fig.savefig(FRAGMENTS_DIR / "hist_fragments_by_vessel_id.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    fragments_counts_split.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Fragmentos derivados por split")
    ax.set_xlabel("Número de muestras derivadas")
    ax.set_ylabel("Split")
    fig.tight_layout()
    fig.savefig(FRAGMENTS_DIR / "fragment_counts_by_split.png", dpi=180)
    plt.close(fig)


def make_report(
    index_all: pd.DataFrame,
    objects: pd.DataFrame,
    fragments: pd.DataFrame,
    counts_dataset: pd.Series,
    counts_label: pd.Series,
    counts_split: pd.Series,
    split_label: pd.DataFrame,
):
    n_all = len(index_all)
    n_objects = len(objects)
    n_fragments = len(fragments)

    n_groups_all = index_all["group_id"].nunique() if "group_id" in index_all.columns else None
    n_groups_objects = objects["group_id"].nunique() if "group_id" in objects.columns else None

    imbalance_ratio = None
    if len(counts_label) > 1 and counts_label.min() > 0:
        imbalance_ratio = counts_label.max() / counts_label.min()

    balanced_info = ""

    if BALANCED_INDEX_PATH.exists():
        balanced = pd.read_csv(BALANCED_INDEX_PATH)
        balanced_counts = balanced["ml_label"].value_counts().to_string()

        balanced_info = (
            "\n## Índice balanceado/capado para modelado\n\n"
            "Existe un índice separado para primeros modelos:\n\n"
            f"{BALANCED_INDEX_PATH}\n\n"
            f"Shape: {balanced.shape}\n\n"
            "Conteo por etiqueta:\n\n"
            f"{balanced_counts}\n"
        )

        (
            balanced["ml_label"]
            .value_counts()
            .rename_axis("ml_label")
            .reset_index(name="n_samples")
            .to_csv(TABLES_DIR / "balanced_counts_by_label.csv", index=False)
        )

    lines = []

    lines.append("# EDA inicial — objetos arqueológicos 3D originales/base")
    lines.append("")
    lines.append("## Criterio de análisis")
    lines.append("")
    lines.append(
        "El EDA principal se realizó sobre objetos base/originales. "
        "Se excluyeron del reporte principal los fragmentos derivados de VoxelFragmentML "
        "para evitar que dominen artificialmente las gráficas."
    )
    lines.append("")
    lines.append("Incluido en EDA principal:")
    lines.append("")
    lines.append("- CeramicNet completo: piezas Sue ware.")
    lines.append("- VoxelFragmentML: solo vasijas completas `iberian_vessel_complete`.")
    lines.append("")
    lines.append("Excluido del EDA principal:")
    lines.append("")
    lines.append("- `iberian_vessel_fragment`, porque son muestras derivadas por fragmentación.")
    lines.append("")
    lines.append("## Resumen")
    lines.append("")
    lines.append(f"Total de registros en `index.csv`: {n_all}")
    lines.append("")
    lines.append(f"Objetos base/originales analizados: {n_objects}")
    lines.append("")
    lines.append(f"Fragmentos derivados excluidos del EDA principal: {n_fragments}")
    lines.append("")
    lines.append(f"`group_id` totales en index completo: {n_groups_all}")
    lines.append("")
    lines.append(f"`group_id` en objetos originales/base: {n_groups_objects}")
    lines.append("")
    lines.append("## Conteo por dataset — objetos originales")
    lines.append("")
    lines.append("```text")
    lines.append(counts_dataset.to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Conteo por etiqueta — objetos originales")
    lines.append("")
    lines.append("```text")
    lines.append(counts_label.to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Conteo por split — objetos originales")
    lines.append("")
    lines.append("```text")
    lines.append(counts_split.to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Split x etiqueta — objetos originales")
    lines.append("")
    lines.append("```text")
    lines.append(split_label.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Desbalance entre clases")
    lines.append("")
    lines.append(f"Razón aproximada max/min entre clases del EDA principal: {imbalance_ratio}")
    lines.append("")
    lines.append(
        "Este desbalance ya es mucho más interpretable que el reporte anterior, "
        "porque ahora describe objetos base y no miles de fragmentos derivados."
    )
    lines.append("")
    lines.append(balanced_info)
    lines.append("")
    lines.append("## Archivos generados")
    lines.append("")
    lines.append(f"Tablas: `{TABLES_DIR}`")
    lines.append("")
    lines.append(f"Gráficas principales: `{PLOTS_DIR}`")
    lines.append("")
    lines.append(f"Ejemplos visuales: `{EXAMPLES_DIR}`")
    lines.append("")
    lines.append(f"Anexo de fragmentos derivados: `{FRAGMENTS_DIR}`")
    lines.append("")
    lines.append("## Gráficas principales")
    lines.append("")
    lines.append("- `original_counts_by_dataset.png`")
    lines.append("- `original_counts_by_label.png`")
    lines.append("- `original_counts_by_split.png`")
    lines.append("- `original_split_by_label_stacked.png`")
    lines.append("- `pca_original_objects_features.png`")
    lines.append("")
    lines.append("## Lectura recomendada")
    lines.append("")
    lines.append(
        "Para describir la colección arqueológica, usar este reporte de objetos originales. "
        "Para discutir el volumen de datos disponible para entrenamiento, usar el anexo de fragmentos."
    )

    report_path = REPORT_DIR / "eda_original_objects_summary.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")

    return report_path


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nLeyendo index:")
    print(INDEX_PATH)

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"No encontré {INDEX_PATH}")

    index_all = pd.read_csv(INDEX_PATH)

    if "ml_label" not in index_all.columns:
        raise ValueError("No existe ml_label en index.csv. Revisa el script 02A.")

    if "dataset" not in index_all.columns:
        raise ValueError("No existe dataset en index.csv.")

    path_col = get_pointcloud_path_column(index_all)

    objects = filter_original_objects(index_all)
    fragments = filter_fragments(index_all)

    print("\nTotal index completo:", len(index_all))
    print("Objetos originales/base:", len(objects))
    print("Fragmentos derivados:", len(fragments))

    print("\nConteo objetos originales por etiqueta:")
    print(objects["ml_label"].value_counts())

    # ========================================================
    # TABLAS PRINCIPALES
    # ========================================================

    counts_dataset = objects["dataset"].value_counts()
    counts_label = objects["ml_label"].value_counts()
    counts_split = objects["split"].value_counts()

    save_series(counts_dataset, "original_counts_by_dataset.csv", "dataset", "n_objects")
    save_series(counts_label, "original_counts_by_label.csv", "ml_label", "n_objects")
    save_series(counts_split, "original_counts_by_split.csv", "split", "n_objects")

    split_label = (
        objects
        .groupby(["split", "ml_label"])
        .size()
        .reset_index(name="n_objects")
        .sort_values(["split", "ml_label"])
    )

    save_table(split_label, "original_counts_by_split_and_label.csv")

    dataset_label = (
        objects
        .groupby(["dataset", "ml_label"])
        .size()
        .reset_index(name="n_objects")
        .sort_values(["dataset", "ml_label"])
    )

    save_table(dataset_label, "original_counts_by_dataset_and_label.csv")

    if "group_id" in objects.columns:
        group_summary = (
            objects
            .groupby(["dataset", "ml_label"])
            .agg(
                n_objects=("sample_id", "size"),
                n_groups=("group_id", "nunique"),
            )
            .reset_index()
        )

        save_table(group_summary, "original_objects_group_summary.csv")

    # ========================================================
    # GRÁFICAS PRINCIPALES
    # ========================================================

    plot_barh(
        counts_dataset,
        title="Número de objetos originales por dataset",
        xlabel="Objetos base/originales",
        ylabel="Dataset",
        filename="original_counts_by_dataset.png",
    )

    plot_barh(
        counts_label,
        title="Número de objetos originales por etiqueta",
        xlabel="Objetos base/originales",
        ylabel="Etiqueta",
        filename="original_counts_by_label.png",
    )

    plot_barh(
        counts_split,
        title="Número de objetos originales por split",
        xlabel="Objetos base/originales",
        ylabel="Split",
        filename="original_counts_by_split.png",
    )

    plot_stacked_split_by_label(
        objects,
        filename="original_split_by_label_stacked.png",
    )

    # ========================================================
    # HISTOGRAMAS DE FEATURES SOLO PARA ORIGINALES
    # ========================================================

    hist_cols = [
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
    ]

    for col in hist_cols:
        if col in objects.columns:
            plot_hist(
                objects,
                column=col,
                title=f"Distribución de {col} — objetos originales",
                filename=f"hist_original_{col}.png",
                log_x=("volume" in col),
            )

    # ========================================================
    # PCA SOLO CON ORIGINALES
    # ========================================================

    pca_path = make_pca_plot(objects)

    # ========================================================
    # EJEMPLOS VISUALES SOLO ORIGINALES
    # ========================================================

    make_examples(objects, path_col=path_col)

    # ========================================================
    # ANEXO DE FRAGMENTOS
    # ========================================================

    make_fragments_anexo(fragments)

    # ========================================================
    # REPORTE
    # ========================================================

    report_path = make_report(
        index_all=index_all,
        objects=objects,
        fragments=fragments,
        counts_dataset=counts_dataset,
        counts_label=counts_label,
        counts_split=counts_split,
        split_label=split_label,
    )

    print("\n===================================================")
    print("EDA DE OBJETOS ORIGINALES TERMINADO")
    print("===================================================")

    print("\nReporte principal:")
    print(report_path)

    print("\nTablas:")
    print(TABLES_DIR)

    print("\nGráficas:")
    print(PLOTS_DIR)

    print("\nEjemplos visuales:")
    print(EXAMPLES_DIR)

    print("\nAnexo de fragmentos:")
    print(FRAGMENTS_DIR)

    if pca_path:
        print("\nPCA:")
        print(pca_path)

    print("\nListo.")


if __name__ == "__main__":
    main()