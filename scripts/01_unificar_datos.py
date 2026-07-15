from pathlib import Path
from typing import Dict, Optional, Tuple
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from plyfile import PlyData


# ============================================================
# RUTAS EXACTAS DE TUS DATOS
# ============================================================

CERAMICNET_DIR = Path(
    "/Users/hectorperalta/Downloads/"
    "dv-wataru-tatsuda-ceramicnet-supplement-067914c/"
    "ceramicnet_data"
)

VFM_DIR = Path(
    "/Users/hectorperalta/Downloads/Vessels_200_obj_ply"
)

PROJECT_DIR = Path("/Users/hectorperalta/Documents/arqueologia_3d")
OUT_DIR = PROJECT_DIR / "data" / "processed"
POINTCLOUDS_DIR = OUT_DIR / "pointclouds"

POINTCLOUDS_DIR.mkdir(parents=True, exist_ok=True)

N_POINTS = 1024


# ============================================================
# ETIQUETAS DE CERAMICNET
# ============================================================

CERAMICNET_LABELS = {
    "B": "bowl",
    "DB": "dish_body",
    "DBR": "dish_body_with_ring_base",
    "DC": "dish_cap",
    "P": "plate",
}


# ============================================================
# FUNCIONES DE LECTURA
# ============================================================

def read_ceramicnet_txt(path: Path) -> np.ndarray:
    """
    Lee archivos .txt de CeramicNet.
    Cada archivo tiene columnas sin encabezado: x y z.
    """
    try:
        points = np.loadtxt(path)
    except Exception:
        points = np.loadtxt(path, delimiter=",")

    if points.ndim == 1:
        points = points.reshape(-1, 3)

    if points.shape[1] < 3:
        raise ValueError(f"Archivo con menos de 3 columnas: {path}")

    return points[:, :3].astype(np.float32)


def read_ply_pointcloud(path: Path) -> np.ndarray:
    """
    Lee archivos .ply de VoxelFragmentML.
    Toma únicamente los vértices x, y, z.
    """
    ply = PlyData.read(path)

    if "vertex" not in ply:
        raise ValueError(f"No tiene elemento vertex: {path}")

    vertex = ply["vertex"]

    required = ["x", "y", "z"]
    available = vertex.data.dtype.names

    for col in required:
        if col not in available:
            raise ValueError(f"Falta columna {col} en {path}")

    points = np.vstack([
        vertex["x"],
        vertex["y"],
        vertex["z"],
    ]).T

    return points[:, :3].astype(np.float32)


# ============================================================
# FUNCIONES PARA ESTANDARIZAR
# ============================================================

def clean_points(points: np.ndarray) -> np.ndarray:
    """
    Elimina puntos con NaN o infinito.
    """
    points = np.asarray(points, dtype=np.float32)
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]

    if len(points) == 0:
        raise ValueError("La nube quedó vacía después de limpiar.")

    return points


def normalize_pointcloud(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Centra la nube en 0 y la escala a una esfera unitaria.
    Esto permite comparar formas sin depender de posición o escala original.
    """
    points = clean_points(points)

    centroid = points.mean(axis=0)
    centered = points - centroid

    scale = np.max(np.linalg.norm(centered, axis=1))

    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered

    return normalized.astype(np.float32), centroid.astype(np.float32), float(scale)


def force_n_points(points: np.ndarray, n_points: int = 1024, seed: int = 42) -> np.ndarray:
    """
    Fuerza todas las nubes al mismo tamaño: 1024 x 3.
    """
    rng = np.random.default_rng(seed)

    points = clean_points(points)
    n = len(points)

    if n == n_points:
        return points.astype(np.float32)

    if n > n_points:
        idx = rng.choice(n, size=n_points, replace=False)
    else:
        idx = rng.choice(n, size=n_points, replace=True)

    return points[idx].astype(np.float32)


def compute_features(points: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Calcula variables básicas útiles para EDA.
    """
    points = clean_points(points)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    widths = maxs - mins

    centroid = points.mean(axis=0)
    radii = np.linalg.norm(points - centroid, axis=1)

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
    }


# ============================================================
# PARSEO DE NOMBRES DE VoxelFragmentML
# ============================================================

def parse_vfm_filename(path: Path) -> Dict[str, Optional[str]]:
    """
    Interpreta nombres como:

    AL_03H_1024p.ply
    AL_03H_10f_128_0it_0_1024p.ply
    """
    name = path.name

    fragment_pattern = re.compile(
        r"^(?P<vessel_id>.+)_"
        r"(?P<n_fragments>\d+)f_"
        r"(?P<resolution>\d+)_"
        r"(?P<iteration>\d+)it_"
        r"(?P<fragment_id>\d+)_"
        r"1024p\.ply$"
    )

    full_pattern = re.compile(
        r"^(?P<vessel_id>.+)_1024p\.ply$"
    )

    match = fragment_pattern.fullmatch(name)

    if match:
        info = match.groupdict()
        info["is_fragment"] = "True"
        info["artifact_kind"] = "vessel_fragment"
        return info

    match = full_pattern.fullmatch(name)

    if match:
        info = match.groupdict()
        info["n_fragments"] = None
        info["resolution"] = None
        info["iteration"] = None
        info["fragment_id"] = None
        info["is_fragment"] = "False"
        info["artifact_kind"] = "iberian_vessel"
        return info

    return {
        "vessel_id": path.stem,
        "n_fragments": None,
        "resolution": None,
        "iteration": None,
        "fragment_id": None,
        "is_fragment": None,
        "artifact_kind": "unknown_vfm_pointcloud",
    }


# ============================================================
# GUARDADO ESTÁNDAR
# ============================================================

def process_and_save(
    points_raw: np.ndarray,
    sample_id: str,
    dataset: str,
    artifact_kind: str,
    label_original: str,
    is_fragment: bool,
    source_path: Path,
    raw_format: str,
    extra_metadata: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, object]:
    """
    Guarda cada nube como .npy estándar de 1024 x 3
    y regresa una fila para metadata.csv.
    """
    points_raw = clean_points(points_raw)

    raw_features = compute_features(points_raw, prefix="raw")

    points_norm, centroid, scale = normalize_pointcloud(points_raw)
    points_final = force_n_points(points_norm, N_POINTS)

    norm_features = compute_features(points_final, prefix="norm")

    out_path = POINTCLOUDS_DIR / f"{sample_id}.npy"
    np.save(out_path, points_final)

    record = {
        "sample_id": sample_id,
        "dataset": dataset,
        "artifact_kind": artifact_kind,
        "label_original": label_original,
        "is_fragment": is_fragment,
        "raw_format": raw_format,
        "source_path": str(source_path),
        "processed_path": str(out_path),
        "n_points_saved": int(points_final.shape[0]),
        "n_dimensions": int(points_final.shape[1]),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "scale_used": scale,
        "normalized": True,
        "standard_shape": "1024x3",
    }

    record.update(raw_features)
    record.update(norm_features)

    if extra_metadata:
        record.update(extra_metadata)

    return record


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    records = []

    print("\nRevisando rutas...")
    print("CeramicNet:", CERAMICNET_DIR)
    print("Existe CeramicNet:", CERAMICNET_DIR.exists())

    print("VoxelFragmentML:", VFM_DIR)
    print("Existe VoxelFragmentML:", VFM_DIR.exists())

    # --------------------------------------------------------
    # 1. CERAMICNET
    # --------------------------------------------------------

    ceramicnet_counter = 0

    if CERAMICNET_DIR.exists():
        print("\nLeyendo CeramicNet...")

        for label_code, artifact_kind in CERAMICNET_LABELS.items():
            class_dir = CERAMICNET_DIR / label_code

            if not class_dir.exists():
                print(f"No encontré carpeta de clase: {class_dir}")
                continue

            txt_files = sorted(class_dir.rglob("*.txt"))

            print(f"Clase {label_code} / {artifact_kind}: {len(txt_files)} archivos")

            for path in tqdm(txt_files, desc=f"CeramicNet {label_code}"):
                try:
                    points = read_ceramicnet_txt(path)

                    sample_id = f"ceramicnet_{ceramicnet_counter:06d}"

                    record = process_and_save(
                        points_raw=points,
                        sample_id=sample_id,
                        dataset="ceramicnet_sue_ware",
                        artifact_kind=artifact_kind,
                        label_original=label_code,
                        is_fragment=False,
                        source_path=path,
                        raw_format="txt_xyz",
                        extra_metadata={
                            "vessel_id": path.stem,
                            "vessel_folder": label_code,
                            "n_fragments": None,
                            "resolution": None,
                            "iteration": None,
                            "fragment_id": None,
                        },
                    )

                    records.append(record)
                    ceramicnet_counter += 1

                except Exception as e:
                    print(f"Error leyendo {path}: {e}")

    else:
        print("No existe la ruta de CeramicNet. Revisa la ruta.")

    # --------------------------------------------------------
    # 2. VOXELFRAGMENTML
    # --------------------------------------------------------

    vfm_counter = 0

    if VFM_DIR.exists():
        print("\nLeyendo VoxelFragmentML...")

        # Usamos SOLO los PLY de nubes de puntos.
        # Excluimos OBJ y MTL porque son mallas/materiales.
        ply_files = sorted(VFM_DIR.rglob("*_1024p.ply"))

        print(f"Archivos *_1024p.ply encontrados: {len(ply_files)}")

        for path in tqdm(ply_files, desc="VoxelFragmentML"):
            try:
                points = read_ply_pointcloud(path)
                info = parse_vfm_filename(path)

                is_fragment_value = info.get("is_fragment")
                is_fragment = True if is_fragment_value == "True" else False

                artifact_kind = info.get("artifact_kind") or "vessel_pointcloud"
                label_original = (
                    "iberian_vessel_fragment"
                    if is_fragment
                    else "iberian_vessel"
                )

                sample_id = f"vfm_{vfm_counter:06d}"

                record = process_and_save(
                    points_raw=points,
                    sample_id=sample_id,
                    dataset="vfm_iberian_vessels",
                    artifact_kind=artifact_kind,
                    label_original=label_original,
                    is_fragment=is_fragment,
                    source_path=path,
                    raw_format="ply_xyz",
                    extra_metadata={
                        "vessel_id": info.get("vessel_id"),
                        "vessel_folder": path.parent.name,
                        "n_fragments": info.get("n_fragments"),
                        "resolution": info.get("resolution"),
                        "iteration": info.get("iteration"),
                        "fragment_id": info.get("fragment_id"),
                    },
                )

                records.append(record)
                vfm_counter += 1

            except Exception as e:
                print(f"Error leyendo {path}: {e}")

    else:
        print("No existe la ruta de VoxelFragmentML. Revisa la ruta.")

    # --------------------------------------------------------
    # 3. GUARDAR METADATA
    # --------------------------------------------------------

    metadata = pd.DataFrame(records)

    metadata_path = OUT_DIR / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    print("\n===================================================")
    print("PROCESO TERMINADO")
    print("===================================================")
    print(f"Total de muestras guardadas: {len(metadata)}")
    print(f"Metadata: {metadata_path}")
    print(f"Nubes procesadas: {POINTCLOUDS_DIR}")

    if len(metadata) > 0:
        print("\nConteo por dataset:")
        print(metadata["dataset"].value_counts())

        print("\nConteo por tipo de artefacto:")
        print(metadata["artifact_kind"].value_counts())

        print("\nConteo por fragmento/no fragmento:")
        print(metadata["is_fragment"].value_counts())

        example_path = metadata.iloc[0]["processed_path"]
        example = np.load(example_path)

        print("\nEjemplo cargado:")
        print("Archivo:", example_path)
        print("Shape:", example.shape)

        summary_path = OUT_DIR / "summary_counts.csv"
        summary = (
            metadata
            .groupby(["dataset", "artifact_kind", "is_fragment"])
            .size()
            .reset_index(name="n_samples")
        )
        summary.to_csv(summary_path, index=False)

        print(f"\nResumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()