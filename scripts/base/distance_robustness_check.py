#!/usr/bin/env python3
"""
Distance metric robustness check for the base paper.

Computes raw Euclidean, Manhattan, and minimum network (shortest-path driving)
distances for observed trips, where each trip is defined as the distance from
an agent's geocoded home address to the centroid of the observed incident block
group. 

Usage:
    python scripts/base/distance_robustness_check.py

Output:
    data/tables/base/distance_robustness_correlations_by_role_crime_type.csv
    data/tables/base/distance_robustness_summary_by_role_crime_type.txt
    data/tables/base/trip_distances.npz
"""

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCKS_PATH = REPO_ROOT / "data" / "features" / "blocks.jsonl"
AGENT_PATHS = {
    "victims": REPO_ROOT / "data" / "features" / "victims.jsonl",
    "offenders": REPO_ROOT / "data" / "features" / "offenders.jsonl",
}
OUTPUT_DIR = REPO_ROOT / "data" / "tables" / "base"
TRIP_OUTPUT_PATH = OUTPUT_DIR / "trip_distances.npz"
ROLE_CRIME_CORR_PATH = (
    OUTPUT_DIR / "distance_robustness_correlations_by_role_crime_type.csv"
)
ROLE_CRIME_SUMMARY_PATH = (
    OUTPUT_DIR / "distance_robustness_summary_by_role_crime_type.txt"
)
SOURCE_CRS = "EPSG:6583"
TARGET_CRS = "EPSG:4326"
BBOX_BUFFER_DEG = 0.02


def load_block_centroids(
    blocks_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Load block-group centroids from JSONL."""
    block_ids, coords = [], []
    with open(blocks_path) as f:
        for line in f:
            rec = json.loads(line)
            block_ids.append(rec["block_id"])
            coords.append(rec["home_coord"])
    block_ids_arr = np.array(block_ids, dtype=np.int32)
    coords_arr = np.array(coords, dtype=np.float64)
    block_lookup = {
        int(block_id): coords_arr[i].copy() for i, block_id in enumerate(block_ids_arr)
    }
    return block_ids_arr, coords_arr, block_lookup


def load_observed_trips(
    agent_paths: dict[str, Path],
    block_lookup: dict[int, np.ndarray],
) -> dict[str, object]:
    """Load observed trips using home coordinates and incident block centroids."""
    roles, crime_types, agent_ids, incident_block_ids = [], [], [], []
    home_coords, incident_centroids = [], []
    role_counts: dict[str, dict[str, int]] = {}

    for role, path in agent_paths.items():
        total = kept = skipped_missing_home = skipped_missing_block = 0
        with open(path) as f:
            for line in f:
                total += 1
                rec = json.loads(line)

                home_coord = np.asarray(rec.get("home_coord"), dtype=np.float64)
                incident_block_id = rec.get("incident_block_id")

                if (
                    home_coord.shape != (2,)
                    or not np.all(np.isfinite(home_coord))
                    or incident_block_id is None
                ):
                    skipped_missing_home += 1
                    continue

                # Agent files also contain `incident_block_coord`, which is the
                # geocoded incident point. For this robustness check we instead
                # use the observed incident block-group centroid, matched by the
                # specific agent's `incident_block_id`.
                incident_centroid = block_lookup.get(int(incident_block_id))
                if incident_centroid is None or not np.all(np.isfinite(incident_centroid)):
                    skipped_missing_block += 1
                    continue

                roles.append(role)
                crime_types.append(str(rec.get("crime_type") or "unknown"))
                agent_ids.append(int(rec["agent_id"]))
                incident_block_ids.append(int(incident_block_id))
                home_coords.append(home_coord)
                incident_centroids.append(incident_centroid)
                kept += 1

        role_counts[role] = {
            "total": total,
            "kept": kept,
            "skipped_missing_home": skipped_missing_home,
            "skipped_missing_block": skipped_missing_block,
        }

    if not home_coords:
        raise ValueError("No valid observed trips found in agent files.")

    return {
        "roles": np.array(roles),
        "crime_types": np.array(crime_types),
        "agent_ids": np.array(agent_ids, dtype=np.int32),
        "incident_block_ids": np.array(incident_block_ids, dtype=np.int32),
        "home_coords": np.array(home_coords, dtype=np.float64),
        "incident_centroids": np.array(incident_centroids, dtype=np.float64),
        "role_counts": role_counts,
    }


def coords_to_lonlat(coords_proj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform projected coordinates (EPSG:6583) to lon/lat (EPSG:4326)."""
    transformer = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)
    lons, lats = transformer.transform(coords_proj[:, 0], coords_proj[:, 1])
    return np.array(lons), np.array(lats)


def compute_euclidean_distances(
    home_coords: np.ndarray,
    incident_centroids: np.ndarray,
) -> np.ndarray:
    """Euclidean distance (meters) from home to incident block centroid."""
    delta = home_coords - incident_centroids
    return np.sqrt(np.sum(delta * delta, axis=1))


def compute_manhattan_distances(
    home_coords: np.ndarray,
    incident_centroids: np.ndarray,
) -> np.ndarray:
    """Manhattan distance (meters) from home to incident block centroid."""
    return np.sum(np.abs(home_coords - incident_centroids), axis=1)


def download_network(lons: np.ndarray, lats: np.ndarray) -> nx.MultiDiGraph:
    """Download the OSM driving network covering all trip endpoints."""
    bbox = (
        float(lons.min()) - BBOX_BUFFER_DEG,
        float(lats.min()) - BBOX_BUFFER_DEG,
        float(lons.max()) + BBOX_BUFFER_DEG,
        float(lats.max()) + BBOX_BUFFER_DEG,
    )
    print(f"  bbox (W,S,E,N): {bbox}")
    G = ox.graph_from_bbox(bbox=bbox, network_type="drive")
    G = ox.distance.add_edge_lengths(G)
    print(f"  nodes: {G.number_of_nodes():,}, edges: {G.number_of_edges():,}")
    return G


def compute_network_trip_distances(
    G: nx.MultiDiGraph,
    home_nodes: np.ndarray,
    dest_nodes: np.ndarray,
) -> np.ndarray:
    """Compute minimum network distances (meters) for observed trips.

    Because OSM driving networks are directional, we take the shorter of the
    two directed routes between the snapped home node and destination node.
    This matches the reviewer's request for minimum network distance.
    """
    unique_home_nodes = np.unique(home_nodes)
    unique_dest_nodes = np.unique(dest_nodes)
    print(f"  {len(unique_home_nodes):,} unique home nodes")
    print(f"  {len(unique_dest_nodes):,} unique destination nodes")

    trip_groups: dict[int, list[int]] = {}
    for trip_idx, dest_node in enumerate(dest_nodes):
        trip_groups.setdefault(int(dest_node), []).append(trip_idx)

    distances = np.full(len(home_nodes), np.inf, dtype=np.float64)
    G_rev = G.reverse(copy=False)
    t0 = time.time()
    for i, dest_node in enumerate(unique_dest_nodes):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(unique_dest_nodes) - i - 1) if i > 0 else 0
            print(
                f"  Dijkstra {i+1}/{len(unique_dest_nodes)}  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
            )

        idx = np.array(trip_groups[int(dest_node)], dtype=np.int64)
        trip_home_nodes = home_nodes[idx]

        to_dest = nx.single_source_dijkstra_path_length(
            G_rev,
            int(dest_node),
            weight="length",
        )
        from_dest = nx.single_source_dijkstra_path_length(
            G,
            int(dest_node),
            weight="length",
        )

        dist_to = np.fromiter(
            (to_dest.get(int(node), np.inf) for node in trip_home_nodes),
            dtype=np.float64,
            count=len(idx),
        )
        dist_from = np.fromiter(
            (from_dest.get(int(node), np.inf) for node in trip_home_nodes),
            dtype=np.float64,
            count=len(idx),
        )
        distances[idx] = np.minimum(dist_to, dist_from)

    unreachable = int(np.sum(~np.isfinite(distances)))
    if unreachable > 0:
        print(f"  WARNING: {unreachable:,} trips are unreachable on the road network")
    return distances


def safe_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """Compute Pearson and Spearman correlations with small-sample guards."""
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan, np.nan

    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan, np.nan, np.nan, np.nan

    r_p = float(np.corrcoef(a, b)[0, 1])
    if len(a) > 2 and abs(r_p) < 1.0:
        t_stat = r_p * np.sqrt((len(a) - 2) / max(1e-12, 1.0 - r_p**2))
        p_p = float(2 * stats.t.sf(abs(t_stat), df=len(a) - 2))
    else:
        p_p = 0.0

    r_s, p_s = stats.spearmanr(a, b)
    return r_p, p_p, float(r_s), float(p_s)


def correlation_table(
    euc: np.ndarray,
    man: np.ndarray,
    net: np.ndarray,
    role: str = "all",
    crime_type: str = "all",
) -> pd.DataFrame:
    """Pairwise Pearson & Spearman correlations between distance metrics.

    Raw distance correlations are the primary reviewer-facing comparison.
    Logged correlations are included as a supplemental check because the model
    uses log-Euclidean distance as the distance covariate.
    """
    valid = np.isfinite(euc) & np.isfinite(man) & np.isfinite(net)
    euc_v, man_v, net_v = euc[valid], man[valid], net[valid]

    # Exclude zero-distance trips for log correlations.
    pos = (euc_v > 0) & (man_v > 0) & (net_v > 0)
    log_euc = np.log(euc_v[pos])
    log_man = np.log(man_v[pos])
    log_net = np.log(net_v[pos])

    pairs = [
        ("raw", "Euclidean", "Manhattan", euc_v, man_v),
        ("raw", "Euclidean", "Network", euc_v, net_v),
        ("raw", "Manhattan", "Network", man_v, net_v),
        ("log", "Euclidean", "Manhattan", log_euc, log_man),
        ("log", "Euclidean", "Network", log_euc, log_net),
        ("log", "Manhattan", "Network", log_man, log_net),
    ]

    rows = []
    for space, name_a, name_b, a, b in pairs:
        r_p, p_p, r_s, p_s = safe_corr(a, b)
        rows.append({
            "Role": role,
            "Crime Type": crime_type,
            "Space": space,
            "Metric A": name_a,
            "Metric B": name_b,
            "Pearson r": round(r_p, 4),
            "Pearson p": p_p,
            "Spearman rho": round(r_s, 4),
            "Spearman p": p_s,
            "N trips": len(a),
        })
    return pd.DataFrame(rows)


def build_grouped_tables(
    trips: dict[str, object],
    euc: np.ndarray,
    man: np.ndarray,
    net: np.ndarray,
) -> pd.DataFrame:
    """Build the by-role x crime-type correlation table."""

    roles = trips["roles"]
    crime_types = trips["crime_types"]

    by_role_crime_frames = []
    for role in trips["role_counts"]:
        role_mask = roles == role
        role_crime_types = sorted(set(crime_types[role_mask]))
        for crime_type in role_crime_types:
            mask = role_mask & (crime_types == crime_type)
            by_role_crime_frames.append(
                correlation_table(
                    euc[mask],
                    man[mask],
                    net[mask],
                    role=role,
                    crime_type=str(crime_type),
                )
            )

    by_role_crime_df = pd.concat(by_role_crime_frames, ignore_index=True)
    return by_role_crime_df


def get_corr_row(
    df: pd.DataFrame,
    role: str,
    crime_type: str,
    space: str,
    metric_a: str,
    metric_b: str,
) -> pd.Series:
    """Return one correlation row from a grouped table."""
    subset = df[
        (df["Role"] == role)
        & (df["Crime Type"] == crime_type)
        & (df["Space"] == space)
        & (df["Metric A"] == metric_a)
        & (df["Metric B"] == metric_b)
    ]
    if subset.empty:
        raise ValueError(
            f"Missing correlation row for role={role}, crime_type={crime_type}, "
            f"space={space}, metrics={metric_a}/{metric_b}."
        )
    return subset.iloc[0]


def append_correlation_block(
    lines: list[str],
    df: pd.DataFrame,
    indent: str,
) -> None:
    """Append raw/log correlation rows for one grouped subset."""
    for space in ("raw", "log"):
        subset = df[df["Space"] == space]
        label = "Raw" if space == "raw" else "Log"
        lines.append(f"{indent}{label}:")
        for _, row in subset.iterrows():
            lines.append(
                f"{indent}  {row['Metric A']} vs {row['Metric B']}: "
                f"Pearson r={row['Pearson r']:.4f}   "
                f"Spearman ρ={row['Spearman rho']:.4f}"
            )


def summary_by_role_crime_text(
    metadata: dict[str, object],
    euc: np.ndarray,
    man: np.ndarray,
    net: np.ndarray,
    by_role_crime_df: pd.DataFrame,
) -> str:
    """Generate a human-readable summary grouped by role and crime type."""
    valid = np.isfinite(euc) & np.isfinite(man) & np.isfinite(net)
    euc_v = euc[valid]
    man_v = man[valid]
    net_v = net[valid]
    ratio_mask = euc_v > 0
    ratio = net_v[ratio_mask] / euc_v[ratio_mask]
    role_counts = metadata["role_counts"]

    lines = [
        "Distance Metric Robustness Check — By Role x Crime Type",
        "=" * 58,
        "Distance definition: agent geocoded home -> incident block-group centroid",
        f"Observed trips: {int(metadata['n_trips']):,}",
        f"  Victims: {role_counts['victims']['kept']:,}",
        f"  Offenders: {role_counts['offenders']['kept']:,}",
        f"Unique home coordinates: {metadata['n_unique_home_coords']:,}",
        f"Unique incident block groups: {metadata['n_unique_incident_blocks']:,}",
        f"Reachable trips (network): {int(np.sum(valid)):,}",
        f"Unreachable trips: {int(np.sum(~valid)):,}",
        "",
        "Distance descriptives (meters, reachable trips only):",
        f"  Euclidean:  mean={np.mean(euc_v):,.0f}  median={np.median(euc_v):,.0f}  SD={np.std(euc_v):,.0f}",
        f"  Manhattan:  mean={np.mean(man_v):,.0f}  median={np.median(man_v):,.0f}  SD={np.std(man_v):,.0f}",
        f"  Network:    mean={np.mean(net_v):,.0f}  median={np.median(net_v):,.0f}  SD={np.std(net_v):,.0f}",
        "",
        "Network / Euclidean ratio (detour factor):",
        f"  mean={np.mean(ratio):.3f}  median={np.median(ratio):.3f}  SD={np.std(ratio):.3f}",
        "",
    ]

    for role in metadata["roles_order"]:
        lines.append(role)
        role_df = by_role_crime_df[by_role_crime_df["Role"] == role]
        crime_types = sorted(set(role_df["Crime Type"]))
        for crime_type in crime_types:
            raw_row = get_corr_row(
                by_role_crime_df,
                role,
                crime_type,
                "raw",
                "Euclidean",
                "Network",
            )
            log_row = get_corr_row(
                by_role_crime_df,
                role,
                crime_type,
                "log",
                "Euclidean",
                "Network",
            )
            lines.append(
                f"  {crime_type} "
                f"(raw N={int(raw_row['N trips']):,}; log N={int(log_row['N trips']):,})"
            )
            crime_df = role_df[role_df["Crime Type"] == crime_type]
            append_correlation_block(lines, crime_df, indent="    ")
        lines.append("")

    lines.append(
        "Note: correlations are reported on both raw and log-distance scales "
        "because the fitted model uses log(Euclidean distance) as the distance "
        "covariate (l2_log)."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading block-group centroids...")
    block_ids, _, block_lookup = load_block_centroids(BLOCKS_PATH)
    print(f"  {len(block_ids):,} block groups")

    print("Loading observed trips...")
    trips = load_observed_trips(AGENT_PATHS, block_lookup)
    home_coords = trips["home_coords"]
    incident_centroids = trips["incident_centroids"]
    print(f"  {len(home_coords):,} observed trips")
    for role, counts in trips["role_counts"].items():
        print(f"  {role}: kept {counts['kept']:,} of {counts['total']:,}")

    print("Computing Euclidean distances...")
    euc = compute_euclidean_distances(home_coords, incident_centroids)

    print("Computing Manhattan distances...")
    man = compute_manhattan_distances(home_coords, incident_centroids)

    unique_home_coords, home_inverse = np.unique(home_coords, axis=0, return_inverse=True)
    unique_dest_coords, dest_inverse = np.unique(
        incident_centroids,
        axis=0,
        return_inverse=True,
    )

    print("Transforming unique trip endpoints to lon/lat...")
    home_lons, home_lats = coords_to_lonlat(unique_home_coords)
    dest_lons, dest_lats = coords_to_lonlat(unique_dest_coords)

    print("Downloading OSM driving network...")
    G = download_network(
        np.concatenate([home_lons, dest_lons]),
        np.concatenate([home_lats, dest_lats]),
    )

    print("Snapping homes and incident centroids to network...")
    unique_home_nodes = np.asarray(ox.nearest_nodes(G, X=home_lons, Y=home_lats))
    unique_dest_nodes = np.asarray(ox.nearest_nodes(G, X=dest_lons, Y=dest_lats))
    home_nodes = unique_home_nodes[home_inverse]
    dest_nodes = unique_dest_nodes[dest_inverse]

    print("Computing minimum network distances...")
    net = compute_network_trip_distances(G, home_nodes, dest_nodes)

    np.savez_compressed(
        TRIP_OUTPUT_PATH,
        role=trips["roles"],
        crime_type=trips["crime_types"],
        agent_id=trips["agent_ids"],
        incident_block_id=trips["incident_block_ids"],
        home_coord=home_coords,
        incident_block_centroid=incident_centroids,
        euclidean=euc,
        manhattan=man,
        network=net,
        home_node=home_nodes,
        incident_block_node=dest_nodes,
    )
    print(f"Trip distance arrays saved → {TRIP_OUTPUT_PATH}")

    print("\nComputing correlations...")
    by_role_crime_df = build_grouped_tables(trips, euc, man, net)
    by_role_crime_df.to_csv(ROLE_CRIME_CORR_PATH, index=False)
    print(f"By-role x crime-type correlation table saved → {ROLE_CRIME_CORR_PATH}")

    metadata = {
        "n_trips": len(home_coords),
        "n_unique_home_coords": len(unique_home_coords),
        "n_unique_incident_blocks": len(np.unique(trips["incident_block_ids"])),
        "role_counts": trips["role_counts"],
        "roles_order": list(trips["role_counts"].keys()),
    }
    role_crime_summary = summary_by_role_crime_text(
        metadata,
        euc,
        man,
        net,
        by_role_crime_df,
    )
    ROLE_CRIME_SUMMARY_PATH.write_text(role_crime_summary)
    print(f"\n{role_crime_summary}")
    print(f"By-role x crime-type summary saved → {ROLE_CRIME_SUMMARY_PATH}")


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
