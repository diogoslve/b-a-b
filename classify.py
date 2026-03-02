#!/usr/bin/env python3
"""
Power Line Classifier - Inference Script

Applies trained Random Forest model to classify power lines from vegetation class
in partially classified LAS files.

Usage:
    python classify_powerlines.py <model.pkl> <kml_file> <partially_classified_las> <output_las>
    
Example:
    python classify_powerlines.py powerline_model.pkl towers.kml input.las output_classified.las
"""

import sys
import time
import numpy as np
import laspy
import joblib
from pathlib import Path
try:
    from fastkml import kml
except ImportError:
    print("Installing fastkml...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "fastkml", "--break-system-packages"], check=True)
    from fastkml import kml


# ============================================================================
# KML LOADING
# ============================================================================
def load_tower_positions(kml_file):
    """
    Load tower positions from KML file.
    
    Returns:
        towers: list of (x, y, z) tuples
    """
    print(f"\nLoading KML: {kml_file}")
    start_time = time.time()
    
    with open(kml_file, 'rb') as f:
        doc = f.read()
    
    k = kml.KML()
    k.from_string(doc)
    
    towers = []
    for feature in k.features():
        for placemark in feature.features():
            if hasattr(placemark, 'geometry') and placemark.geometry:
                geom = placemark.geometry
                if geom.geom_type == 'Point':
                    # KML coordinates are (lon, lat, alt)
                    coords = geom.coords[0]
                    towers.append(coords)
    
    towers = np.array(towers)
    print(f"  Loaded {len(towers)} towers in {time.time() - start_time:.2f}s")
    
    return towers


# ============================================================================
# CORRIDOR EXTRACTION
# ============================================================================
def point_to_line_distance(points, line_start, line_end):
    """
    Calculate perpendicular distance from points to line segment.
    """
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len
    
    point_vecs = points - line_start
    projections = np.dot(point_vecs, line_unit)
    projections = np.clip(projections, 0, line_len)
    
    closest = line_start + projections[:, np.newaxis] * line_unit
    distances = np.linalg.norm(points - closest, axis=1)
    
    return distances


def extract_corridor_mask(points, towers, corridor_width):
    """
    Extract mask of points within corridor.
    """
    print(f"\nExtracting corridor points (width: ±{corridor_width}m)...")
    start_time = time.time()
    
    mask = np.zeros(len(points), dtype=bool)
    
    for i in range(len(towers) - 1):
        distances = point_to_line_distance(points, towers[i], towers[i + 1])
        in_corridor = distances <= corridor_width
        mask |= in_corridor
        print(f"  Corridor {i+1}/{len(towers)-1}: {in_corridor.sum():,} points")
    
    print(f"  Total: {mask.sum():,} / {len(points):,} in {time.time()-start_time:.2f}s")
    return mask


# ============================================================================
# FEATURE COMPUTATION
# ============================================================================
def compute_geometric_features(points, radius, min_neighbors):
    """
    Compute geometric features for each point.
    """
    from scipy.spatial import cKDTree
    
    print(f"\nComputing geometric features (radius={radius}m)...")
    start_time = time.time()
    
    tree = cKDTree(points)
    n_points = len(points)
    features = np.zeros((n_points, 5))
    
    for i in range(n_points):
        if i % 100000 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_points - i) / rate if rate > 0 else 0
            print(f"  Progress: {i:,}/{n_points:,} ({100*i/n_points:.1f}%) - "
                  f"ETA: {eta:.0f}s", end='\r')
        
        indices = tree.query_ball_point(points[i], radius)
        
        if len(indices) < min_neighbors:
            continue
        
        neighborhood = points[indices]
        centered = neighborhood - neighborhood.mean(axis=0)
        
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        ev_sum = eigenvalues.sum()
        if ev_sum < 1e-10:
            continue
        
        e1, e2, e3 = eigenvalues / ev_sum
        
        linearity = (e1 - e2) / e1 if e1 > 0 else 0
        planarity = (e2 - e3) / e1 if e1 > 0 else 0
        
        eigenvectors = np.linalg.eigh(cov)[1]
        primary_direction = eigenvectors[:, -1]
        verticality = abs(primary_direction[2])
        
        volume = (4/3) * np.pi * radius**3
        density = len(indices) / volume
        
        height_var = neighborhood[:, 2].var()
        
        features[i] = [linearity, planarity, verticality, density, height_var]
    
    print(f"\n  Feature computation complete in {time.time()-start_time:.2f}s")
    return features


def compute_corridor_features(points, towers):
    """
    Compute corridor-based features.
    """
    print(f"\nComputing corridor features...")
    start_time = time.time()
    
    from scipy.spatial.distance import cdist
    
    n_points = len(points)
    features = np.zeros((n_points, 3))
    
    # Distance to nearest tower
    tower_distances = cdist(points, towers)
    features[:, 0] = tower_distances.min(axis=1)
    
    # Distance to nearest corridor centerline
    min_corridor_dist = np.full(n_points, np.inf)
    for i in range(len(towers) - 1):
        distances = point_to_line_distance(points, towers[i], towers[i+1])
        min_corridor_dist = np.minimum(min_corridor_dist, distances)
    
    features[:, 1] = min_corridor_dist
    
    # Height above ground
    ground_z = np.percentile(points[:, 2], 10)
    features[:, 2] = points[:, 2] - ground_z
    
    print(f"  Corridor features complete in {time.time()-start_time:.2f}s")
    return features


# ============================================================================
# MAIN INFERENCE
# ============================================================================
def main():
    if len(sys.argv) != 5:
        print("Usage: python classify_powerlines.py <model.pkl> <kml_file> <input_las> <output_las>")
        print("\nExample:")
        print("  python classify_powerlines.py model.pkl towers.kml input.las output.las")
        sys.exit(1)
    
    model_file = sys.argv[1]
    kml_file = sys.argv[2]
    input_las = sys.argv[3]
    output_las = sys.argv[4]
    
    total_start = time.time()
    
    # Load model
    print(f"\nLoading model: {model_file}")
    model_data = joblib.load(model_file)
    rf = model_data['model']
    config = model_data['config']
    
    print(f"  Model configuration:")
    print(f"    Corridor width: {config['corridor_width']}m")
    print(f"    Neighborhood radius: {config['neighborhood_radius']}m")
    print(f"    Min neighbors: {config['min_neighbors']}")
    
    CORRIDOR_WIDTH = config['corridor_width']
    NEIGHBORHOOD_RADIUS = config['neighborhood_radius']
    MIN_NEIGHBORS = config['min_neighbors']
    CLASS_VEGETATION = config['class_codes']['vegetation']
    CLASS_POWERLINE = config['class_codes']['powerline']
    
    # Load tower positions
    towers = load_tower_positions(kml_file)
    
    # Load LAS file
    print(f"\nLoading LAS file: {input_las}")
    las_start = time.time()
    las = laspy.read(input_las)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    classes = np.array(las.classification)  # Make a copy
    print(f"  Loaded {len(points):,} points in {time.time()-las_start:.2f}s")
    
    # Class distribution before
    unique, counts = np.unique(classes, return_counts=True)
    print("\n  Input class distribution:")
    for cls, count in zip(unique, counts):
        print(f"    Class {cls}: {count:,} ({100*count/len(points):.1f}%)")
    
    # Extract vegetation points in corridors
    corridor_mask = extract_corridor_mask(points, towers, CORRIDOR_WIDTH)
    vegetation_mask = classes == CLASS_VEGETATION
    target_mask = corridor_mask & vegetation_mask
    
    target_points = points[target_mask]
    print(f"\nTarget points (vegetation in corridors): {len(target_points):,}")
    
    if len(target_points) == 0:
        print("\nWARNING: No vegetation points found in corridors!")
        print("Nothing to classify. Exiting.")
        sys.exit(1)
    
    # Compute features
    print("\n" + "="*70)
    print("COMPUTING FEATURES")
    print("="*70)
    
    geom_features = compute_geometric_features(target_points, 
                                               NEIGHBORHOOD_RADIUS,
                                               MIN_NEIGHBORS)
    corridor_features = compute_corridor_features(target_points, towers)
    
    # Combine features
    all_features = np.hstack([geom_features, corridor_features])
    
    # Handle invalid features
    valid_mask = np.all(np.isfinite(all_features), axis=1)
    print(f"\nValid features: {valid_mask.sum():,} / {len(all_features):,} points")
    
    # Predict
    print("\n" + "="*70)
    print("CLASSIFYING")
    print("="*70)
    
    predict_start = time.time()
    predictions = np.zeros(len(all_features), dtype=int)
    
    if valid_mask.sum() > 0:
        predictions[valid_mask] = rf.predict(all_features[valid_mask])
    
    print(f"  Prediction complete in {time.time()-predict_start:.2f}s")
    
    # Count predictions
    n_powerlines = predictions.sum()
    n_vegetation = len(predictions) - n_powerlines
    print(f"\n  Predictions:")
    print(f"    Powerlines: {n_powerlines:,} ({100*n_powerlines/len(predictions):.1f}%)")
    print(f"    Vegetation: {n_vegetation:,} ({100*n_vegetation/len(predictions):.1f}%)")
    
    # Update classifications
    print("\n" + "="*70)
    print("UPDATING CLASSIFICATIONS")
    print("="*70)
    
    # Get indices of target points in original array
    target_indices = np.where(target_mask)[0]
    
    # Update classes for predicted powerlines
    powerline_predictions = predictions == 1
    powerline_indices = target_indices[powerline_predictions]
    
    print(f"\nReclassifying {len(powerline_indices):,} points from vegetation to powerline...")
    classes[powerline_indices] = CLASS_POWERLINE
    
    # Class distribution after
    unique, counts = np.unique(classes, return_counts=True)
    print("\n  Output class distribution:")
    for cls, count in zip(unique, counts):
        print(f"    Class {cls}: {count:,} ({100*count/len(points):.1f}%)")
    
    # Save output LAS
    print(f"\nSaving output: {output_las}")
    save_start = time.time()
    
    # Create output LAS with updated classifications
    las.classification = classes
    las.write(output_las)
    
    print(f"  Saved in {time.time()-save_start:.2f}s")
    
    total_time = time.time() - total_start
    print(f"\n" + "="*70)
    print(f"INFERENCE COMPLETE - Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print("="*70)
    
    print(f"\nSummary:")
    print(f"  Input: {input_las}")
    print(f"  Output: {output_las}")
    print(f"  Powerlines classified: {len(powerline_indices):,}")


if __name__ == "__main__":
    main()
