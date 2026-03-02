#!/usr/bin/env python3
"""
Power Line Classifier - Training Script

Trains a Random Forest model to classify power lines from contaminated vegetation class.
Uses fully classified LAS files with KML tower positions.

Usage:
    python train_powerline_model.py <kml_file> <fully_classified_las> <output_model.pkl>
    
Example:
    python train_powerline_model.py towers.kml training_data.las powerline_model.pkl
"""

import sys
import time
import numpy as np
import laspy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path
try:
    from fastkml import kml
except ImportError:
    print("Installing fastkml...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "fastkml", "--break-system-packages"], check=True)
    from fastkml import kml
from xml.etree import ElementTree as ET


# ============================================================================
# CONFIGURATION
# ============================================================================
CORRIDOR_WIDTH = 20.0  # meters (±20m from tower-tower line)
NEIGHBORHOOD_RADIUS = 1.0  # meters for geometric feature computation
MIN_NEIGHBORS = 10  # minimum points in neighborhood for valid features

# LAS class codes
CLASS_GROUND = 2
CLASS_VEGETATION = 3
CLASS_TOWER = 14
CLASS_POWERLINE = 15


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
    print(f"  Tower bounds: X[{towers[:, 0].min():.2f}, {towers[:, 0].max():.2f}], "
          f"Y[{towers[:, 1].min():.2f}, {towers[:, 1].max():.2f}]")
    
    return towers


# ============================================================================
# CORRIDOR EXTRACTION
# ============================================================================
def point_to_line_distance(points, line_start, line_end):
    """
    Calculate perpendicular distance from points to line segment.
    
    Args:
        points: (N, 3) array of points
        line_start: (3,) start of line segment
        line_end: (3,) end of line segment
        
    Returns:
        distances: (N,) perpendicular distances
    """
    # Vector from start to end
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len
    
    # Vector from start to each point
    point_vecs = points - line_start
    
    # Project onto line
    projections = np.dot(point_vecs, line_unit)
    
    # Clamp to line segment
    projections = np.clip(projections, 0, line_len)
    
    # Closest point on line segment
    closest = line_start + projections[:, np.newaxis] * line_unit
    
    # Distance to closest point
    distances = np.linalg.norm(points - closest, axis=1)
    
    return distances


def extract_corridor_points(points, classes, towers, corridor_width):
    """
    Extract points within corridor defined by consecutive towers.
    
    Args:
        points: (N, 3) point cloud
        classes: (N,) classification values
        towers: (M, 3) tower positions
        corridor_width: width in meters
        
    Returns:
        mask: (N,) boolean mask of points in any corridor
    """
    print(f"\nExtracting corridor points (width: ±{corridor_width}m)...")
    start_time = time.time()
    
    mask = np.zeros(len(points), dtype=bool)
    
    # For each consecutive tower pair
    for i in range(len(towers) - 1):
        tower1 = towers[i]
        tower2 = towers[i + 1]
        
        # Calculate distance to this corridor segment
        distances = point_to_line_distance(points, tower1, tower2)
        
        # Points within corridor width
        in_corridor = distances <= corridor_width
        mask |= in_corridor
        
        n_in_corridor = in_corridor.sum()
        print(f"  Corridor {i+1}/{len(towers)-1}: {n_in_corridor:,} points")
    
    total_in_corridor = mask.sum()
    print(f"  Total corridor points: {total_in_corridor:,} / {len(points):,} "
          f"({100*total_in_corridor/len(points):.1f}%) in {time.time()-start_time:.2f}s")
    
    return mask


# ============================================================================
# FEATURE COMPUTATION
# ============================================================================
def compute_geometric_features(points, radius=1.0, min_neighbors=10):
    """
    Compute geometric features for each point using local neighborhood.
    
    Features:
        - Linearity (PCA eigenvalue ratios)
        - Planarity
        - Verticality
        - Local density
        - Height variance
        
    Args:
        points: (N, 3) array
        radius: neighborhood radius
        min_neighbors: minimum points for valid features
        
    Returns:
        features: (N, n_features) array
    """
    from scipy.spatial import cKDTree
    
    print(f"\nComputing geometric features (radius={radius}m)...")
    start_time = time.time()
    
    tree = cKDTree(points)
    n_points = len(points)
    
    # Features: linearity, planarity, verticality, density, height_var
    features = np.zeros((n_points, 5))
    
    for i in range(n_points):
        if i % 100000 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_points - i) / rate if rate > 0 else 0
            print(f"  Progress: {i:,}/{n_points:,} ({100*i/n_points:.1f}%) - "
                  f"ETA: {eta:.0f}s", end='\r')
        
        # Find neighbors
        indices = tree.query_ball_point(points[i], radius)
        
        if len(indices) < min_neighbors:
            continue
        
        neighborhood = points[indices]
        
        # Center the neighborhood
        centered = neighborhood - neighborhood.mean(axis=0)
        
        # PCA via covariance matrix
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]  # descending order
        
        # Avoid division by zero
        ev_sum = eigenvalues.sum()
        if ev_sum < 1e-10:
            continue
        
        # Normalized eigenvalues
        e1, e2, e3 = eigenvalues / ev_sum
        
        # Linearity: how much variance along primary axis
        linearity = (e1 - e2) / e1 if e1 > 0 else 0
        
        # Planarity: how flat the distribution is
        planarity = (e2 - e3) / e1 if e1 > 0 else 0
        
        # Verticality: use eigenvector of largest eigenvalue
        eigenvectors = np.linalg.eigh(cov)[1]
        primary_direction = eigenvectors[:, -1]  # largest eigenvalue
        verticality = abs(primary_direction[2])  # z-component
        
        # Density: points per volume
        volume = (4/3) * np.pi * radius**3
        density = len(indices) / volume
        
        # Height variance in neighborhood
        height_var = neighborhood[:, 2].var()
        
        features[i] = [linearity, planarity, verticality, density, height_var]
    
    print(f"\n  Feature computation complete in {time.time()-start_time:.2f}s")
    
    return features


def compute_corridor_features(points, towers, corridor_width):
    """
    Compute corridor-based features.
    
    Features:
        - Distance to nearest tower
        - Distance to nearest corridor centerline
        - Height above ground (approximated by z-coordinate percentile)
        
    Args:
        points: (N, 3) array
        towers: (M, 3) tower positions
        corridor_width: corridor width
        
    Returns:
        features: (N, 3) array
    """
    print(f"\nComputing corridor features...")
    start_time = time.time()
    
    n_points = len(points)
    features = np.zeros((n_points, 3))
    
    # Distance to nearest tower
    from scipy.spatial.distance import cdist
    tower_distances = cdist(points, towers)
    features[:, 0] = tower_distances.min(axis=1)
    
    # Distance to nearest corridor centerline
    min_corridor_dist = np.full(n_points, np.inf)
    for i in range(len(towers) - 1):
        distances = point_to_line_distance(points, towers[i], towers[i+1])
        min_corridor_dist = np.minimum(min_corridor_dist, distances)
    
    features[:, 1] = min_corridor_dist
    
    # Height above ground (use 10th percentile of z as ground estimate)
    ground_z = np.percentile(points[:, 2], 10)
    features[:, 2] = points[:, 2] - ground_z
    
    print(f"  Corridor features complete in {time.time()-start_time:.2f}s")
    
    return features


# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    if len(sys.argv) != 4:
        print("Usage: python train_powerline_model.py <kml_file> <fully_classified_las> <output_model.pkl>")
        print("\nExample:")
        print("  python train_powerline_model.py towers.kml training.las model.pkl")
        sys.exit(1)
    
    kml_file = sys.argv[1]
    las_file = sys.argv[2]
    output_model = sys.argv[3]
    
    total_start = time.time()
    
    # Load tower positions
    towers = load_tower_positions(kml_file)
    
    # Load LAS file
    print(f"\nLoading LAS file: {las_file}")
    las_start = time.time()
    las = laspy.read(las_file)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    classes = las.classification
    print(f"  Loaded {len(points):,} points in {time.time()-las_start:.2f}s")
    
    # Class distribution
    unique, counts = np.unique(classes, return_counts=True)
    print("\n  Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"    Class {cls}: {count:,} ({100*count/len(points):.1f}%)")
    
    # Extract corridor points
    corridor_mask = extract_corridor_points(points, classes, towers, CORRIDOR_WIDTH)
    corridor_points = points[corridor_mask]
    corridor_classes = classes[corridor_mask]
    
    # Filter to training classes (exclude ground and towers)
    train_mask = (corridor_classes == CLASS_VEGETATION) | \
                 (corridor_classes == CLASS_POWERLINE)
    
    train_points = corridor_points[train_mask]
    train_classes = corridor_classes[train_mask]
    
    print(f"\nTraining data:")
    print(f"  Total points: {len(train_points):,}")
    unique, counts = np.unique(train_classes, return_counts=True)
    for cls, count in zip(unique, counts):
        class_name = "POWERLINE" if cls == CLASS_POWERLINE else "VEGETATION/OTHER"
        print(f"    {class_name}: {count:,} ({100*count/len(train_points):.1f}%)")
    
    # Create binary labels (1 = powerline, 0 = not powerline)
    labels = (train_classes == CLASS_POWERLINE).astype(int)
    
    # Compute features
    print("\n" + "="*70)
    print("COMPUTING FEATURES")
    print("="*70)
    
    geom_features = compute_geometric_features(train_points, 
                                               radius=NEIGHBORHOOD_RADIUS,
                                               min_neighbors=MIN_NEIGHBORS)
    corridor_features = compute_corridor_features(train_points, towers, CORRIDOR_WIDTH)
    
    # Combine all features
    all_features = np.hstack([geom_features, corridor_features])
    
    # Remove any rows with NaN or inf
    valid_mask = np.all(np.isfinite(all_features), axis=1)
    all_features = all_features[valid_mask]
    labels = labels[valid_mask]
    
    print(f"\nFinal training set: {len(all_features):,} points with {all_features.shape[1]} features")
    print(f"  Powerlines: {labels.sum():,} ({100*labels.sum()/len(labels):.1f}%)")
    print(f"  Other: {(~labels.astype(bool)).sum():,} ({100*(~labels.astype(bool)).sum()/len(labels):.1f}%)")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train Random Forest
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST")
    print("="*70)
    
    train_start = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    rf.fit(X_train, y_train)
    print(f"\nTraining complete in {time.time()-train_start:.2f}s")
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    y_pred = rf.predict(X_test)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Not Powerline', 'Powerline']))
    
    # Feature importance
    print("\nFeature Importance:")
    feature_names = ['linearity', 'planarity', 'verticality', 'density', 'height_var',
                    'dist_to_tower', 'dist_to_corridor', 'height_above_ground']
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Save model with configuration
    print(f"\nSaving model to: {output_model}")
    model_data = {
        'model': rf,
        'config': {
            'corridor_width': CORRIDOR_WIDTH,
            'neighborhood_radius': NEIGHBORHOOD_RADIUS,
            'min_neighbors': MIN_NEIGHBORS,
            'feature_names': feature_names,
            'class_codes': {
                'ground': CLASS_GROUND,
                'vegetation': CLASS_VEGETATION,
                'tower': CLASS_TOWER,
                'powerline': CLASS_POWERLINE
            }
        }
    }
    joblib.dump(model_data, output_model)
    
    total_time = time.time() - total_start
    print(f"\n" + "="*70)
    print(f"TRAINING COMPLETE - Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print("="*70)


if __name__ == "__main__":
    main()
