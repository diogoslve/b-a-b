# b-a-b
# Power Line Classification System

Classifies power lines from contaminated vegetation class in LAS files using Random Forest and KML tower positions.

## Overview

This system consists of two scripts:

1. **train_powerline_model.py** - Trains a Random Forest classifier using fully classified LAS files
2. **classify_powerlines.py** - Applies the trained model to partially classified LAS files

## Requirements

```bash
pip install laspy scikit-learn joblib fastkml scipy numpy --break-system-packages
```

## Configuration Parameters

Both scripts use the same configuration parameters (defined in training script):

### CORRIDOR_WIDTH
- **Default**: 20.0 meters
- **Description**: Distance from tower-to-tower centerline to include points (±20m corridor)
- **Adjust if**: Your power line corridors are wider/narrower
- **Location in code**: Line 38 in `train_powerline_model.py`

### NEIGHBORHOOD_RADIUS  
- **Default**: 1.0 meters
- **Description**: Radius for computing geometric features (linearity, planarity, etc.)
- **Adjust if**: Power lines are very dense or sparse
- **Location in code**: Line 39 in `train_powerline_model.py`

### MIN_NEIGHBORS
- **Default**: 10 points
- **Description**: Minimum points needed in neighborhood for valid feature computation
- **Adjust if**: Point cloud density is very low/high
- **Location in code**: Line 40 in `train_powerline_model.py`

### LAS Class Codes
- **CLASS_GROUND** = 2
- **CLASS_VEGETATION** = 3
- **CLASS_TOWER** = 14
- **CLASS_POWERLINE** = 15

**To change**: Edit lines 43-46 in `train_powerline_model.py`

## Usage

### Step 1: Train the Model

```bash
python train_powerline_model.py <kml_file> <fully_classified_las> <output_model.pkl>
```

**Example:**
```bash
python train_powerline_model.py towers.kml training_data.las powerline_model.pkl
```

**Inputs:**
- `kml_file`: KML file with tower positions (one vertex per tower)
- `fully_classified_las`: LAS file with correct power line classifications
- `output_model.pkl`: Where to save the trained model

**Expected ETA**: 5-20 minutes depending on file size

**Output:**
- Trained model saved as `.pkl` file
- Training metrics (accuracy, precision, recall)
- Feature importance rankings

### Step 2: Classify New Files

```bash
python classify_powerlines.py <model.pkl> <kml_file> <input_las> <output_las>
```

**Example:**
```bash
python classify_powerlines.py powerline_model.pkl towers.kml partially_classified.las fully_classified.las
```

**Inputs:**
- `model.pkl`: Trained model from Step 1
- `kml_file`: KML with tower positions for this corridor
- `input_las`: Partially classified LAS (power lines labeled as vegetation)
- `output_las`: Where to save fully classified LAS

**Expected ETA**: 5-15 minutes depending on file size

**Output:**
- Fully classified LAS file with power lines correctly labeled
- Classification statistics

## How It Works

### Features Computed

**Geometric features (from local neighborhood):**
1. **Linearity** - How linear the points are (power lines = high)
2. **Planarity** - How planar the distribution is
3. **Verticality** - Vertical orientation (trees often vertical, power lines less so)
4. **Density** - Points per unit volume (cables = sparse)
5. **Height variance** - Z-coordinate variance (cables = low)

**Corridor features:**
6. **Distance to nearest tower** - Power lines connect towers
7. **Distance to corridor centerline** - Power lines follow corridor axis
8. **Height above ground** - Power lines at consistent height

### Training Process

1. Load KML tower positions
2. Load fully classified LAS file
3. Define corridors between consecutive towers
4. Extract vegetation + power line points from corridors
5. Compute 8 features for each point
6. Train Random Forest (100 trees)
7. Evaluate on 20% test split
8. Save model with configuration

### Inference Process

1. Load trained model
2. Load KML and partially classified LAS
3. Define corridors
4. Extract only vegetation points from corridors
5. Compute same 8 features
6. Predict: power line or not
7. Reclassify predicted power lines
8. Save fully classified LAS

## Troubleshooting

### "No vegetation points found in corridors"
- Check that KML coordinates match LAS coordinate system
- Increase `CORRIDOR_WIDTH`
- Verify KML has correct tower positions

### Poor classification accuracy
- Increase `NEIGHBORHOOD_RADIUS` if features seem noisy
- Decrease `CORRIDOR_WIDTH` if too much non-power-line vegetation included
- Train on more diverse data

### Very slow processing
- Decrease `NEIGHBORHOOD_RADIUS` (reduces computation time)
- Process smaller files
- Use fewer points for training

## Expected Performance

- **Typical accuracy**: 90-98% (depends on data quality)
- **Training time**: ~1-2 min/million points
- **Inference time**: ~1-2 min/million points

## File Format Notes

- **KML**: Should contain Point placemarks at tower locations
- **LAS**: Standard LAS 1.2+ format
- **Coordinates**: KML and LAS must use same coordinate system
