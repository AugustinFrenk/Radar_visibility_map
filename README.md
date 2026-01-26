# Radar Visibility Map Project

This project calculates radar visibility maps for aircraft at different flight levels, taking into account terrain occlusion and Earth's curvature.

## Overview

The project computes which areas are visible from a radar station at various aircraft altitudes. It uses ray casting algorithms to determine if terrain blocks the line of sight between the radar and potential target locations.

## Files

- **`Data_loading.py`**: Loads terrain data and sets up radar configuration
  - Loads terrain elevation data from `Nice_Terrain_Data.npz`
  - Defines radar position (latitude, longitude, height)
  - Converts geographic coordinates to local x/y coordinates
  - Configures flight levels to analyze

- **`Radar_Visibility_Model.py`**: Main visibility computation engine
  - Computes visibility maps using ray casting
  - Accounts for Earth's curvature in calculations
  - Exports results to KMZ format for Google Earth visualization
  - Uses Numba for optimized performance

## Usage

1. Ensure you have the required dependencies:
   - `numpy`
   - `numba`
   - `matplotlib`
   - `simplekml`

2. Make sure `Nice_Terrain_Data.npz` is in the project directory

3. Run the main script:
   ```bash
   python Radar_Visibility_Model.py
   ```

4. The script will:
   - Compute visibility maps for each configured flight level
   - Generate overlay images (`vis_overlay_FL*.png`)
   - Export a KMZ file (`radar_visibility.kmz`) for visualization in Google Earth

## Configuration

Radar position and flight levels can be modified in `Data_loading.py`:
- Radar location: `lat_radar`, `lon_radar`, `radar_height`
- Flight levels: `flight_levels` array

## Output

- PNG overlay images for each flight level showing visible areas
- KMZ file containing all layers for Google Earth visualization
