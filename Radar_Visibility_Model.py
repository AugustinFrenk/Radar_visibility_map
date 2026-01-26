import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import Data_loading as ld  
import time
import simplekml

R = 6371000 * 4/3
ter_sub = ld.ter_sub
lat_sub = ld.lat_sub
lon_sub = ld.lon_sub
X = ld.X
Y = ld.Y
z_r = ld.z_r
lat0 = ld.lat0
lon0 = ld.lon0
flight_levels = ld.flight_levels
radar_alt = ld.radar_alt

# Parameters
samples_per_km = 50
min_samples = 20
max_samples = 1000000

# Earth's Curvature
@njit
def earth_curvature(X, Y, ter):
    d = np.sqrt(X**2 + Y**2)
    h = (d**2) / (2 * R)
    return ter - h
@njit
def earth_curvature_point(x, y):
    d = np.sqrt(x**2 + y**2)  
    h = (d**2)/(2*R)  
    return h

# Interpulator
@njit
def fast_terrain_interp_array(lat_grid, lon_grid, terrain_data, lat_array, lon_array):
    n = len(lat_array)
    results = np.empty(n, dtype=np.float64)
    
    d_lat = (lat_grid[-1] - lat_grid[0]) / (len(lat_grid) - 1)
    d_lon = (lon_grid[-1] - lon_grid[0]) / (len(lon_grid) - 1)
    
    lat_min, lat_max = lat_grid[0], lat_grid[-1]
    lon_min, lon_max = lon_grid[0], lon_grid[-1]

    for k in range(n):
        l_val = lat_array[k]
        o_val = lon_array[k]

        if l_val < lat_min or l_val > lat_max or o_val < lon_min or o_val > lon_max:
            results[k] = np.inf
            continue

        i = int((l_val - lat_min) / d_lat)
        j = int((o_val - lon_min) / d_lon)
        i = min(i, len(lat_grid) - 2) 
        j = min(j, len(lon_grid) - 2)

        lat_w = (l_val - lat_grid[i]) / d_lat
        lon_w = (o_val - lon_grid[j]) / d_lon

        v00 = terrain_data[i, j]
        v10 = terrain_data[i + 1, j]
        v01 = terrain_data[i, j + 1]
        v11 = terrain_data[i + 1, j + 1]

        interp_lat0 = v00 * (1 - lat_w) + v10 * lat_w
        interp_lat1 = v01 * (1 - lat_w) + v11 * lat_w
        results[k] = interp_lat0 * (1 - lon_w) + interp_lat1 * lon_w
        
    return results

# Visibility Calc 
@njit(parallel=True)
def visibility_computation(ter_sub, X, Y, flight_alt, z_r, lat0, lon0, lat_sub, lon_sub, samples_per_km, min_samples, max_samples):
    rows, cols = ter_sub.shape
    local_mask = np.zeros((rows, cols), dtype=np.bool_)

    for i in prange(rows):
        for j in range(cols):
            x_t, y_t = X[i, j], Y[i, j]
            z_t = flight_alt - earth_curvature_point(x_t, y_t)

            dist_km = np.sqrt(x_t**2 + y_t**2) / 1000.0
            
            n_samples = int(dist_km * (samples_per_km))
            n_samples = max(min_samples, min(n_samples, max_samples))
            #Ray
            t = np.linspace(0, 1, n_samples + 2)[1:-1]
            x_ray = t * x_t 
            y_ray = t * y_t 
            z_ray = z_r + t * (z_t - z_r)

            lat_ray = lat0 + (y_ray / 6371000) * (180 / np.pi)
            lon_ray = lon0 + (x_ray / (6371000 * np.cos(np.deg2rad(lat0)))) * (180 / np.pi)

            terrain_under_ray = fast_terrain_interp_array(lat_sub, lon_sub, ter_sub, lat_ray, lon_ray)

            if not np.any(terrain_under_ray >= z_ray):
                local_mask[i, j] = True
                
    return local_mask

# Export to GoogleMaps
def export_to_kml_toggled(vis_layers, flight_levels, lat_sub, lon_sub, filename="radar_visibility.kmz"):
    kml = simplekml.Kml()
    
    # All boundarys must be floats
    north, south = float(lat_sub.max()), float(lat_sub.min())
    east, west = float(lon_sub.max()), float(lon_sub.min())

    layer_colors = [
        [1.0, 0.0, 0.0, 0.7],  # Red, slightly more opaque
        [1.0, 0.65, 0.0, 0.7], # Orange
    ]

    for idx, mask in enumerate(vis_layers):
        alt = flight_levels[idx]
        folder = kml.newfolder(name=f"Flight Level {alt}m")
        image_filename = f"vis_overlay_FL{alt}.png"
        
        # Create RGBA array
        rgba_data = np.zeros((mask.shape[0], mask.shape[1], 4))
        color = layer_colors[idx % len(layer_colors)]
        
        # Fill visible areas
        rgba_data[mask, 0] = color[0]
        rgba_data[mask, 1] = color[1]
        rgba_data[mask, 2] = color[2]
        rgba_data[mask, 3] = color[3]
        
        plt.imsave(image_filename, rgba_data, origin='lower')

        ground = folder.newgroundoverlay(name=f"Coverage Map FL{alt}")
        ground.icon.href = image_filename
        
        # Set the Box
        ground.latlonbox.north = north
        ground.latlonbox.south = south
        ground.latlonbox.east = east
        ground.latlonbox.west = west
        
        ground.altitudemode = simplekml.AltitudeMode.clamptoground 
        
        # Radar position
        pnt = folder.newpoint(name="Radar Station", coords=[(lon0, lat0, radar_alt)])
        pnt.altitudemode = simplekml.AltitudeMode.absolute

    kml.savekmz(filename)
    print(f"KMZ exported: {filename}")


if __name__ == "__main__":
    vis_layers = []
    
    for flight_alt in flight_levels:
        start = time.time()
        mask = visibility_computation(
            ter_sub, X, Y, flight_alt, z_r, lat0, lon0, 
            lat_sub, lon_sub, samples_per_km, min_samples, max_samples
        )
        vis_layers.append(mask)
        elapsed = time.time() - start
        print(f"Completed Flight Level: {flight_alt}m ({elapsed:.2f}s)")
    export_to_kml_toggled(vis_layers, flight_levels, lat_sub, lon_sub)
    
