import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

data = np.load('Nice_Terrain_Data.npz')
ter = data['ter']       
latitudes = data['lat']
longitudes = data['lon']
print(len(latitudes), len(longitudes))
ter_sub = ter
lat_sub = latitudes
lon_sub = longitudes
lon_mg, lat_mg = np.meshgrid(lon_sub, lat_sub)
R = 6371000  
@njit
def latlon_to_xy(lat, lon, lat0, lon0):
    x = np.deg2rad(lon - lon0) * R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * R
    return x, y

# Radar Placement Airport Radar(43.66472, 7.20450) 43.6475491, 7.1033260 (ZAK 43.718099, 7.028494)
lat_radar, lon_radar = 43.6475491, 7.1033260
radar_height = 30
lat0, lon0 = lat_radar, lon_radar

# Index of coordinates
idx_lat_r = np.argmin(np.abs(lat_sub - lat_radar))
idx_lon_r = np.argmin(np.abs(lon_sub - lon_radar))
radar_alt = (ter_sub[idx_lat_r, idx_lon_r] + radar_height) 

# Flight Levels
flight_levels = [152, 305, 610, 1524, 3048, 6096, 9144, 12192]
colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'purple']

# Coordinates Relative to Radar at 0,0
X, Y = latlon_to_xy(lat_mg, lon_mg, lat0, lon0)
z_r = radar_alt