# data_prep.py
import os, json
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, mapping
import geopandas as gpd

os.makedirs("data", exist_ok=True)

# Synthetic 311 data (NYC-like lat/lon ranges)
np.random.seed(42)
n = 1000
lats = np.random.uniform(40.55, 40.85, n)
lons = np.random.uniform(-74.05, -73.75, n)
types = np.random.choice(["Pothole","Street Light","Garbage","Noise","Water Leak"], n, p=[0.25,0.2,0.3,0.15,0.1])
created = pd.date_range("2025-10-01", periods=n, freq="H").astype(str)
closed = [pd.NaT if np.random.rand()<0.2 else (pd.Timestamp(c)+pd.Timedelta(hours=int(np.random.exponential(48)))).isoformat() for c in created]

df = pd.DataFrame({
    "unique_key": range(10000,10000+n),
    "created_date": created,
    "closed_date": closed,
    "complaint_type": types,
    "status": np.where(np.random.rand(n)<0.7,"Closed","Open"),
    "latitude": lats,
    "longitude": lons,
    "borough": np.random.choice(["MANHATTAN","BROOKLYN","QUEENS"], n)
})
df.to_csv("data/311_data.csv", index=False)
print("311_data.csv written:", df.shape)

# Social posts
posts = [
    ("Roads are terrible near downtown", 40.72, -74.00),
    ("Bus service is improving", 40.75, -73.99),
    ("Garbage not collected", 40.68, -73.94),
    ("Parks are clean and safe", 40.73, -73.79),
]
sdf = pd.DataFrame(posts, columns=["text","lat","lon"])
sdf["post_id"] = range(1, len(sdf)+1)
sdf.to_csv("data/social_posts.csv", index=False)
print("social_posts.csv written")

# Transit sample
trans = pd.DataFrame({
    "route_id": ["R1","R1","R2","R2","R3"],
    "stop_id": [101,102,201,202,301],
    "event_time": pd.date_range("2025-10-10", periods=5, freq="H").astype(str),
    "delay_minutes": [5,10,0,3,20],
    "lat": [40.72,40.73,40.68,40.70,40.74],
    "lon": [-74.00,-73.98,-73.94,-73.96,-73.95]
})
trans.to_csv("data/transit.csv", index=False)
print("transit.csv written")

# Neighborhoods geojson (4 rough polygons)
polys = [
    {"name":"Downtown", "poly": Polygon([(-74.01,40.70),(-73.99,40.70),(-73.99,40.74),(-74.01,40.74)])},
    {"name":"Midtown", "poly": Polygon([(-73.99,40.74),(-73.95,40.74),(-73.95,40.78),(-73.99,40.78)])},
    {"name":"Brookside", "poly": Polygon([(-73.98,40.66),(-73.92,40.66),(-73.92,40.71),(-73.98,40.71)])},
    {"name":"QueensEdge","poly": Polygon([(-73.89,40.72),(-73.83,40.72),(-73.83,40.76),(-73.89,40.76)])},
]
gdf = gpd.GeoDataFrame([{"neighborhood":p["name"], "geometry":p["poly"]} for p in polys], crs="EPSG:4326")
gdf.to_file("data/neighborhoods.geojson", driver="GeoJSON")
print("neighborhoods.geojson written")