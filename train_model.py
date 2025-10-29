# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import geopandas as gpd
from shapely.geometry import Point

def agg_by_neighborhood():
    sr = pd.read_csv("data/311_data.csv", parse_dates=["created_date","closed_date"])
    nb = gpd.read_file("data/neighborhoods.geojson")
    points = gpd.GeoDataFrame(sr, geometry=gpd.points_from_xy(sr.longitude, sr.latitude), crs="EPSG:4326")
    joined = gpd.sjoin(points, nb, predicate="within", how="left")
    ag = joined.groupby("neighborhood").agg(
        req_count=("unique_key","count"),
        avg_res_time=("closed_date", lambda s: (pd.to_datetime(s).dropna() - pd.to_datetime(joined.loc[s.index,"created_date"].iloc[0])).mean().total_seconds()/3600 if True else 0)
    ).reset_index().fillna(0)
    # synthetic target: neighborhood needs attention if req_count > median
    ag["target"] = (ag["req_count"] > ag["req_count"].median()).astype(int)
    return ag

if __name__=="__main__":
    ag = agg_by_neighborhood()
    if ag.shape[0] < 2:
        print("Not enough neighborhoods in demo")
    else:
        X = ag[["req_count","avg_res_time"]].fillna(0)
        y = ag["target"]
        X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state=42)
        m = RandomForestClassifier(n_estimators=100, random_state=42)
        m.fit(X_train,y_train)
        print("Train score:", m.score(X_train,y_train), "Val score:", m.score(X_val,y_val))
        dump(m, "models/risk_model.joblib")
        print("Model saved to models/risk_model.joblib")