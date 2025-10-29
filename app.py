# app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
from joblib import load
from sentiment import score_texts
from topic_model import run_topics
from shapely.geometry import Point

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(layout="wide", page_title="Urban Development & Public Sentiment – India")

# ----------------------------------------------------------
# POWER BI–STYLE THEME
# ----------------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #F5F6F8;
    color: #000000;
    font-family: "Segoe UI", sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: #2B2B2B;
    border-right: 2px solid #F2C811;
}
h1, h2, h3 {
    color: #F2C811 !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------
st.markdown("""
<h1 style='text-align:center;'> Urban Development & Public Sentiment Dashboard – India</h1>
<hr style="border:1px solid #F2C811;">
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
    sr = pd.read_csv("data/311_data.csv")
    nb = gpd.read_file("data/neighborhoods.geojson")
    social = pd.read_csv("data/social_posts.csv")
    transit = pd.read_csv("data/transit.csv")
    return sr, nb, social, transit

sr, nb, social, transit = load_data()

# ----------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg", width=120)
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section:",
    ["Overview Map", "Sentiment", "Transit", "Topics", "NeedScore & Risk"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("Explore real Indian civic data, social sentiment, and risk prediction.")

# ----------------------------------------------------------
# DATE FILTER
# ----------------------------------------------------------
st.sidebar.markdown("### 📅 Filter by Date")
sr["created_date"] = pd.to_datetime(sr["created_date"], errors="coerce")
min_d, max_d = sr["created_date"].min(), sr["created_date"].max()
rng = st.sidebar.date_input("Select range:", [min_d, max_d])
if len(rng) == 2:
    sr = sr[(sr["created_date"] >= pd.Timestamp(rng[0])) & (sr["created_date"] <= pd.Timestamp(rng[1]))]

# ----------------------------------------------------------
# Download helper
# ----------------------------------------------------------
def download_button(df, label):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=f" {label}", data=csv, file_name=f"{label.replace(' ','_').lower()}.csv", mime="text/csv")

# ----------------------------------------------------------
# 1. OVERVIEW MAP
# ----------------------------------------------------------
if page == "Overview Map":
    st.header(" Urban Complaints Heatmap (India)")
    sr = sr.dropna(subset=["latitude", "longitude"])
    fig = px.density_mapbox(
        sr, lat="latitude", lon="longitude",
        hover_name="complaint_type",
        color_continuous_scale="Turbo", radius=15,
        center=dict(lat=22.9734, lon=78.6569), zoom=4,
        mapbox_style="carto-positron", height=700)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    vc = sr["complaint_type"].value_counts().reset_index()
    vc.columns = ["Complaint Type", "Count"]
    col1.subheader("Top Complaint Categories")
    col1.dataframe(vc)
    download_button(vc, "Complaint Summary")

    col2.subheader("Complaint Status Summary")
    status_df = sr["status"].value_counts().reset_index()
    status_df.columns = ["Status", "Count"]
    col2.bar_chart(status_df.set_index("Status"))
    download_button(status_df, "Status Summary")

# ----------------------------------------------------------
# 2. SENTIMENT PAGE
# ----------------------------------------------------------
elif page == "Sentiment":
    st.header(" Public Sentiment from Social Media")
    scored = score_texts(social, "text")
    st.map(scored.rename(columns={"lat":"latitude","lon":"longitude"})[["latitude","longitude"]])

    col1, col2 = st.columns(2)
    sent_df = scored["sentiment_label"].value_counts().reset_index()
    sent_df.columns = ["Sentiment","Count"]
    col1.subheader("Sentiment Distribution")
    col1.bar_chart(sent_df.set_index("Sentiment"))
    download_button(sent_df, "Sentiment Summary")

    col2.subheader("Example Posts")
    col2.dataframe(scored[["text","sentiment_label"]])
    download_button(scored, "Social Posts Data")

# ----------------------------------------------------------
# 3. TRANSIT PAGE
# ----------------------------------------------------------
elif page == "Transit":
    st.header(" Transit Performance KPIs – Metro & Bus")
    avg_delay = transit.groupby("route_id")["delay_minutes"].mean().reset_index()
    fig = px.bar(avg_delay, x="route_id", y="delay_minutes",
                 color="delay_minutes", color_continuous_scale="Sunset",
                 title="Average Delay (minutes) by Route")
    st.plotly_chart(fig, use_container_width=True)
    download_button(avg_delay, "Transit Delay Report")

    st.subheader("Transit Stops Map")
    st.map(transit.rename(columns={"lat":"latitude","lon":"longitude"})[["latitude","longitude"]])

# ----------------------------------------------------------
# 4. TOPIC MODELING
# ----------------------------------------------------------
elif page == "Topics":
    st.header(" Topic Modeling – Complaint Clusters")
    texts = sr["complaint_type"].astype(str).tolist()
    labels, top_terms = run_topics(texts, n_topics=4)
    out = pd.DataFrame({"Complaint":texts, "Topic":labels})

    st.write("Top Terms per Topic:")
    for k,v in top_terms.items():
        st.markdown(f"**Topic {k}:** {', '.join(v[:8])}")
    st.write("Sample Complaints per Topic:")
    st.dataframe(out.groupby("Topic").head(5))
    download_button(out, "Topic Clusters")

# ----------------------------------------------------------
# 5. RISK PAGE
# ----------------------------------------------------------
elif page == "NeedScore & Risk":
    st.header(" NeedScore & Predictive Risk – Indian Cities")
    points = gpd.GeoDataFrame(sr, geometry=gpd.points_from_xy(sr.longitude, sr.latitude), crs="EPSG:4326")
    joined = gpd.sjoin(points, nb, predicate="within", how="left")
    ag = joined.groupby("neighborhood").agg(req_count=("unique_key","count")).reset_index().fillna(0)
    ag["req_norm"] = (ag["req_count"] - ag["req_count"].min())/(ag["req_count"].max()-ag["req_count"].min()+1e-9)

    st.map(joined.rename(columns={"latitude":"lat","longitude":"lon"})[["lat","lon"]])
    st.subheader("Complaint Volume per City")
    st.dataframe(ag.sort_values("req_count", ascending=False))
    download_button(ag, "Complaint Volume")

    try:
        model = load("models/risk_model.joblib")
        X = ag[["req_count"]].fillna(0)
        preds = model.predict_proba(X)[:,1]
        ag["risk_score"] = preds
        st.subheader("Predicted Risk Scores")
        fig = px.bar(ag, x="neighborhood", y="risk_score",
                     color="risk_score", color_continuous_scale="Reds",
                     title="Predicted Urban Risk Levels")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ag.sort_values("risk_score", ascending=False))
        download_button(ag, "Risk Scores")
    except Exception as e:
        st.warning("⚠️ Run train_model.py to generate models/risk_model.joblib")
        st.info(str(e))