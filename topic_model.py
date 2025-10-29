# topic_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

def run_topics(df_texts, n_topics=5):
    vec = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1,2))
    X = vec.fit_transform(df_texts)
    km = KMeans(n_clusters=n_topics, random_state=42)
    labels = km.fit_predict(X)
    terms = vec.get_feature_names_out()
    top_terms = {}
    for i in range(n_topics):
        center = km.cluster_centers_[i]
        top_idx = center.argsort()[-10:][::-1]
        top_terms[i] = [terms[j] for j in top_idx]
    return labels, top_terms

if __name__=="__main__":
    # small demo using 311 texts
    import pandas as pd
    df = pd.read_csv("data/311_data.csv")
    sample_texts = df["complaint_type"].astype(str).tolist()
    labels, top_terms = run_topics(sample_texts, n_topics=4)
    print(top_terms)