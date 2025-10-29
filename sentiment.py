# sentiment.py
from textblob import TextBlob
import pandas as pd
import numpy as np

def score_texts(df, text_col="text"):
    df = df.copy()
    df["sentiment"] = df[text_col].astype(str).apply(lambda t: TextBlob(t).sentiment.polarity)
    df["sentiment_label"] = pd.cut(df["sentiment"], bins=[-1.1,-0.01,0.01,1.1], labels=["Negative","Neutral","Positive"])
    return df

if __name__=="__main__":
    df = pd.read_csv("data/social_posts.csv")
    print(score_texts(df))