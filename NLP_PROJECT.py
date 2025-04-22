import os
import json
import re
import datetime
from collections import Counter

import nltk
import matplotlib
matplotlib.use("Agg")
from youtube_comment_downloader import YoutubeCommentDownloader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# If first run, uncomment to download stopwords:
# nltk.download("stopwords")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_comments(video_id: str, max_comments: int = 1000):
    """
    Step 1/5: Download up to max_comments from YouTube video and save to raw_comments.json.
    Note: we do NOT pass sort_by_time here, to avoid the TypeError.
    """
    downloader = YoutubeCommentDownloader()
    comments = []
    # Pass the full URL or ID; this library’s get_comments only takes one positional arg.
    for c in downloader.get_comments(video_id):
        comments.append({
            "author": c["author"],
            "text": c["text"],
            "time": c["time"],  # ms timestamp
        })
        if len(comments) >= max_comments:
            break
    with open(f"{DATA_DIR}/raw_comments.json", "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)
    print(f"[1/5] Fetched {len(comments)} comments.")

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower().strip()
    stops = set(nltk.corpus.stopwords.words("english"))
    return " ".join(tok for tok in text.split() if tok not in stops)

def preprocess():
    """Step 2/5: Read raw_comments.json → add clean_text → write clean_comments.json."""
    with open(f"{DATA_DIR}/raw_comments.json", encoding="utf-8") as f:
        data = json.load(f)
    for c in data:
        c["clean_text"] = clean_text(c["text"])
    with open(f"{DATA_DIR}/clean_comments.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("[2/5] Preprocessing done.")

def analyze_sentiment():
    """Step 3/5: Read clean_comments.json → VADER analysis → write sentiment_comments.json."""
    analyzer = SentimentIntensityAnalyzer()
    with open(f"{DATA_DIR}/clean_comments.json", encoding="utf-8") as f:
        data = json.load(f)
    for c in data:
        c["sentiment"] = analyzer.polarity_scores(c["clean_text"])
    with open(f"{DATA_DIR}/sentiment_comments.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("[3/5] Sentiment analysis done.")

def load_data():
    with open(f"{DATA_DIR}/sentiment_comments.json", encoding="utf-8") as f:
        return json.load(f)

def sentiment_distribution(data):
    dist = Counter()
    for c in data:
        comp = c["sentiment"]["compound"]
        if comp >= 0.05:      dist["positive"] += 1
        elif comp <= -0.05:   dist["negative"] += 1
        else:                 dist["neutral"]  += 1
    return dist

def plot_sentiment(dist):
    labels, counts = zip(*dist.items())
    plt.figure()
    plt.bar(labels, counts)
    plt.title("Sentiment Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/sentiment_dist.png")
    plt.close()
    print("[4a/5] Saved sentiment_dist.png")

def length_distribution(data):
    lengths = [len(c["clean_text"].split()) for c in data]
    plt.figure()
    plt.hist(lengths, bins=20)
    plt.title("Comment Length Distribution")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/length_dist.png")
    plt.close()
    print("[4b/5] Saved length_dist.png")

def time_series(data):
    """
    Step 4c/5: Create a time series plot of comment counts per day.
    Handles potential missing or invalid timestamps gracefully.
    """
    valid_dates = []
    for c in data:
        try:
            # Convert milliseconds to seconds, then to datetime.date
            ts = int(c["time"]) / 1000
            date = datetime.datetime.fromtimestamp(ts).date()
            valid_dates.append(date)
        except (KeyError, TypeError, ValueError):
            continue  # Skip any comment without valid timestamp

    if not valid_dates:
        print("⚠️ No valid timestamps found. Skipping time_series plot.")
        return

    # Count comments per date
    date_counts = Counter(valid_dates)
    sorted_dates = sorted(date_counts.items())

    x, y = zip(*sorted_dates)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title("Comments Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Comments")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/time_series.png")
    plt.close()
    print("[4c/5] Saved time_series.png")


def top_keywords(data, top_n=10):
    tokens = []
    for c in data:
        tokens.extend(c["clean_text"].split())
    return Counter(tokens).most_common(top_n)

def main():
    video_id = "fK85SQzm0Z0"
    fetch_comments(video_id)
    preprocess()
    analyze_sentiment()

    data = load_data()
    dist = sentiment_distribution(data)
    print(f"[5/5] Sentiment counts = {dict(dist)}")

    plot_sentiment(dist)
    length_distribution(data)
    time_series(data)

    print("Top 10 keywords:", top_keywords(data))

if __name__ == "__main__":
    main()
