import streamlit as st
import requests
from textblob import TextBlob
import pandas as pd

st.title("Machine Learning News Sentiment Analyzer")

# API setup
NEWS_API_KEY = "55c89558304a4484ad2c40b18e1c831d"  
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

query = st.text_input("Enter a topic to search news for", value="Machine Learning")

if st.button("Fetch and Analyze News") and query:
    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 50,
        "apiKey": NEWS_API_KEY
    }

    try:
        response = requests.get(NEWS_API_ENDPOINT, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        st.success(f"Fetched {len(articles)} articles.")

        # Process articles into DataFrame
        df = pd.DataFrame([{
            "source": a["source"]["name"],
            "title": a["title"],
            "description": a["description"],
            "content": a["content"],
            "publishedAt": a["publishedAt"],
            "url": a["url"]
        } for a in articles])

        # Use content or fallback to description
        df["text"] = df["content"].fillna(df["description"])

        # Sentiment Analysis
        def analyze_sentiment(text):
            if text:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                sentiment = (
                    "Positive" if polarity > 0
                    else "Negative" if polarity < 0
                    else "Neutral"
                )
                return pd.Series([polarity, sentiment])
            else:
                return pd.Series([None, None])

        df[["polarity", "sentiment"]] = df["text"].apply(analyze_sentiment)

        # Display results
        st.dataframe(df[["title", "sentiment", "polarity"]])

        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        sentiment_counts = df["sentiment"].value_counts()
        st.bar_chart(sentiment_counts)

    except Exception as e:
        st.error(f"Error fetching or analyzing news: {e}")
