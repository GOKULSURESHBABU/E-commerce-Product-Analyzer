"""
Feedback Themes & Insights Analyzer
Extracts and clusters common topics from positive and negative reviews
to provide actionable business recommendations.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np


def extract_feedback_themes(df: pd.DataFrame, sentiment_col: str = "predicted_sentiment", text_col: str = "clean_review", n_themes: int = 4) -> dict:
    """
    Extract key themes from positive and negative reviews using TF-IDF + KMeans clustering.
    
    Args:
        df: DataFrame with sentiment and review text
        sentiment_col: column name for sentiment predictions
        text_col: column name for cleaned review text
        n_themes: number of themes to extract per sentiment
    
    Returns:
        Dictionary with themes for positive, negative, and neutral reviews
    """
    themes = {"positive": [], "negative": [], "neutral": []}
    
    for sentiment in ["Positive", "Negative", "Neutral"]:
        subset = df[df[sentiment_col] == sentiment][text_col].values
        if len(subset) < 5:
            themes[sentiment.lower()] = [{"theme": f"Not enough {sentiment.lower()} reviews", "keywords": [], "count": len(subset)}]
            continue
        
        # Vectorize reviews
        vectorizer = TfidfVectorizer(max_features=50, stop_words="english", min_df=2, max_df=0.8)
        try:
            tfidf_matrix = vectorizer.fit_transform(subset)
        except ValueError:
            themes[sentiment.lower()] = [{"theme": f"Insufficient vocabulary in {sentiment.lower()} reviews", "keywords": [], "count": len(subset)}]
            continue
        
        # Cluster into themes
        n_clusters = min(n_themes, len(subset) // 3, tfidf_matrix.shape[1])
        if n_clusters < 2:
            n_clusters = 1
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(tfidf_matrix)
        
        # Extract top words per cluster
        feature_names = vectorizer.get_feature_names_out()
        for cluster_id in range(n_clusters):
            # Get centroid's top features
            top_indices = kmeans.cluster_centers_[cluster_id].argsort()[-5:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            # Count reviews in cluster
            cluster_count = sum(kmeans.labels_ == cluster_id)
            theme_name = " + ".join(keywords[:3]).title()
            
            themes[sentiment.lower()].append({
                "theme": theme_name,
                "keywords": keywords,
                "count": cluster_count,
                "percentage": round(100 * cluster_count / len(subset), 1)
            })
    
    return themes


def generate_recommendations(df: pd.DataFrame, themes: dict, sentiment_col: str = "predicted_sentiment") -> list:
    """
    Generate actionable business recommendations based on feedback themes.
    
    Args:
        df: DataFrame with sentiment data
        themes: themes dictionary from extract_feedback_themes
        sentiment_col: column name for sentiment
    
    Returns:
        List of actionable recommendations
    """
    recommendations = []
    
    # Positive themes: reinforce strengths
    for theme in themes.get("positive", []):
        if "count" in theme and theme["count"] > 0:
            recommendations.append({
                "type": "Reinforce",
                "action": f"Strengthen: {theme['theme']}",
                "details": f"Customers praise this aspect ({theme['count']} reviews). Highlight in marketing.",
                "priority": "High" if theme.get("percentage", 0) > 15 else "Medium"
            })
    
    # Negative themes: address weaknesses
    for theme in themes.get("negative", []):
        if "count" in theme and theme["count"] > 0:
            recommendations.append({
                "type": "Address",
                "action": f"Improve: {theme['theme']}",
                "details": f"Customers complain about this ({theme['count']} reviews). Create action plan.",
                "priority": "Critical" if theme.get("percentage", 0) > 20 else "High"
            })
    
    # Overall sentiment health
    sentiment_counts = df[sentiment_col].value_counts()
    if "Positive" in sentiment_counts.index:
        pos_pct = 100 * sentiment_counts["Positive"] / len(df)
        if pos_pct < 50:
            recommendations.insert(0, {
                "type": "Health Check",
                "action": "Low positive sentiment detected",
                "details": f"Only {pos_pct:.1f}% positive reviews. Investigate and address root causes.",
                "priority": "Critical"
            })
        elif pos_pct > 80:
            recommendations.insert(0, {
                "type": "Health Check",
                "action": "Excellent customer satisfaction",
                "details": f"{pos_pct:.1f}% positive reviews. Maintain quality standards.",
                "priority": "Info"
            })
    
    return recommendations


def get_top_keywords_by_sentiment(df: pd.DataFrame, text_col: str = "clean_review", sentiment_col: str = "predicted_sentiment", top_n: int = 15) -> dict:
    """
    Get most frequent keywords per sentiment for quick insight.
    
    Args:
        df: DataFrame with reviews
        text_col: column name for review text
        sentiment_col: column name for sentiment
        top_n: number of top keywords to return
    
    Returns:
        Dictionary with top keywords per sentiment
    """
    keywords = {}
    
    for sentiment in ["Positive", "Negative", "Neutral"]:
        subset = df[df[sentiment_col] == sentiment][text_col]
        if len(subset) == 0:
            keywords[sentiment.lower()] = []
            continue
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english", min_df=1)
        try:
            vectorizer.fit(subset)
            feature_names = vectorizer.get_feature_names_out()
            # Get overall IDF scores (importance)
            idf_scores = vectorizer.idf_
            top_indices = idf_scores.argsort()[-top_n:][::-1]
            keywords[sentiment.lower()] = [{"word": feature_names[i], "importance": round(float(idf_scores[i]), 2)} for i in top_indices]
        except (ValueError, IndexError):
            keywords[sentiment.lower()] = []
    
    return keywords
