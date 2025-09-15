import requests
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# NLTK Setup (downloads if not already present)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configuration
API_KEY = os.environ.get('NEWSAPI_KEY')
if not API_KEY:
    raise ValueError("NEWSAPI_KEY environment variable not set.")
DAYS_BACK = 3  # Number of days to analyze (excluding today)

def fetch_headlines(days_back=DAYS_BACK):
    """
    Fetch headlines for the last (days_back + 1) days using NewsAPI.
    """
    headlines = []
    for i in range(days_back, -1, -1):
        date = (datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d')
        url = f'https://newsapi.org/v2/everything?q=news&from={date}&to={date}&sortBy=popularity&apiKey={API_KEY}'
        try:
            response = requests.get(url).json()
            if response.get('status') != 'ok':
                print(f"Error fetching data for {date}: {response.get('message', 'Unknown error')}")
                continue
            day_headlines = [article['title'] for article in response.get('articles', []) if article['title']]
            headlines.append({'date': date, 'headlines': day_headlines})
        except Exception as e:
            print(f"Exception fetching data for {date}: {e}")
    return headlines

def preprocess(text):
    """
    Preprocess text: tokenize, lowercase, remove stopwords and non-alpha tokens.
    """
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(tokens)

def vectorize_past_headlines(past_headlines):
    """
    Vectorize headlines from past days using TF-IDF.
    """
    all_past_text = [preprocess(h) for day in past_headlines for h in day['headlines']]
    if not all_past_text:
        return None, None
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_past_text)
    return tfidf_matrix, vectorizer

def find_similar_events(tfidf_matrix, vectorizer, current_headlines):
    """
    Compute average cosine similarity of current headlines to past headlines.
    """
    if tfidf_matrix is None:
        return [0.0] * len(current_headlines)
    current_vectors = vectorizer.transform([preprocess(h) for h in current_headlines])
    similarities = cosine_similarity(current_vectors, tfidf_matrix)
    avg_similarity = similarities.mean(axis=1)
    return avg_similarity

def predict_outcomes(headlines):
    """
    Compute similarity scores (as probabilities) for today's headlines compared to past.
    """
    if len(headlines) < 2:
        print("Not enough historical data to compute similarities.")
        return [{'date': headlines[0]['date'], 'headline': h, 'probability': 0.0} for h in headlines[0]['headlines']]
    
    past_headlines = headlines[:-1]
    current = headlines[-1]
    tfidf_matrix, vectorizer = vectorize_past_headlines(past_headlines)
    similarity_scores = find_similar_events(tfidf_matrix, vectorizer, current['headlines'])
    
    results = []
    for h, score in zip(current['headlines'], similarity_scores):
        results.append({'date': current['date'], 'headline': h, 'probability': round(score * 100, 2)})
    return results

def plot_matrix_heatmap(predictions):
    """
    Generate a Matrix-themed heatmap of headline probabilities.
    """
    df = pd.DataFrame(predictions)
    # Sort by probability descending for better visualization
    df = df.sort_values('probability', ascending=False)
    heatmap_data = df.pivot(index='headline', columns='date', values='probability').fillna(0)
    
    plt.figure(figsize=(14, len(df) * 0.3 + 2))  # Adjust height based on number of headlines
    sns.set(style="white")
    ax = sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".1f", 
        cmap="Greens", 
        cbar_kws={'label': 'Similarity Probability (%)'}, 
        linewidths=0.5, 
        linecolor='#00FF00',
        annot_kws={"size": 10, "weight": 'bold', "color": "#00FF00"}
    )
    
    # Dark background
    ax.set_facecolor('black')
    plt.gcf().patch.set_facecolor('black')
    
    plt.title('Matrix-Themed Headline Similarity Heatmap', fontsize=18, color='#00FF00', weight='bold')
    plt.ylabel('Headline', fontsize=12, color='#00FF00')
    plt.xlabel('Date', fontsize=12, color='#00FF00')
    
    plt.xticks(rotation=45, color='#00FF00')
    plt.yticks(rotation=0, color='#00FF00')
    
    plt.tight_layout()
    plt.show()

def run_pipeline():
    """
    Run the full pipeline: fetch, predict, plot, and print top predictions.
    """
    headlines_data = fetch_headlines()
    if not headlines_data:
        print("No data fetched.")
        return
    predictions = predict_outcomes(headlines_data)
    plot_matrix_heatmap(predictions)
    # Print top predictions
    print("Top Predicted Headlines:")
    top_predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)[:10]
    for p in top_predictions:
        print(f"{p['date']}: {p['headline']} --> Probability: {p['probability']}%")

# Schedule Daily Run (Example for Linux/Mac Cron Job)
# Save this script as daily_headlines.py
# Add the following cron entry to run at 8 AM daily:
# 0 8 * * * /usr/bin/python3 /path/to/daily_headlines.py

if __name__ == "__main__":
    run_pipeline()
