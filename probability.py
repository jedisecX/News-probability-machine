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

# --- NLTK Setup ---
nltk.download('punkt')
nltk.download('stopwords')

# --- Configuration ---
API_KEY = 'YOUR_NEWSAPI_KEY'
DAYS_BACK = 3  # Number of days to analyze

# --- 1. Fetch Headlines ---
def fetch_headlines(days_back=DAYS_BACK):
    headlines = []
    for i in range(days_back, -1, -1):
        date = (datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d')
        url = f'https://newsapi.org/v2/everything?q=*&from={date}&to={date}&sortBy=popularity&apiKey={API_KEY}'
        response = requests.get(url).json()
        day_headlines = [article['title'] for article in response.get('articles', [])]
        headlines.append({'date': date, 'headlines': day_headlines})
    return headlines

# --- 2. Preprocess Headlines ---
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stopwords.words('english')]
    return ' '.join(tokens)

# --- 3. Vectorize Headlines ---
def vectorize_headlines(headlines):
    all_text = []
    for day in headlines:
        for h in day['headlines']:
            all_text.append(preprocess(h))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)
    return tfidf_matrix, vectorizer

# --- 4. Find Similar Events ---
def find_similar_events(tfidf_matrix, vectorizer, current_headlines):
    current_vectors = vectorizer.transform([preprocess(h) for h in current_headlines])
    similarities = cosine_similarity(current_vectors, tfidf_matrix)
    avg_similarity = similarities.mean(axis=1)
    return avg_similarity

# --- 5. Predict Probable Outcomes ---
def predict_outcomes(headlines):
    tfidf_matrix, vectorizer = vectorize_headlines(headlines)
    today_headlines = headlines[-1]['headlines']
    similarity_scores = find_similar_events(tfidf_matrix, vectorizer, today_headlines)
    
    results = []
    for h, score in zip(today_headlines, similarity_scores):
        results.append({'date': headlines[-1]['date'], 'headline': h, 'probability': round(score*100, 2)})
    return results

# --- 6. Generate Matrix-Themed Heatmap ---
def plot_matrix_heatmap(predictions):
    df = pd.DataFrame(predictions)
    heatmap_data = df.pivot(index='headline', columns='date', values='probability').fillna(0)
    
    plt.figure(figsize=(14, 9))
    sns.set(style="white")
    ax = sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".1f", 
        cmap="Greens", 
        cbar_kws={'label': 'Probability (%)'}, 
        linewidths=0.5, 
        linecolor='#00FF00',
        annot_kws={"size":10, "weight":'bold', "color":"#00FF00"}
    )
    
    # Dark background
    plt.gca().set_facecolor('black')
    plt.gcf().patch.set_facecolor('black')
    
    plt.title('Matrix-Themed Headline Probability Heatmap', fontsize=18, color='#00FF00', weight='bold')
    plt.ylabel('Headline', fontsize=12, color='#00FF00')
    plt.xlabel('Date', fontsize=12, color='#00FF00')
    
    plt.xticks(rotation=45, color='#00FF00')
    plt.yticks(rotation=0, color='#00FF00')
    
    plt.tight_layout()
    plt.show()

# --- 7. Full Pipeline ---
def run_pipeline():
    headlines_data = fetch_headlines()
    predictions = predict_outcomes(headlines_data)
    plot_matrix_heatmap(predictions)
    # Optional: print top predictions
    print("Top Predicted Headlines:")
    for p in sorted(predictions, key=lambda x: x['probability'], reverse=True)[:10]:
        print(f"{p['date']}: {p['headline']} --> Probability: {p['probability']}%")

# --- 8. Schedule Daily Run (Example for Linux/Mac Cron Job) ---
# Save this script as daily_headlines.py
# Add the following cron entry to run at 8 AM daily:
# 0 8 * * * /usr/bin/python3 /path/to/daily_headlines.py

if __name__ == "__main__":
    run_pipeline()
