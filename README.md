Matrix News Probability Heatmap

**Matrix News Probability Heatmap** is a Python-based predictive system that analyzes daily news headlines, calculates the probability of major events based on historical trends, and visualizes them in a **Matrix-themed heatmap**. It is fully compatible with **Termux** on Android and can be automated to run daily.

---

## Features

- Fetches the last 3 days of headlines from NewsAPI.
- Preprocesses and vectorizes text using TF-IDF for pattern analysis.
- Predicts the probability of headline relevance based on recent trends.
- Generates a **Matrix-style heatmap** (dark theme with neon green text) for easy visualization.
- Saves heatmaps as PNG files (no GUI required) for Termux.
- Can be automated to run daily using Termux Job Scheduler or loops.

---

## Installation

1. **Install Termux packages**:
```bash
pkg install python
pkg install git

2. Clone this repository:



git clone https://github.com/YOUR_USERNAME/matrix-news-heatmap.git
cd matrix-news-heatmap

3. Install Python dependencies:



pip install requests nltk scikit-learn matplotlib seaborn pandas

4. Download NLTK data:



import nltk
nltk.download('punkt')
nltk.download('stopwords')

5. Add your NewsAPI key:
Replace 'YOUR_NEWSAPI_KEY' in the script with your API key.




---

Usage

Run the script in Termux:

python3 daily_headlines.py

The heatmap will be saved to:

/data/data/com.termux/files/home/news_heatmaps/heatmap_YYYY-MM-DD.png

Top predicted headlines will be printed in the console.


---

Automation

Option 1: Termux Job Scheduler

termux-job-scheduler --period-ms 86400000 --persisted true -- python3 /data/data/com.termux/files/home/daily_headlines.py

Option 2: Loop Script

while true; do
    python3 daily_headlines.py
    sleep 86400  # Run once every 24 hours
done


---

Customization

DAYS_BACK: Change the number of past days to analyze (default is 3).

Output Directory: Change OUTPUT_DIR to save heatmaps elsewhere.

Theme: Customize sns.heatmap colors for different Matrix-style aesthetics.



---

Dependencies

Python 3

requests

nltk

scikit-learn

matplotlib

seaborn

pandas



---

License

This project is licensed under the MIT License.


---
