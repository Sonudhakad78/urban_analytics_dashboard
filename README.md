# 🇮🇳 Urban Development & Public Sentiment Dashboard – India

An interactive **Streamlit + Machine Learning** dashboard for analyzing Indian urban complaints, public sentiment, and transit performance.  
It visualizes open civic data, applies predictive analytics, and presents insights in a **Power BI–inspired design**.

---

## 🚀 Features
- 🗺️ **Overview Map:** Heatmap of 311 complaints across Indian cities  
- 💬 **Sentiment Analysis:** NLP sentiment scoring on social media posts  
- 🚍 **Transit KPIs:** Route-wise average delays and performance charts  
- 🧠 **Topic Modeling:** Complaint clustering using TF-IDF + K-Means  
- 📊 **Predictive Risk Score:** Random Forest model to detect high-risk areas  
- 📥 **Export Reports:** Download CSV outputs for every section  
- 🎨 **Power BI Theme:** Dark sidebar, yellow highlights, and white data cards  

---

## 🧠 Tech Stack
| Category | Tools |
|-----------|-------|
| **Frontend** | Streamlit |
| **Data Science** | pandas, geopandas, scikit-learn, TextBlob |
| **Visualization** | Plotly Express |
| **Machine Learning** | Random Forest Classifier |
| **Data Formats** | CSV, GeoJSON |
| **Styling** | Custom CSS (Power BI look) |

---

## 📂 Project Structure
urban_analytics/
│
├── app.py                  # Streamlit main app
├── train_model.py          # Trains Random Forest risk model
├── sentiment.py            # Sentiment analysis helper
├── topic_model.py          # Topic modeling helper
│
├── data/                   # Input datasets
│   ├── 311_data.csv
│   ├── social_posts.csv
│   ├── transit.csv
│   └── neighborhoods.geojson
│
├── models/                 # Saved ML models
│   └── risk_model.joblib
│
└── README.md
---

## ⚙️ Setup Guide

### 1️⃣ Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
pip install streamlit pandas geopandas plotly scikit-learn textblob joblib shapely pyproj
python train_model.py
streamlit run app.py

Then open http://localhost:8501 in your browser.

⸻

📸 Dashboard Pages
Page
Description
🗺️ Overview Map
Complaint density and category summary
💬 Sentiment
Public emotion analysis from social media
🚍 Transit
Delay KPIs by route
🧠 Topics
Complaint clusters (TF-IDF + K-Means)
📊 NeedScore & Risk
Predictive risk scoring for cities
📈 Insights
	•	Solid-waste and traffic dominate citizen complaints
	•	Sentiment analysis shows mostly neutral public mood
	•	Transit delays vary by metro region
	•	Predictive risk model highlights critical zones for planners

⸻

🧩 Future Enhancements
	•	Real-time civic data APIs
	•	Multilingual NLP for Indian languages
	•	BERT-based sentiment analysis
	•	Auto-generated PDF city reports

⸻

👨‍💻 Author

Sonu Dhakad
Department of Information Technology
Oriental Institute of Science and Technology, Bhopal (India)

⸻

📜 License

Released under the MIT License.