# ❤️ HeartHealth AI: Predictive Diagnostic Tool

HeartHealth AI is a modern, web-based machine learning application designed to predict the likelihood of heart disease based on clinical parameters. Built with **Flask** and **XGBoost**, the app features a high-end **Glassmorphism UI** for a seamless user experience.



## 🚀 Features
* **Real-time Prediction:** Instant results using a pre-trained XGBoost model.
* **Risk Probability:** Visual feedback through a dynamic progress bar indicating risk percentage.
* **Modern UI:** A sleek, dark-themed dashboard using CSS glassmorphism and responsive design.
* **Robust Backend:** Efficient data processing using `pandas` and `joblib`.

## 🛠️ Tech Stack
* **Frontend:** HTML5, CSS3 (Glassmorphism), Jinja2
* **Backend:** Flask (Python)
* **Machine Learning:** XGBoost, Scikit-learn
* **Data Handling:** Pandas, NumPy

## 📋 Clinical Parameters Used
The model analyzes the following features to determine cardiovascular risk:
1.  **Sex:** (Male/Female)
2.  **Chest Pain Type:** (Typical Angina, Atypical, Non-anginal, Asymptomatic)
3.  **Cholesterol:** Serum cholestoral in mg/dl
4.  **Resting ECG:** Electrocardiographic results
5.  **Exercise Induced Angina:** Presence of chest pain during exercise
6.  **Oldpeak:** ST depression induced by exercise relative to rest
7.  **ST Slope:** The slope of the peak exercise ST segment

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/hearthealth-ai.git](https://github.com/yourusername/hearthealth-ai.git)
   cd hearthealth-ai
