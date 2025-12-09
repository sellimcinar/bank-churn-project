# 🏦 AI Risk Sentinel: Bank Customer Retention System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/Model-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Live-success)

> **"Don't just track churn, prevent it."**

A next-generation **Decision Support System** designed for banking professionals. This application uses **Machine Learning (Random Forest)** to predict customer churn probability in real-time, helping banks identify high-risk accounts before they leave.

---

## 🚀 Live Demo
**[Click Here to Launch the App 🎈](https://bank-churn-project.streamlit.app/)**

---

## ⚡ Key Features

* **🔮 Real-Time Prediction Simulator:** Interactive sidebar to test "What-If" scenarios by adjusting customer parameters (Age, Balance, Credit Score).
* **🧠 Explainable AI (XAI):** "Feature Importance" analysis shows *why* the model made a specific decision (e.g., Is Age more critical than Balance?).
* **🎨 Cyberpunk / Dark UI:** A custom-designed, high-contrast user interface built with advanced CSS for maximum readability and modern aesthetics.
* **📄 Instant Reporting:** Auto-generates and downloads a detailed risk analysis report (`.txt`) for operational teams.
* **🌲 Algorithm Logic:** Powered by a **Random Forest Classifier** (The "Council of Experts" approach) for high accuracy and stability.
* **🎉 Gamification:** Interactive visual feedback (balloons) for safe/loyal customer profiles.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (Python) + Custom CSS
* **Backend Logic:** Python
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly (Interactive Gauges & Charts)

---

## 🌲 How It Works (The AI Model)

This system does not rely on a single guess. It utilizes the **Random Forest** algorithm:
1.  **Council of Experts:** The model builds 100+ Decision Trees, each acting as an individual expert trained on different data subsets.
2.  **Voting Mechanism:** When a new customer profile is input, every tree votes on whether the customer will "Churn" or "Stay".
3.  **Majority Verdict:** The system takes the majority vote to produce the final probability score, ensuring robustness against overfitting.

---

## 💻 Local Installation

To run this application on your local machine:

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/KULLANICI_ADIN/bank-churn-project.git](https://github.com/KULLANICI_ADIN/bank-churn-project.git)
    cd bank-churn-project
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app**
    ```bash
    streamlit run app.py
    ```

---

## 👨‍💻 Author

Developed by **[Abdullah Selim Cinar]**
* *Data Science & Analytics Student*
* *Powered by Random Forest AI*

---
