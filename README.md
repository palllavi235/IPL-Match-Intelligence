<h1 align="center">🏏 IPL Match Intelligence</h1>

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=28&pause=1000&color=FF6B00&center=true&vCenter=true&width=900&lines=IPL+Analytics+%F0%9F%8F%8F;Machine+Learning+Powered+Predictions+%F0%9F%A4%96;Pressure+Analysis+%F0%9F%94%A5;XGBoost+%7C+SHAP+%7C+Streamlit+%F0%9F%9A%80" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-E76F00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-6A5ACD?style=for-the-badge"/>
</p>

---

# 🚀 About The Project

🏏 An advanced cricket analytics + machine learning project that predicts IPL chase outcomes using real-time match pressure, wickets, momentum, and run-rate dynamics.

This project combines:

✨ Exploratory Data Analysis  
🤖 Machine Learning Models  
📊 SHAP Explainability  
🔥 Pressure Index Engineering  
🎯 Live Match Prediction  
🌐 Streamlit Deployment

---

# 🧠 Problem Statement

> What actually separates a successful IPL chase from a collapse under pressure?

This project studies live chase behavior using:

- 🎯 Target Score
- ⚡ Required Run Rate
- 🏏 Wickets Left
- 🔥 Pressure Index
- ⏳ Balls Remaining
- 📍 Venue Effects
- 🪙 Toss Decision
- 📈 Momentum Patterns

---

# 🏆 Final Model Performance

| 🚀 Model | 🎯 Accuracy | 📊 AUC Score |
|---|---|---|
| Logistic Regression | 80.96% | 0.898 |
| Random Forest | 83.25% | 0.922 |
| XGBoost 🔥 | **90.35%** | **0.970** |

---

# 🔥 Why XGBoost Won

✅ Captured pressure-based collapses  
✅ Learned wicket-risk interactions  
✅ Understood death-over pressure  
✅ Best class separation  
✅ Highest prediction confidence

---

# 📊 Dashboard Preview

## 🏏 Model Comparison

![Model accuracy comparison](images/model_accuracy_comparison.png)

## 📈 ROC Curve

![ROC curve comparison](images/roc_curve_comparison.png)

## 🧠 Feature Importance

![Feature importance](images/feature_importance.png)

## 🔍 SHAP Explainability

![SHAP summary](images/summary_plot.png)

---

# ⚡ Streamlit Features

✨ Live IPL Chase Predictor  
📊 Interactive Analytics Dashboard  
🔥 Pressure Analysis  
🧠 SHAP Explainability  
🎯 Match Situation Notes  
📈 Model Comparison  
🏏 Team & Venue Insights

---

# 🛠️ Tech Stack

<p align="center">

<img src="https://skillicons.dev/icons?i=python,vscode,git,github"/>

</p>

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib

---

# 📂 Project Structure

```bash
IPL-Match-Intelligence/
│
├── app.py
├── ipl.ipynb
├── ipl_chase_model.pkl
├── requirements.txt
├── README.md
│
├── images/
│   ├── model_accuracy_comparison.png
│   ├── roc_curve_comparison.png
│   ├── feature_importance.png
│   └── summary_plot.png
│
└── ipl.csv