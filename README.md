# IPL Match Intelligence

An end-to-end cricket analytics and machine learning project for understanding IPL chase behavior, pressure situations, wicket impact, and live match prediction.

This project combines exploratory data analysis, tactical cricket storytelling, machine learning model comparison, SHAP explainability, and a Streamlit dashboard that predicts whether an IPL chase is likely to succeed or fail.

## Project Snapshot

IPL matches often feel unpredictable, especially during run chases. A team can look comfortable for 12 overs and still collapse under pressure. This project studies that exact problem:

**What actually separates a successful IPL chase from a failed one?**

The analysis focuses on live chase variables such as:

- Target score
- Current score
- Runs remaining
- Balls remaining
- Wickets lost
- Wickets left
- Current run rate
- Required run rate
- Pressure index
- Toss decision
- Venue
- Chasing and defending teams

The final deployed model is **XGBoost**, selected because it achieved the best performance in the notebook.

## Live App Features

The Streamlit dashboard includes:

- Interactive IPL chase predictor
- EDA story sections with saved visualizations
- Machine learning model comparison
- Feature importance analysis
- SHAP explainability summary
- Live calculation of run rate, required rate, wickets left, balls remaining, and pressure index
- Match situation notes based on model inputs
- Current IPL teams shown first, with historical teams still supported where available

## Model Performance

Three models were trained and compared:

| Model | Accuracy | AUC Score |
|---|---:|---:|
| Logistic Regression | 80.96% | 0.898 |
| Random Forest | 83.25% | 0.922 |
| XGBoost | 90.35% | 0.970 |

XGBoost was selected for the final Streamlit app because it produced the highest accuracy and the strongest class separation.

## Why XGBoost Was Chosen

IPL chase prediction is not decided by one feature alone. A chase can change quickly when required run rate rises, wickets fall, and balls remaining decrease at the same time.

XGBoost performed best because it can capture these combined effects more effectively than simpler models. It learned pressure-based patterns involving:

- High required run rate with low wickets left
- Large targets with weak current scoring rate
- Rising pressure index in the middle and death overs
- Chase stability when wickets are preserved
- Difficult finishing situations with fewer balls remaining

## Dashboard Preview

### Model Comparison

![Model accuracy comparison](images/model_accuracy_comparison.png)

### ROC Curve Comparison

![ROC curve comparison](images/roc_curve_comparison.png)

### Feature Importance

![Feature importance](images/feature_importance.png)

### SHAP Summary

![SHAP summary](images/summary_plot.png)

## Key Analysis Insights

### Toss Is Useful, But Not Decisive

The toss can influence strategy, especially when teams choose to field first. However, the analysis suggests that toss advantage alone is much weaker than actual match execution.

### Middle Overs Matter More Than They Look

The middle overs often create the biggest separation between winners and losers. Stable scoring and wicket preservation between overs 7 and 15 are major signs of a controlled chase.

### Wickets Are Match Resources

The model and EDA both show that wickets are not just dismissals. They decide how much risk a batting side can take later in the innings.

### Required Run Rate Drives Pressure

Required run rate and pressure index were among the strongest predictors. When the required rate moves too far ahead of current scoring rate, the chase becomes much harder.

### SHAP Makes The Model Easier To Explain

SHAP shows not only which features matter, but also how they push the prediction toward success or failure. High required run rate, high pressure index, large targets, and wicket loss generally push predictions toward failed chases.

## Project Structure

```text
IPL-Match-Intelligence/
|-- app.py
|-- ipl.ipynb
|-- ipl_chase_model.pkl
|-- requirements.txt
|-- README.md
|-- images/
|   |-- model_accuracy_comparison.png
|   |-- roc_curve_comparison.png
|   |-- feature_importance.png
|   |-- summary_plot.png
|   |-- ...
|-- ipl.csv
```

## Streamlit App Pages

### Home

Introduces the project, key model metrics, and navigation.

### EDA Analysis

Shows toss behavior, scoring phases, venue effects, pressure analysis, chase patterns, player analysis, and momentum collapse.

### ML Analysis

Compares Logistic Regression, Random Forest, and XGBoost using accuracy, AUC, confusion matrices, feature importance, and SHAP.

### Predictor

Allows users to enter a live chase situation and get:

- Chase success probability
- Prediction label
- Required run rate
- Current run rate
- Balls remaining
- Runs remaining
- Match situation notes

### About

Explains the project idea, tools, and final model choice.

## How To Run Locally

Clone the repository:

```bash
git clone <your-repo-url>
cd IPL-Match-Intelligence
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Deployment Notes

For Streamlit Community Cloud, deploy these files:

```text
app.py
requirements.txt
ipl_chase_model.pkl
images/
```

The app does not need `ipl.csv` to run because it loads the trained model directly from `ipl_chase_model.pkl`.

Do not deploy:

```text
venv/
__pycache__/
```

## Model Input Design

The predictor page sends the following feature columns to the model:

```text
target
current_score
runs_remaining
balls_remaining
innings2_wickets
wickets_left
current_rr
required_rr
pressure_index
won_toss
chasing_team
defending_team
venue
toss_decision
batter_category
bowler_category
```

The app calculates most of the live match values automatically from user input.

## Player Category Note

The notebook trained the model using selected star-player categories and an `Others` category. The app includes more current player names for a better user experience, but names not seen during training are safely mapped to `Others` before prediction.

This keeps the app user-friendly without breaking the model pipeline.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Final Takeaway

This project shows that IPL chases are not only about power hitting. Successful chases usually leave measurable signals: controlled required run rate, wickets in hand, manageable pressure, and stable scoring momentum.

The Streamlit app turns those ideas into an interactive prediction system that reads the current match situation and estimates the chance of a successful chase.
