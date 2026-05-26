import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ipl_chase_model.pkl"
IMAGE_DIR = BASE_DIR / "images"

NUMERIC_FEATURES = [
    "target",
    "current_score",
    "runs_remaining",
    "balls_remaining",
    "innings2_wickets",
    "wickets_left",
    "current_rr",
    "required_rr",
    "pressure_index",
    "won_toss",
]

CATEGORICAL_FEATURES = [
    "chasing_team",
    "defending_team",
    "venue",
    "toss_decision",
    "batter_category",
    "bowler_category",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FALLBACK_TEAMS = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans",
    "Deccan Chargers",
    "Pune Warriors",
    "Gujarat Lions",
]

FALLBACK_VENUES = [
    "MA Chidambaram Stadium, Chepauk",
    "Wankhede Stadium, Mumbai",
    "M Chinnaswamy Stadium",
    "Eden Gardens",
    "Arun Jaitley Stadium",
    "Rajiv Gandhi International Stadium, Uppal",
    "Narendra Modi Stadium, Ahmedabad",
    "Sawai Mansingh Stadium",
    "Punjab Cricket Association Stadium, Mohali",
    "Brabourne Stadium",
    "Dr DY Patil Sports Academy",
    "Sharjah Cricket Stadium",
]

FALLBACK_BATTERS = [
    "Others",
    "Virat Kohli",
    "MS Dhoni",
    "Rohit Sharma",
    "David Warner",
    "KL Rahul",
    "AB de Villiers",
    "Shubman Gill",
    "Yashasvi Jaiswal",
    "Suryakumar Yadav",
    "Ruturaj Gaikwad",
    "Rishabh Pant",
    "Sanju Samson",
    "Jos Buttler",
    "Faf du Plessis",
    "Glenn Maxwell",
    "Andre Russell",
    "Hardik Pandya",
    "Rinku Singh",
    "Heinrich Klaasen",
    "Travis Head",
]

FALLBACK_BOWLERS = [
    "Others",
    "JJ Bumrah",
    "Rashid Khan",
    "SL Malinga",
    "YS Chahal",
    "DJ Bravo",
    "B Kumar",
    "Mohammed Shami",
    "Mohammed Siraj",
    "Arshdeep Singh",
    "Kuldeep Yadav",
    "Ravindra Jadeja",
    "Sunil Narine",
    "Varun Chakaravarthy",
    "Trent Boult",
    "Kagiso Rabada",
    "Pat Cummins",
    "T Natarajan",
    "Matheesha Pathirana",
]

CURRENT_IPL_TEAMS = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans",
]


st.set_page_config(
    page_title="IPL Match Intelligence",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def image_path(name):
    return str(IMAGE_DIR / name)


def get_classifier_name(pipeline):
    classifier = getattr(pipeline, "named_steps", {}).get("classifier")
    if classifier is None:
        return "Unknown model"
    return classifier.__class__.__name__


def get_model_categories(pipeline):
    categories = {}
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        cat_pipe = preprocessor.named_transformers_["cat"]
        encoder = cat_pipe.named_steps["encoder"]
        for feature, values in zip(CATEGORICAL_FEATURES, encoder.categories_):
            categories[feature] = sorted(str(value) for value in values)
    except Exception:
        return {}
    return categories


model = load_model()
model_categories = get_model_categories(model)
classifier_name = get_classifier_name(model)


def options_for(feature, fallback):
    values = model_categories.get(feature, fallback)
    cleaned = [value for value in values if value and value != "nan"]
    if feature in {"batter_category", "bowler_category"} and "Others" in cleaned:
        cleaned = ["Others"] + [value for value in cleaned if value != "Others"]
    return cleaned or fallback


def team_options(feature):
    values = options_for(feature, FALLBACK_TEAMS)
    current = [team for team in CURRENT_IPL_TEAMS if team in values]
    legacy = [team for team in values if team not in current]
    return current + legacy


def player_display_options(feature, fallback):
    trained_values = set(options_for(feature, fallback))
    display = []
    for player in fallback + sorted(trained_values):
        if player not in display:
            display.append(player)
    if "Others" in display:
        display.remove("Others")
        display.insert(0, "Others")
    return display


def model_player_category(selected, feature):
    trained_values = set(options_for(feature, ["Others"]))
    if selected in trained_values:
        return selected
    return "Others"


def render_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #020617, #0f172a, #111827);
            color: white;
        }

        h1, h2, h3, h4, h5, h6 { color: white; }
        p, li, label { color: #cbd5e1; }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1120, #111827, #020617);
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        [data-testid="stSidebar"] h1 {
            font-size: 34px !important;
            font-weight: 850 !important;
            line-height: 1.15;
            background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-title {
            font-size: clamp(42px, 6vw, 72px);
            font-weight: 900;
            line-height: 1.08;
            color: white;
            margin-bottom: 12px;
            text-shadow: 0 0 30px rgba(59,130,246,0.34);
        }

        .hero-sub {
            font-size: 21px;
            line-height: 1.8;
            color: #cbd5e1;
            max-width: 1180px;
            margin-bottom: 28px;
        }

        .hero-panel {
            padding: 44px;
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(37,99,235,0.22), rgba(124,58,237,0.20), rgba(219,39,119,0.16));
            background-size: 180% 180%;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 0 48px rgba(59,130,246,0.18);
            margin-bottom: 34px;
            animation: panelShift 9s ease infinite;
        }

        .card {
            min-height: 300px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            background: linear-gradient(145deg, rgba(17,24,39,0.96), rgba(30,41,59,0.90));
            backdrop-filter: blur(18px);
            padding: 30px;
            border-radius: 26px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 0 25px rgba(0,0,0,0.28);
            transition: all 0.35s ease;
            overflow: hidden;
            position: relative;
        }

        .card:hover {
            transform: translateY(-6px);
            box-shadow: 0 0 44px rgba(59,130,246,0.22), 0 0 64px rgba(168,85,247,0.12);
            border: 1px solid rgba(96,165,250,0.28);
        }

        .card h2 {
            font-size: 28px;
            font-weight: 760;
            color: white;
            margin-bottom: 4px;
            line-height: 1.25;
        }

        .card p, .card div {
            line-height: 1.85;
            font-size: 17px;
            color: #cbd5e1;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 20px;
            margin: 20px 0 28px;
        }

        .metric-card {
            min-height: 190px;
            padding: 28px;
            border-radius: 24px;
            text-align: center;
            background: linear-gradient(135deg, rgba(37,99,235,0.28), rgba(124,58,237,0.18));
            border: 1px solid rgba(255,255,255,0.09);
            box-shadow: 0 0 28px rgba(0,0,0,0.25);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }

        .metric-card h1 {
            font-size: clamp(34px, 4vw, 56px);
            font-weight: 900;
            margin: 0 0 8px;
            line-height: 1.05;
        }

        .metric-card p {
            color: #e2e8f0;
            font-size: 19px;
            line-height: 1.45;
            margin: 0;
        }

        .section-container, .result-card, .footer-box, .insight-card, .state-card {
            background: linear-gradient(135deg, rgba(15,23,42,0.94), rgba(30,41,59,0.88));
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 0 30px rgba(0,0,0,0.30);
        }

        .section-container {
            padding: 28px;
            border-radius: 26px;
            margin: 20px 0 24px;
        }

        .result-card {
            padding: 42px;
            border-radius: 30px;
            text-align: center;
            background: linear-gradient(135deg, rgba(16,185,129,0.26), rgba(37,99,235,0.24), rgba(124,58,237,0.24));
            animation: resultRise 0.65s ease both;
        }

        .probability {
            font-size: clamp(56px, 8vw, 96px);
            font-weight: 900;
            color: white;
            margin-top: 12px;
            text-shadow: 0 0 30px rgba(255,255,255,0.20);
        }

        .insight-card {
            padding: 20px 22px;
            border-radius: 18px;
            margin-bottom: 14px;
            font-size: 17px;
            line-height: 1.75;
            color: #e2e8f0;
            animation: resultRise 0.45s ease both;
        }

        .state-card {
            padding: 22px;
            border-radius: 20px;
            text-align: center;
            min-height: 130px;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }

        .state-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 0 32px rgba(59,130,246,0.22);
        }

        .state-card h3 {
            font-size: 16px;
            color: #94a3b8;
            margin-bottom: 8px;
        }

        .state-card h2 {
            font-size: 30px;
            margin: 0;
        }

        .footer-box {
            margin-top: 40px;
            padding: 34px;
            border-radius: 26px;
            line-height: 1.9;
            font-size: 17px;
            color: #d1fae5;
        }

        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #2563eb, #7c3aed, #db2777);
            background-size: 220% 220%;
            color: white;
            border: none;
            padding: 14px 22px;
            border-radius: 16px;
            font-size: 17px;
            font-weight: 760;
            transition: all 0.28s ease;
            box-shadow: 0 0 20px rgba(99,102,241,0.30);
            animation: buttonGlow 3.2s ease-in-out infinite;
        }

        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 0 34px rgba(59,130,246,0.45);
        }

        .predictor-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
            margin: -10px 0 28px;
        }

        .mini-chip {
            padding: 18px 20px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(37,99,235,0.16));
            border: 1px solid rgba(255,255,255,0.08);
            color: #dbeafe;
            font-size: 16px;
            line-height: 1.55;
            box-shadow: 0 0 24px rgba(0,0,0,0.22);
            animation: resultRise 0.7s ease both;
        }

        .mini-chip strong {
            display: block;
            color: white;
            font-size: 20px;
            margin-bottom: 4px;
        }

        .confidence-bar {
            position: relative;
            height: 14px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(148,163,184,0.18);
            margin-top: 22px;
        }

        .confidence-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);
            box-shadow: 0 0 24px rgba(16,185,129,0.34);
            animation: fillIn 1s ease both;
        }

        @keyframes panelShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes buttonGlow {
            0% { background-position: 0% 50%; box-shadow: 0 0 18px rgba(99,102,241,0.26); }
            50% { background-position: 100% 50%; box-shadow: 0 0 30px rgba(236,72,153,0.34); }
            100% { background-position: 0% 50%; box-shadow: 0 0 18px rgba(99,102,241,0.26); }
        }

        @keyframes resultRise {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fillIn {
            from { width: 0%; }
        }

        .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
            min-height: 58px;
            border-radius: 16px !important;
            background: linear-gradient(135deg, rgba(30,41,59,0.98), rgba(51,65,85,0.92)) !important;
            color: white !important;
            border: 1px solid rgba(124,58,237,0.24) !important;
        }

        @media (max-width: 900px) {
            .metric-grid { grid-template-columns: 1fr; }
            .predictor-strip { grid-template-columns: 1fr; }
            .hero-panel { padding: 30px; }
            .card { min-height: auto; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(title, body):
    formatted = body.strip().replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="card">
            <h2>{title}</h2>
            <div>{formatted}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_section(title, image, heading, insight):
    st.markdown("---")
    st.subheader(title)
    col1, col2 = st.columns([1.35, 1], vertical_alignment="center")
    with col1:
        st.image(image_path(image), use_container_width=True)
    with col2:
        card(heading, insight)


def sidebar():
    st.sidebar.markdown("# IPL Match Intelligence")
    st.sidebar.markdown(
        """
        <p style="color:#94a3b8; font-size:15px; line-height:1.8; margin-top:-8px;">
        Data-driven IPL chase analysis
        </p>
        """,
        unsafe_allow_html=True,
    )

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    pages = ["Home", "EDA Analysis", "ML Analysis", "Predictor", "About"]
    page = st.sidebar.radio(
        "Explore Dashboard",
        pages,
        index=pages.index(st.session_state.page),
    )
    st.session_state.page = page

    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div style="
            padding:20px;
            border-radius:20px;
            background:linear-gradient(135deg, rgba(59,130,246,0.18), rgba(168,85,247,0.15));
            border:1px solid rgba(255,255,255,0.08);
            text-align:center;">
            <h3 style="color:white;">Final Model</h3>
            <p style="color:#cbd5e1; line-height:1.8; font-size:14px;">
            {classifier_name}<br>
            Accuracy: 90.35%<br>
            AUC: 0.970
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page


def render_home():
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title">IPL Match Intelligence System</div>
            <div class="hero-sub">
            A data project for studying IPL chases: how targets, wickets,
            required run rate, venue context, and pressure shape match outcomes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="metric-grid">
            <div class="metric-card">
                <h1>90.35%</h1>
                <p>Best Accuracy<br>XGBoost</p>
            </div>
            <div class="metric-card">
                <h1>0.970</h1>
                <p>Best AUC Score<br>Class Separation</p>
            </div>
            <div class="metric-card">
                <h1>89.42%</h1>
                <p>Tuned Random Forest<br>Test Accuracy</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        card(
            "EDA Analysis",
            """
            Explore toss behavior, chase-friendly venues, phase scoring,
            middle-over control, pressure changes, wicket impact,
            clutch players, and tactical match patterns.
            """,
        )
        if st.button("Open EDA Analysis", use_container_width=True):
            st.session_state.page = "EDA Analysis"
            st.rerun()

    with col2:
        card(
            "ML Analysis",
            """
            Compare Logistic Regression, Random Forest, tuned Random Forest,
            and XGBoost using accuracy, AUC, confusion matrices,
            feature importance, and SHAP.
            """,
        )
        if st.button("Open ML Analysis", use_container_width=True):
            st.session_state.page = "ML Analysis"
            st.rerun()

    col3, col4 = st.columns(2)
    with col3:
        card(
            "Predictor",
            """
            Predict live IPL chase success using target, current score,
            balls remaining, wickets lost, required run rate,
            pressure index, teams, venue, toss, and player category.
            """,
        )
        if st.button("Open Predictor", use_container_width=True):
            st.session_state.page = "Predictor"
            st.rerun()

    with col4:
        card(
            "About",
            """
            A complete project that connects cricket analysis,
            machine learning, model interpretation, and a live predictor.
            """,
        )
        if st.button("Open About", use_container_width=True):
            st.session_state.page = "About"
            st.rerun()


def render_eda():
    st.title("IPL Chase Analysis")
    st.markdown(
        """
        <p style="font-size:20px; color:#cbd5e1;">
        Exploring toss behavior, scoring phases, venue effects, pressure,
        wickets, momentum collapse, and chase dynamics across IPL matches.
        </p>
        """,
        unsafe_allow_html=True,
    )

    sections = [
        (
            "Toss Winner vs Match Winner",
            "toss_analysis.png",
            "The Toss Helps, But It Does Not Decide Everything",
            """
            The toss is often treated as a match-changing moment, but the data shows a more balanced story.
            Toss winners do not automatically become match winners.

            The result depends much more on how teams bat, bowl, and handle pressure after the toss.
            """,
        ),
        (
            "Toss Decision Impact",
            "toss_decision_impact.png",
            "Why Chasing Is So Popular",
            """
            Teams choosing to field first showed stronger results than teams choosing to bat first.

            This supports the modern IPL preference for chasing. Teams often like knowing the target and pacing the innings around a clear number.
            """,
        ),
        (
            "Average Runs by Phase",
            "phase_avg.png",
            "Middle Overs Do A Lot Of Quiet Work",
            """
            Middle overs contribute the largest run volume because they cover the longest phase.
            This is where teams rotate strike, protect wickets, and set up the final push.
            """,
        ),
        (
            "Scoring Intensity",
            "phase_intensity.png",
            "Death Overs Are The Fastest Phase",
            """
            Death overs show the highest run rate per over.
            Teams that keep enough wickets can usually attack harder at the end.
            """,
        ),
        (
            "Winner vs Loser Phase Scoring",
            "winner_phase_comparison.png",
            "The Middle Overs Create The Gap",
            """
            Winners separate themselves most clearly during the middle overs.
            This suggests that overs 7 to 15 often decide whether a chase stays calm or becomes difficult.
            """,
        ),
        (
            "Innings Phase Comparison",
            "innings_phase_comparison.png",
            "Scoreboard Pressure Changes The Ending",
            """
            First-innings teams usually score more freely at the death.
            Chasing teams either face more pressure late or finish before using all death overs, so their pattern looks different.
            """,
        ),
        (
            "Defending Success by Score",
            "defending_success.png",
            "The 180-Run Pressure Line",
            """
            Scores above 180 create a noticeable defending advantage.
            Once the target crosses this range, chasing teams face higher required run rates and a greater chance of pressure-driven mistakes.
            """,
        ),
        (
            "Venue Intelligence",
            "venue_analysis.png",
            "Venues Change The Game",
            """
            IPL venues do not play the same way.
            Some grounds reward aggressive chasing, while others support defensive bowling, bigger boundaries, or scoreboard pressure.
            """,
        ),
        (
            "Successful vs Failed Chases",
            "chase_analysis.png",
            "Stable Chases Usually Win",
            """
            Successful chases are built through stable powerplay and middle-over scoring.
            Failed chases often fall behind earlier and then need risky acceleration later.
            """,
        ),
        (
            "Momentum Collapse",
            "momentum.png",
            "Wickets Break Acceleration",
            """
            Losing four or more wickets in the middle overs sharply reduces death-over scoring potential.
            Once batting resources disappear, even aggressive intent cannot fully recover the chase.
            """,
        ),
        (
            "Pressure Analysis",
            "pressure_analysis.png",
            "Pressure Does Not Always Mean Slow Scoring",
            """
            Scoring rate does not collapse automatically under pressure.
            The important question is whether teams can keep wickets and avoid turning pressure into a collapse.
            """,
        ),
        (
            "Correlation Heatmap",
            "correlation_heatmap.png",
            "Live Match Variables Matter Most",
            """
            Required run rate, pressure index, wickets, runs remaining, and balls remaining carry stronger signal than toss narratives.
            The chase situation itself matters most.
            """,
        ),
        (
            "Top Batters",
            "top_batters.png",
            "Consistency Over Time",
            """
            The batting leaderboard highlights players who combine consistency with long-term run production across seasons.
            This adds player context to the match-level analysis.
            """,
        ),
        (
            "Top Bowlers",
            "top_bowlers.png",
            "Wicket-Taking Value",
            """
            Leading wicket-takers influence matches by breaking partnerships and creating pressure.
            Their impact connects directly with the wicket-preservation theme in the analysis.
            """,
        ),
        (
            "IPL Champions",
            "champions.png",
            "Why Some Teams Keep Winning",
            """
            Successful franchises usually combine squad quality, adaptability, pressure handling, and finishing ability.
            Championship success is not only about star power, but repeatable execution.
            """,
        ),
    ]

    for section in sections:
        insight_section(*section)

    st.markdown("---")
    st.success(
        """
        Final EDA conclusion: IPL chases are not just random hitting.
        Successful teams preserve wickets, control middle overs, manage pressure,
        and accelerate when the situation gives them enough room.
        """
    )


def render_ml():
    st.title("IPL Machine Learning Intelligence")
    st.markdown(
        """
        <p style="font-size:20px; color:#cbd5e1;">
        Updated model insights based on the current notebook results:
        Logistic Regression 80.96%, Random Forest 83.25%,
        tuned Random Forest 89.42%, and XGBoost 90.35%.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="metric-grid">
            <div class="metric-card">
                <h1>80.96%</h1>
                <p>Logistic Regression<br>AUC 0.898</p>
            </div>
            <div class="metric-card">
                <h1>83.25%</h1>
                <p>Random Forest<br>AUC 0.922</p>
            </div>
            <div class="metric-card">
                <h1>90.35%</h1>
                <p>XGBoost<br>AUC 0.970</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ml_sections = [
        (
            "Model Accuracy Comparison",
            "model_accuracy_comparison.png",
            "XGBoost Worked Best For Prediction",
            """
            Model performance improved as the algorithm became better at reading combinations of match factors.
            Logistic Regression gave a solid baseline, Random Forest improved the interpretation, and XGBoost produced the best final result.

            Accuracy progression: 80.96% to 83.25% to 90.35%.
            """,
        ),
        (
            "Logistic Regression",
            "confusion_matrix_lr.png",
            "A Strong Baseline",
            """
            Logistic Regression achieved 80.96% accuracy with an AUC of 0.898.
            This suggests that the engineered chase features already contain useful signal.

            It identified successful chases slightly better than failed ones, suggesting that winning chases often leave clearer patterns in the scorecard.
            """,
        ),
        (
            "Random Forest",
            "confusion_matrix_rf.png",
            "Learning Feature Interactions",
            """
            Random Forest reached 83.25% accuracy and 0.922 AUC.
            It improved over Logistic Regression by learning how required run rate, pressure, wickets, target, and runs remaining work together.

            This model is especially useful for understanding what the prediction is paying attention to.
            """,
        ),
        (
            "XGBoost",
            "confusion_matrix_xgb.png",
            "Best Final Model",
            """
            XGBoost achieved the strongest performance: 90.35% accuracy and 0.970 AUC.
            It reduced both false successful predictions and false failed predictions compared with the earlier models.

            This is why it was selected for the Streamlit prediction page.
            """,
        ),
        (
            "ROC-AUC Comparison",
            "roc_curve_comparison.png",
            "How Well The Models Separate Outcomes",
            """
            The ROC comparison shows that all models separate failed and successful chases better than random guessing.
            XGBoost has the strongest curve, so it ranks chase situations most reliably.
            """,
        ),
        (
            "Feature Importance",
            "feature_importance.png",
            "What The Model Relies On",
            """
            Random Forest feature importance shows that required run rate, pressure index, target, wickets lost, wickets left, and runs remaining dominate prediction.

            Team, venue, and toss effects matter less than the live match situation.
            """,
        ),
        (
            "SHAP Explainability",
            "summary_plot.png",
            "How Features Move The Prediction",
            """
            SHAP adds direction, not just importance.
            High required run rate, high pressure index, large targets, and wicket loss push predictions toward failed chases.

            More wickets left and manageable pressure push predictions toward successful chases.
            """,
        ),
    ]

    for section in ml_sections:
        insight_section(*section)

    st.markdown("---")
    st.markdown(
        """
        <div class="footer-box">
            <h2>Final Machine Learning Takeaway</h2>
            The updated model results tell a simple story: Logistic Regression shows that the features are useful,
            Random Forest shows which match factors matter most, tuning improves the tree model,
            and XGBoost gives the strongest final prediction.
            <br><br>
            Across all models, the same cricket logic keeps appearing:
            required run rate, pressure index, wickets, target size, and resource control are the core drivers of chase success.
        </div>
        """,
        unsafe_allow_html=True,
    )


def chase_state_inputs():
    teams = team_options("chasing_team")
    defending_teams = team_options("defending_team")
    venues = options_for("venue", FALLBACK_VENUES)
    batter_choices = player_display_options("batter_category", FALLBACK_BATTERS)
    bowler_choices = player_display_options("bowler_category", FALLBACK_BOWLERS)
    toss_decisions = options_for("toss_decision", ["bat", "field"])

    st.markdown(
        """
        <div class="section-container">
            <h1>Live Chase Inputs</h1>
            <p>Enter the current chase situation. The app calculates run rates, pressure index, wickets left, balls remaining, and runs remaining for the model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        target = st.number_input("Target", min_value=1, max_value=350, value=181, step=1)
        current_score = st.number_input("Current Score", min_value=0, max_value=350, value=92, step=1)
    with col2:
        completed_overs = st.number_input("Completed Overs", min_value=0, max_value=19, value=10, step=1)
        balls_in_current_over = st.number_input("Balls In Current Over", min_value=0, max_value=5, value=0, step=1)
    with col3:
        innings2_wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, value=3, step=1)
        won_toss_label = st.selectbox("Did Chasing Team Win Toss?", ["No", "Yes"])

    balls_bowled = int(completed_overs * 6 + balls_in_current_over)
    balls_bowled = max(0, min(120, balls_bowled))
    balls_remaining = max(0, 120 - balls_bowled)
    runs_remaining = max(0, target - current_score)
    wickets_left = max(0, 10 - innings2_wickets)
    current_rr = (current_score * 6 / balls_bowled) if balls_bowled > 0 else 0.0
    required_rr = (runs_remaining * 6 / balls_remaining) if balls_remaining > 0 else 99.0
    pressure_index = required_rr - current_rr
    won_toss = 1 if won_toss_label == "Yes" else 0

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        state_metric("Runs Remaining", runs_remaining)
    with m2:
        state_metric("Balls Remaining", balls_remaining)
    with m3:
        state_metric("Current RR", f"{current_rr:.2f}")
    with m4:
        state_metric("Required RR", f"{required_rr:.2f}")

    st.markdown(
        """
        <div class="section-container">
            <h1>Team, Venue & Player Context</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col4, col5 = st.columns(2)
    with col4:
        chasing_team = st.selectbox("Chasing Team", teams)
        venue = st.selectbox("Venue", venues)
        selected_batter = st.selectbox("Current Batter", batter_choices)
    with col5:
        valid_defenders = [team for team in defending_teams if team != chasing_team]
        defending_team = st.selectbox("Defending Team", valid_defenders or defending_teams)
        toss_decision = st.selectbox("Toss Decision", toss_decisions)
        selected_bowler = st.selectbox("Current Bowler", bowler_choices)

    batter_category = model_player_category(selected_batter, "batter_category")
    bowler_category = model_player_category(selected_bowler, "bowler_category")

    if selected_batter != batter_category or selected_bowler != bowler_category:
        st.caption(
            "Player note: names outside the notebook's trained star-player list are treated as 'Others' by the model."
        )

    input_df = pd.DataFrame(
        [
            {
                "target": target,
                "current_score": current_score,
                "runs_remaining": runs_remaining,
                "balls_remaining": balls_remaining,
                "innings2_wickets": innings2_wickets,
                "wickets_left": wickets_left,
                "current_rr": current_rr,
                "required_rr": required_rr,
                "pressure_index": pressure_index,
                "won_toss": won_toss,
                "chasing_team": chasing_team,
                "defending_team": defending_team,
                "venue": venue,
                "toss_decision": toss_decision,
                "batter_category": batter_category,
                "bowler_category": bowler_category,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    state = {
        "target": target,
        "current_score": current_score,
        "runs_remaining": runs_remaining,
        "balls_remaining": balls_remaining,
        "innings2_wickets": innings2_wickets,
        "wickets_left": wickets_left,
        "current_rr": current_rr,
        "required_rr": required_rr,
        "pressure_index": pressure_index,
        "won_toss": won_toss,
        "selected_batter": selected_batter,
        "selected_bowler": selected_bowler,
        "batter_category": batter_category,
        "bowler_category": bowler_category,
    }
    return input_df, state


def state_metric(label, value):
    st.markdown(
        f"""
        <div class="state-card">
            <h3>{label}</h3>
            <h2>{value}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_band(probability):
    if probability >= 0.70:
        return "Strong Chase Position"
    if probability >= 0.55:
        return "Slight Chase Advantage"
    if probability >= 0.40:
        return "Unstable / Balanced Chase"
    return "High-Risk Chase"


def result_summary(probability, state):
    pressure = "manageable" if state["pressure_index"] <= 2 else "rising"
    wickets = "healthy" if state["wickets_left"] >= 6 else "thin"
    return (
        f"Pressure is {pressure}, wickets are {wickets}, "
        f"and the required rate is {state['required_rr']:.2f}."
    )


def tactical_insights(state, probability):
    insights = []

    if state["required_rr"] >= 13:
        insights.append("Required run rate is very high, so the chase is entering a high-risk acceleration zone.")
    elif state["required_rr"] <= 8:
        insights.append("Required run rate is manageable, which gives the batting side tactical control.")

    if state["pressure_index"] > 4:
        insights.append("Pressure index is high because required rate is well above current scoring rate.")
    elif state["pressure_index"] < 0:
        insights.append("Current run rate is ahead of the required rate, reducing scoreboard pressure.")

    if state["wickets_left"] >= 7:
        insights.append("Strong wicket preservation gives the chasing team freedom to attack later.")
    elif state["wickets_left"] <= 4:
        insights.append("Low wickets remaining reduces risk-taking ability and increases collapse danger.")

    if state["balls_remaining"] <= 24 and state["runs_remaining"] > 45:
        insights.append("The final phase requires heavy acceleration, making death-over execution critical.")
    elif state["balls_remaining"] >= 48 and state["runs_remaining"] <= 70:
        insights.append("Enough balls remain for controlled pacing instead of panic hitting.")

    if state["target"] >= 200:
        insights.append("The target is above 200, which historically creates heavy scoreboard pressure.")
    elif state["target"] < 160:
        insights.append("The target is below 160, giving the chasing side more room for a stable chase.")

    if probability >= 0.70:
        insights.append("Model confidence is strong because the current chase state aligns with successful chase patterns.")
    elif probability < 0.40:
        insights.append("Model confidence leans toward failure because pressure and resource indicators are unfavorable.")

    return insights


def render_predictor():
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title">IPL Chase Predictor</div>
            <div class="hero-sub">
            Predict a live IPL chase using the updated XGBoost model and the same feature setup used in the notebook:
            required run rate, pressure index, target, wickets, balls remaining, teams, venue, toss, and player categories.
            </div>
        </div>
        <div class="predictor-strip">
            <div class="mini-chip"><strong>Live State</strong>Score, overs, wickets, and target drive the prediction.</div>
            <div class="mini-chip"><strong>Pressure</strong>The app calculates required rate and pressure index automatically.</div>
            <div class="mini-chip"><strong>Model</strong>XGBoost reads the chase situation and returns success probability.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if classifier_name != "XGBClassifier":
        st.warning(
            f"The loaded model is {classifier_name}. Your notebook says the final Streamlit model should be XGBoost. "
            "Run the notebook cell: joblib.dump(xgb_model, 'ipl_chase_model.pkl') if you want the deployed model to be XGBoost."
        )

    input_df, state = chase_state_inputs()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Analyze Tactical Match Outcome", use_container_width=True):
        with st.spinner("Analyzing live chase pressure..."):
            time.sleep(0.8)
            prediction = int(model.predict(input_df)[0])
            probability = float(model.predict_proba(input_df)[0][1])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-card">
                <h1>Chase Success Probability</h1>
                <div class="probability">{probability * 100:.2f}%</div>
                <h2>{result_band(probability)}</h2>
                <p style="font-size:19px; color:#dbeafe; margin-top:14px;">
                {result_summary(probability, state)}
                </p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{probability * 100:.2f}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(int(probability * 100))

        if prediction == 1:
            st.success("Successful chase predicted.")
        else:
            st.error("Failed chase predicted.")

        st.subheader("Match Situation Notes")
        for point in tactical_insights(state, probability):
            st.markdown(
                f"""
                <div class="insight-card">
                {point}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Show model input row"):
            st.dataframe(input_df, use_container_width=True)


def render_about():
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title">About The Project</div>
            <div class="hero-sub">
            A cricket analytics project built with EDA, machine learning,
            model comparison, SHAP, and a live IPL chase predictor.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.markdown(
            """
            ## Why This Project Exists

            IPL cricket looks chaotic, but the analysis shows that successful chases often follow measurable patterns.

            This project studies:

            - toss decisions and chase strategy
            - phase-wise scoring behavior
            - pressure index and required run rate
            - wicket preservation
            - model comparison and feature importance
            - SHAP explainability
            - live prediction through Streamlit

            The final deployed model is XGBoost because it achieved the best notebook performance:
            **90.35% accuracy** and **0.970 AUC**.
            """
        )

    with col2:
        card(
            "Technologies Used",
            """
            Python
            Streamlit
            Pandas
            Scikit-learn
            XGBoost
            SHAP
            Matplotlib
            Seaborn
            Joblib
            """,
        )


render_css()
page = sidebar()

if page == "Home":
    render_home()
elif page == "EDA Analysis":
    render_eda()
elif page == "ML Analysis":
    render_ml()
elif page == "Predictor":
    render_predictor()
elif page == "About":
    render_about()
