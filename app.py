import pandas as pd
import streamlit as st

from model import train_model


st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="SP",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #101828;
        --muted: #475467;
        --line: #d0d5dd;
        --brand: #3457d5;
        --accent: #00a76f;
        --danger: #d92d20;
        --panel: #ffffff;
        --page: #f4f7fb;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(52, 87, 213, 0.18), transparent 34rem),
            radial-gradient(circle at 82% 12%, rgba(0, 167, 111, 0.18), transparent 28rem),
            var(--page);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, p, label,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
    }

    .stSlider label,
    .stNumberInput label {
        color: var(--ink) !important;
        font-weight: 700 !important;
    }

    .stSlider [data-baseweb="slider"] div {
        color: var(--brand);
    }

    .hero {
        align-items: center;
        background: linear-gradient(135deg, #ffffff 0%, #eef4ff 58%, #e9fff6 100%);
        border: 1px solid rgba(16, 24, 40, 0.08);
        border-radius: 8px;
        box-shadow: 0 22px 55px rgba(16, 24, 40, 0.12);
        display: grid;
        gap: 1.5rem;
        grid-template-columns: minmax(0, 1.4fr) minmax(240px, 0.6fr);
        margin-bottom: 1.5rem;
        overflow: hidden;
        padding: 2rem;
        position: relative;
    }

    .hero-copy {
        position: relative;
        z-index: 2;
    }

    .eyebrow {
        color: #175cd3 !important;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #0b1220 !important;
        font-size: clamp(2.15rem, 5vw, 4.3rem);
        line-height: 1.02;
        margin: 0;
    }

    .hero p {
        color: #344054 !important;
        font-size: 1.07rem;
        line-height: 1.65;
        margin: 1rem 0 0;
        max-width: 700px;
    }

    .hero-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1.25rem;
    }

    .hero-pill {
        background: rgba(255, 255, 255, 0.76);
        border: 1px solid rgba(16, 24, 40, 0.1);
        border-radius: 999px;
        color: #182230;
        font-size: 0.92rem;
        font-weight: 700;
        padding: 0.55rem 0.85rem;
    }

    .scene {
        height: 250px;
        perspective: 900px;
        position: relative;
    }

    .orbital {
        animation: floaty 4.8s ease-in-out infinite;
        height: 210px;
        left: 50%;
        position: absolute;
        top: 50%;
        transform: translate(-50%, -50%);
        transform-style: preserve-3d;
        width: 210px;
    }

    .cube {
        animation: spin 9s linear infinite;
        height: 128px;
        left: 41px;
        position: absolute;
        top: 41px;
        transform-style: preserve-3d;
        width: 128px;
    }

    .face {
        align-items: center;
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(52, 87, 213, 0.34);
        box-shadow: inset 0 0 28px rgba(52, 87, 213, 0.14);
        color: #175cd3;
        display: flex;
        font-size: 2rem;
        font-weight: 900;
        height: 128px;
        justify-content: center;
        position: absolute;
        width: 128px;
    }

    .front { transform: rotateY(0deg) translateZ(64px); }
    .back { transform: rotateY(180deg) translateZ(64px); }
    .right { transform: rotateY(90deg) translateZ(64px); }
    .left { transform: rotateY(-90deg) translateZ(64px); }
    .top { transform: rotateX(90deg) translateZ(64px); }
    .bottom { transform: rotateX(-90deg) translateZ(64px); }

    .ring {
        animation: pulse 2.8s ease-in-out infinite;
        border: 2px solid rgba(0, 167, 111, 0.34);
        border-radius: 999px;
        height: 190px;
        left: 10px;
        position: absolute;
        top: 10px;
        transform: rotateX(72deg) rotateZ(-20deg);
        width: 190px;
    }

    .dot {
        animation: orbit 4.2s linear infinite;
        background: #00a76f;
        border-radius: 999px;
        box-shadow: 0 0 22px rgba(0, 167, 111, 0.55);
        height: 15px;
        left: 97px;
        position: absolute;
        top: 0;
        transform-origin: 8px 105px;
        width: 15px;
    }

    .section-card {
        background: var(--panel);
        border: 1px solid rgba(16, 24, 40, 0.1);
        border-radius: 8px;
        box-shadow: 0 18px 40px rgba(16, 24, 40, 0.08);
        margin-bottom: 0.85rem;
        padding: 1rem 1.1rem;
    }

    .section-title {
        color: var(--ink) !important;
        font-size: 1.35rem;
        font-weight: 850;
        margin: 0;
    }

    .section-note {
        color: var(--muted) !important;
        margin: 0.2rem 0 0;
    }

    .metric-row {
        display: grid;
        gap: 0.75rem;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        margin: 0.35rem 0 0.9rem;
    }

    .mini-metric {
        background: #f8fafc;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.9rem;
    }

    .mini-label {
        color: #667085;
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
    }

    .mini-value {
        color: #101828;
        font-size: 1.65rem;
        font-weight: 900;
        line-height: 1.2;
        margin-top: 0.2rem;
    }

    .result-good,
    .result-poor,
    .result-idle {
        border-radius: 8px;
        margin-top: 0.9rem;
        padding: 1.25rem;
    }

    .result-good {
        background: #ecfdf3;
        border: 1px solid #75e0a7;
        color: #05603a;
    }

    .result-poor {
        background: #fff4ed;
        border: 1px solid #fdb022;
        color: #7a2e0e;
    }

    .result-idle {
        background: #eff4ff;
        border: 1px solid #84adff;
        color: #1849a9;
    }

    .result-label {
        font-size: 0.82rem;
        font-weight: 850;
        text-transform: uppercase;
    }

    .result-value {
        font-size: clamp(1.8rem, 3vw, 2.35rem);
        font-weight: 950;
        line-height: 1.1;
        margin-top: 0.25rem;
    }

    .tip {
        background: #ffffff;
        border-left: 4px solid var(--brand);
        border-radius: 8px;
        color: #344054 !important;
        margin-top: 0.85rem;
        padding: 0.9rem 1rem;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #3457d5, #00a76f);
        border: 0;
        border-radius: 8px;
        color: #ffffff;
        font-weight: 850;
        min-height: 3rem;
        width: 100%;
    }

    div.stButton > button:hover {
        border: 0;
        color: #ffffff;
        filter: brightness(0.96);
    }

    @keyframes spin {
        from { transform: rotateX(-18deg) rotateY(0deg); }
        to { transform: rotateX(-18deg) rotateY(360deg); }
    }

    @keyframes floaty {
        0%, 100% { transform: translate(-50%, -50%) translateY(0); }
        50% { transform: translate(-50%, -50%) translateY(-14px); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 0.55; transform: rotateX(72deg) rotateZ(-20deg) scale(0.94); }
        50% { opacity: 1; transform: rotateX(72deg) rotateZ(-20deg) scale(1.03); }
    }

    @keyframes orbit {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    @media (max-width: 820px) {
        .hero {
            grid-template-columns: 1fr;
            padding: 1.35rem;
        }

        .scene {
            height: 190px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training prediction model...")
def get_model():
    return train_model()


model = get_model()

st.markdown(
    """
    <section class="hero">
        <div class="hero-copy">
            <div class="eyebrow">AI Student Analytics</div>
            <h1>Student Performance Predictor</h1>
            <p>
                A cleaner, high-contrast dashboard for checking student scores,
                balance factors, and performance risk in one quick view.
            </p>
            <div class="hero-stats">
                <div class="hero-pill">Academic scores</div>
                <div class="hero-pill">Sleep and stress</div>
                <div class="hero-pill">Instant prediction</div>
            </div>
        </div>
        <div class="scene" aria-hidden="true">
            <div class="orbital">
                <div class="ring"></div>
                <div class="dot"></div>
                <div class="cube">
                    <div class="face front">AI</div>
                    <div class="face back">ML</div>
                    <div class="face right">%</div>
                    <div class="face left">SP</div>
                    <div class="face top">A+</div>
                    <div class="face bottom">OK</div>
                </div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

input_col, result_col = st.columns([1.3, 1], gap="large")

with input_col:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Student Details</div>
            <p class="section-note">Move the sliders and enter current lifestyle values.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_col_1, score_col_2, score_col_3 = st.columns(3)
    with score_col_1:
        math = st.slider("Math Score", 0, 100, 70)
    with score_col_2:
        reading = st.slider("Reading Score", 0, 100, 72)
    with score_col_3:
        writing = st.slider("Writing Score", 0, 100, 74)

    st.divider()

    lifestyle_col_1, lifestyle_col_2, lifestyle_col_3 = st.columns(3)
    with lifestyle_col_1:
        sleep = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
    with lifestyle_col_2:
        stress = st.slider("Stress Level", 1, 10, 5)
    with lifestyle_col_3:
        social = st.slider("Social Media Usage", 0, 10, 3)

    predict = st.button("Predict Performance", type="primary")

average = (math + reading + writing) / 3

with result_col:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Prediction Summary</div>
            <p class="section-note">The model combines score and lifestyle signals.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="mini-metric">
                <div class="mini-label">Average Score</div>
                <div class="mini-value">{average:.1f}</div>
            </div>
            <div class="mini-metric">
                <div class="mini-label">Sleep</div>
                <div class="mini-value">{sleep}h</div>
            </div>
        </div>
        <div class="tip">
            Stress level <strong>{stress}/10</strong> and social media usage
            <strong>{social}/10</strong> are included in this prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if predict:
        new_data = pd.DataFrame(
            [[math, reading, writing, average, stress, sleep, social]],
            columns=[
                "math score",
                "reading score",
                "writing score",
                "average_score",
                "stress_level",
                "sleep_hours",
                "social_media_usage",
            ],
        )

        prediction = model.predict(new_data)

        if prediction[0] == 1:
            st.markdown(
                """
                <div class="result-good">
                    <div class="result-label">Predicted Outcome</div>
                    <div class="result-value">Good Performance</div>
                </div>
                <div class="tip">
                    The current inputs suggest the student is on a strong path.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-poor">
                    <div class="result-label">Predicted Outcome</div>
                    <div class="result-value">Needs Support</div>
                </div>
                <div class="tip">
                    Try improving weak subject scores, reducing stress, or checking
                    sleep consistency before the next review.
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="result-idle">
                <div class="result-label">Waiting For Input</div>
                <div class="result-value">Ready to Predict</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
