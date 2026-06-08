from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from auth import authenticate_user, create_access_token, decode_access_token, register_user
from data_analysis.sql_insights import keyword_signal_table, product_risk_table, satisfaction_summary
from database import get_recent_runs, init_db, record_analysis_run
from deployment.app_config import APP_NAME, REQUIRED_REVIEW_COLUMNS
from feature_selection.selector import top_discriminative_terms
from feedback_analyzer import extract_feedback_themes, generate_recommendations, get_top_keywords_by_sentiment
from google_oauth import complete_google_login, get_google_login_url, is_google_oauth_configured
from model_evaluation.evaluator import evaluate_model
from model_tuning.tuning import train_tuned_classifier
from preprocessing.text_cleaning import load_dataset, normalize_review_dataset, save_processed_dataset
from styles import apply_theme
from visualization.charts import category_satisfaction_bar, confusion_heatmap, sentiment_donut, trend_line


st.set_page_config(page_title=APP_NAME, page_icon="cart", layout="wide", initial_sidebar_state="expanded")


def initialize_state() -> None:
    init_db()
    st.session_state.setdefault("jwt", None)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("last_recorded_signature", None)
    st.session_state.setdefault("theme_mode", "Default")


def hydrate_user_from_token() -> None:
    if st.session_state.user or not st.session_state.jwt:
        return
    payload = decode_access_token(st.session_state.jwt)
    if payload:
        st.session_state.user = {
            "id": payload["sub"],
            "name": payload.get("name", "User"),
            "email": payload.get("email", ""),
            "provider": payload.get("provider", "local"),
        }


def set_logged_in(user: dict) -> None:
    st.session_state.jwt = create_access_token(user)
    st.session_state.user = user
    st.query_params.clear()
    st.rerun()


def theme_switcher(location: str) -> None:
    _, default_col, light_col, dark_col = st.columns([1, 0.055, 0.055, 0.055])
    with default_col:
        if st.button("⚙", key=f"default_theme_{location}", help="Default mode"):
            st.session_state.theme_mode = "Default"
            st.rerun()
    with light_col:
        if st.button("☀", key=f"light_theme_{location}", help="Light mode"):
            st.session_state.theme_mode = "Light"
            st.rerun()
    with dark_col:
        if st.button("☾", key=f"dark_theme_{location}", help="Dark mode"):
            st.session_state.theme_mode = "Dark"
            st.rerun()
    st.markdown(f'<div class="theme-status">{st.session_state.theme_mode} mode</div>', unsafe_allow_html=True)


def auth_page() -> None:
    google_user = complete_google_login()
    if google_user:
        set_logged_in(google_user)

    apply_theme(st.session_state.theme_mode)
    theme_switcher("auth")

    st.markdown(
        """
        <div class="auth-shell">
            <div class="auth-header">
                <h1>E-commerce Product Review Analyzer</h1>
                <p>Securely sign in to classify review sentiment, query insights with SQL, and visualize product satisfaction metrics.</p>
                <div class="google-badge"><span class="google-mark">G</span><span>Gmail OAuth supported on this page</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, register_tab, gmail_tab = st.tabs(["Login", "Register", "Continue with Gmail"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="analyst@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            user = authenticate_user(email, password)
            if user:
                set_logged_in(user)
            else:
                st.error("Invalid email or password.")

    with register_tab:
        with st.form("register_form"):
            name = st.text_input("Full name")
            reg_email = st.text_input("Work email")
            reg_password = st.text_input("Create password", type="password", placeholder="4 to 7 characters")
            created = st.form_submit_button("Create account", use_container_width=True)
        if created:
            try:
                set_logged_in(register_user(name, reg_email, reg_password))
            except ValueError as exc:
                st.error(str(exc))

    with gmail_tab:
        st.markdown('<div class="google-badge"><span class="google-mark">G</span><span>Sign in with your Gmail account</span></div>', unsafe_allow_html=True)
        if is_google_oauth_configured():
            st.link_button("Continue with Google", get_google_login_url(), use_container_width=True)
        else:
            st.info("Google OAuth is ready in the code. Add Google client settings in `.env` to activate the Gmail login button.")
            st.code(
                "GOOGLE_CLIENT_ID=your-client-id\nGOOGLE_CLIENT_SECRET=your-client-secret\nGOOGLE_REDIRECT_URI=http://localhost:8501",
                language="bash",
            )


def sidebar() -> None:
    user = st.session_state.user
    with st.sidebar:
        st.subheader(user["name"])
        st.caption(user["email"])
        st.divider()
        st.write("Required upload format")
        st.caption(REQUIRED_REVIEW_COLUMNS["minimum"])
        st.caption(REQUIRED_REVIEW_COLUMNS["recommended"])
        st.divider()
        recent_runs = get_recent_runs(user["id"])
        if recent_runs:
            st.write("Recent analysis runs")
            st.dataframe(pd.DataFrame(recent_runs), use_container_width=True, hide_index=True)
        if st.button("Log out", use_container_width=True):
            st.session_state.jwt = None
            st.session_state.user = None
            st.rerun()


def add_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, object, dict]:
    model, tuning_info = train_tuned_classifier(df)
    analyzed = df.copy()
    analyzed["predicted_sentiment"] = model.predict(analyzed["clean_review"])
    analyzed["confidence"] = model.predict_proba(analyzed["clean_review"]).max(axis=1) if hasattr(model, "predict_proba") else 0
    return analyzed, model, tuning_info


def record_run_once(file_name: str, analyzed: pd.DataFrame) -> None:
    signature = f"{file_name}:{len(analyzed)}:{round(float(analyzed['rating'].mean()), 4)}"
    if st.session_state.last_recorded_signature == signature:
        return
    positive_rate = float((analyzed["predicted_sentiment"] == "Positive").mean() * 100)
    record_analysis_run(
        st.session_state.user["id"],
        file_name,
        len(analyzed),
        positive_rate,
        float(analyzed["rating"].mean()),
    )
    st.session_state.last_recorded_signature = signature


def analyzer_page() -> None:
    sidebar()
    apply_theme(st.session_state.theme_mode)
    theme_switcher("dashboard")
    st.title(APP_NAME)
    st.caption("A professional Streamlit analytics workspace for SQL-based review insights, text classification, feature selection, model evaluation, and customer satisfaction visualization.")

    uploaded = st.file_uploader("Upload customer reviews CSV or Excel", type=["csv", "xlsx", "xls"])
    try:
        raw_df = load_dataset(uploaded)
        file_name = uploaded.name if uploaded else "sample_reviews.csv"
        normalized = normalize_review_dataset(raw_df)
    except Exception as exc:
        st.error(str(exc))
        return

    if normalized.empty:
        st.warning("The dataset has no valid review text after preprocessing.")
        return

    analyzed, model, tuning_info = add_predictions(normalized)
    processed_path = save_processed_dataset(analyzed)
    record_run_once(file_name, analyzed)

    positive_rate = (analyzed["predicted_sentiment"] == "Positive").mean() * 100
    negative_rate = (analyzed["predicted_sentiment"] == "Negative").mean() * 100
    avg_rating = analyzed["rating"].mean()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Reviews analyzed", f"{len(analyzed):,}")
    metric_cols[1].metric("Average rating", f"{avg_rating:.2f} / 5")
    metric_cols[2].metric("Positive sentiment", f"{positive_rate:.1f}%")
    metric_cols[3].metric("Negative sentiment", f"{negative_rate:.1f}%")

    tabs = st.tabs(["Overview", "SQL Insights", "Classification", "Feature Selection", "Customer Feedback", "Data Preview"])

    with tabs[0]:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Sentiment mix")
            st.plotly_chart(sentiment_donut(analyzed), use_container_width=True)
        with right:
            st.subheader("Category satisfaction")
            st.plotly_chart(category_satisfaction_bar(analyzed), use_container_width=True)
        st.subheader("Rating trend")
        st.plotly_chart(trend_line(analyzed), use_container_width=True)

    with tabs[1]:
        st.subheader("SQL-generated customer satisfaction summary")
        st.dataframe(satisfaction_summary(analyzed), use_container_width=True, hide_index=True)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.write("Products needing attention")
            st.dataframe(product_risk_table(analyzed), use_container_width=True, hide_index=True)
        with col_b:
            st.write("Sentiment behavior")
            st.dataframe(keyword_signal_table(analyzed), use_container_width=True, hide_index=True)

        with st.expander("Run a custom SQL query against the reviews table"):
            default_query = "SELECT product_name, category, rating, predicted_sentiment FROM reviews LIMIT 10"
            query = st.text_area("SQL query", default_query, height=120)
            if st.button("Run SQL"):
                try:
                    from data_analysis.sql_insights import run_query

                    st.dataframe(run_query(analyzed, query), use_container_width=True)
                except Exception as exc:
                    st.error(f"SQL error: {exc}")

    with tabs[2]:
        st.subheader("Text classification model")
        st.write("Target label is derived from rating: 4-5 Positive, 3 Neutral, 1-2 Negative.")
        eval_result = evaluate_model(model, analyzed)
        cols = st.columns(2)
        cols[0].metric("Evaluation accuracy", f"{eval_result.accuracy:.2%}")
        cols[1].metric("Tuning mode", tuning_info["mode"].replace("_", " ").title())
        if tuning_info["best_params"]:
            st.json(tuning_info["best_params"])
        if tuning_info["cv_score"] is not None:
            st.info(f"Best cross-validation score: {tuning_info['cv_score']:.2%}")
        st.plotly_chart(confusion_heatmap(eval_result.confusion, eval_result.labels), use_container_width=True)
        st.dataframe(eval_result.report, use_container_width=True)

    with tabs[3]:
        st.subheader("Important terms selected with TF-IDF and chi-square")
        try:
            terms = top_discriminative_terms(analyzed["clean_review"], analyzed["rating_sentiment"])
            st.dataframe(terms, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Feature selection needs more text variety: {exc}")

    with tabs[4]:
        st.subheader("Customer Feedback Themes & Insights")
        st.markdown("Automatically extracted themes from customer reviews to identify what customers praise and what they complain about.")
        
        # Extract themes and generate recommendations
        themes = extract_feedback_themes(analyzed, n_themes=4)
        recommendations = generate_recommendations(analyzed, themes)
        keywords = get_top_keywords_by_sentiment(analyzed, top_n=10)
        
        # Display recommendations with color coding
        st.subheader("Action Items")
        if recommendations:
            for rec in recommendations:
                color_map = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Info": "🔵"}
                priority_icon = color_map.get(rec["priority"], "⚪")
                with st.expander(f"{priority_icon} {rec['action']} ({rec['priority']})"):
                    st.write(rec["details"])
        
        # Display themes by sentiment
        col_pos, col_neg, col_neu = st.columns(3)
        
        with col_pos:
            st.markdown("### 🟢 What Customers Love")
            for theme in themes.get("positive", []):
                if "count" in theme:
                    st.markdown(f"**{theme.get('theme', 'N/A')}**")
                    st.caption(f"{theme['count']} reviews ({theme.get('percentage', 0):.0f}%)")
                    if theme.get("keywords"):
                        st.caption(f"Keywords: {', '.join(theme['keywords'][:3])}")
        
        with col_neg:
            st.markdown("### 🔴 What Needs Improvement")
            for theme in themes.get("negative", []):
                if "count" in theme:
                    st.markdown(f"**{theme.get('theme', 'N/A')}**")
                    st.caption(f"{theme['count']} reviews ({theme.get('percentage', 0):.0f}%)")
                    if theme.get("keywords"):
                        st.caption(f"Keywords: {', '.join(theme['keywords'][:3])}")
        
        with col_neu:
            st.markdown("### 🟡 Neutral Feedback")
            for theme in themes.get("neutral", []):
                if "count" in theme:
                    st.markdown(f"**{theme.get('theme', 'N/A')}**")
                    st.caption(f"{theme['count']} reviews ({theme.get('percentage', 0):.0f}%)")
                    if theme.get("keywords"):
                        st.caption(f"Keywords: {', '.join(theme['keywords'][:3])}")
        
        # Top keywords
        st.subheader("Top Keywords by Sentiment")
        kw_col1, kw_col2, kw_col3 = st.columns(3)
        with kw_col1:
            st.markdown("**Positive Words**")
            for kw in keywords.get("positive", []):
                st.caption(f"{kw['word']} ({kw['importance']})")
        with kw_col2:
            st.markdown("**Negative Words**")
            for kw in keywords.get("negative", []):
                st.caption(f"{kw['word']} ({kw['importance']})")
        with kw_col3:
            st.markdown("**Neutral Words**")
            for kw in keywords.get("neutral", []):
                st.caption(f"{kw['word']} ({kw['importance']})")

    with tabs[5]:
        st.subheader("Processed review data")
        st.caption(f"Processed dataset saved to {Path(processed_path).as_posix()}")
        st.dataframe(
            analyzed[
                [
                    "product_name",
                    "category",
                    "rating",
                    "predicted_sentiment",
                    "confidence",
                    "review_date",
                    "review_text",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download analyzed reviews",
            analyzed.to_csv(index=False).encode("utf-8"),
            file_name="analyzed_product_reviews.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    initialize_state()
    hydrate_user_from_token()
    apply_theme(st.session_state.theme_mode)

    if st.session_state.user:
        analyzer_page()
    else:
        auth_page()


if __name__ == "__main__":
    main()
