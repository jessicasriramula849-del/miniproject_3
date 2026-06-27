import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Miniproject 3", layout="wide")

with st.sidebar:
    st.title("Navigation") 
    selected = option_menu(
        menu_title=None, 
        options=["Home", "Revenue Estimation"],
        icons=["house", "currency-exchange"], 
        default_index=0,
    )

if selected == "Home":
    st.title("Welcome to My Miniproject 3.")
    st.header('Content Monetization Modeler.')
    st.write(
        "As video creators and media companies increasingly rely on platforms like YouTube for income, "
        "predicting potential youtube video revenue becomes essential for business planning and content strategy. "
        "To achieve this, I have created this streamlit application to show an estimation of individual "
        "ysouTube video earnings using the Gradient Boosting Regressor model."
    )
    st.image("youtube.jpg", caption="Jessica Sriramula")

if selected == "Revenue Estimation":
    st.title("Revenue Estimation.")
    st.subheader("Here we can explore and find out the estimated youtube ad video revenue for each and every single youtube video.")

    @st.cache_resource
    def train_gradient_boosting_model():
        path = r"C:\Users\pc\Desktop\miniproject3\youtube_ad_revenue_dataset.csv.xlsx"
        df = pd.read_excel(path)

        columns_to_fill = ["likes", "comments", "watch_time_minutes"]
        df[columns_to_fill] = df[columns_to_fill].fillna(0)

        unique_categories = sorted(df["category"].dropna().unique().tolist())
        unique_devices = sorted(df["device"].dropna().unique().tolist())
        unique_countries = sorted(df["country"].dropna().unique().tolist())

        df["engagement_rate"] = (df["likes"] + df["comments"]) / (df["views"] + 1e-5)
        df["view_to_sub_ratio"] = df["views"] / (df["subscribers"] + 1e-5)
        df["avg_view_duration_mins"] = df["watch_time_minutes"] / (df["views"] + 1e-5)
        df["comment_to_like_ratio"] = df["comments"] / (df["likes"] + 1e-5)

        df_clean = df.drop(columns=["video_id", "date"], errors="ignore")

        categorical_cols = ["category", "device", "country"]
        df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True, dtype=int)

        y = df_encoded["ad_revenue_usd"]
        X = df_encoded.drop(columns=["ad_revenue_usd"])

        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        r2_score = model.score(X, y)

        return model, X.columns.tolist(), r2_score, unique_categories, unique_devices, unique_countries

    (gb_model, model_features, r2_accuracy, categories, devices, countries) = train_gradient_boosting_model()

    st.sidebar.header("Model Evaluation")
    st.sidebar.metric(label="Gradient Boosting R² Accuracy", value=f"{r2_accuracy:.2%}")

    st.subheader("Step 1: Please select your youtube ad video features")
    col1, col2 = st.columns(2)
    with col1:
        views = st.number_input("Total Views", min_value=0, value=25000, step=1000, format="%d")
        likes = st.number_input("Total Likes", min_value=0, value=1200, step=100, format="%d")
        comments = st.number_input("Total Comments", min_value=0, value=150, step=10, format="%d")
    with col2:
        watch_time = st.number_input("Total Watch Time (minutes)", min_value=0.0, value=75000.0, step=500.0, format="%.1f")
        video_length = st.number_input("Video Duration (minutes)", min_value=0.0, value=12.5, step=0.5, format="%.1f")
        subscribers = st.number_input("Total Subscribers", min_value=0, value=15000, step=500, format="%d")

    st.subheader("Step 2: Please choose video, device and country")
    col3, col4, col5 = st.columns(3)
    with col3:
        selected_category = st.selectbox("Video", categories)
    with col4:
        selected_device = st.selectbox("Device", devices)
    with col5:
        selected_country = st.selectbox("Country", countries)

    if st.button("Estimated Youtube Ad Video Revenue", type="primary"):
        input_engagement = (likes + comments) / (views + 1e-5)
        input_view_sub = views / (subscribers + 1e-5)
        input_avg_duration = watch_time / (views + 1e-5)
        input_comm_like = comments / (likes + 1e-5)

        live_prediction_data = {
            "views": views,
            "likes": likes,
            "comments": comments,
            "watch_time_minutes": watch_time,
            "video_length_minutes": video_length,
            "subscribers": subscribers,
            "engagement_rate": input_engagement,
            "view_to_sub_ratio": input_view_sub,
            "avg_view_duration_mins": input_avg_duration,
            "comment_to_like_ratio": input_comm_like,
        }

        for feature in model_features:
            if feature not in live_prediction_data:
                live_prediction_data[feature] = 0

        target_category_column = f"category_{selected_category}"
        target_device_column = f"device_{selected_device}"
        target_country_column = f"country_{selected_country}"

        if target_category_column in live_prediction_data:
            live_prediction_data[target_category_column] = 1
        if target_device_column in live_prediction_data:
            live_prediction_data[target_device_column] = 1
        if target_country_column in live_prediction_data:
            live_prediction_data[target_country_column] = 1

        final_input_df = pd.DataFrame([live_prediction_data])[model_features]
        raw_prediction = gb_model.predict(final_input_df)[0]
        calculated_revenue = max(0.0, raw_prediction)

        st.success(f"### Estimated Youtube Ad Video Revenue: **${calculated_revenue:,.2f}**")