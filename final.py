import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

st.set_page_config(page_title="Miniproject 3 - Random Forest", layout="wide")

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
        "predicting potential YouTube video revenue becomes essential for business planning and content strategy. "
        "To achieve this, I have created this Streamlit application to show an estimation of individual "
        "YouTube video earnings using a high-performance Random Forest Regressor model."
    )
    st.image("youtube.jpg", caption="Jessica Sriramula")

if selected == "Revenue Estimation":
    st.title("Revenue Estimation.")
    st.subheader("Here we can explore and find out the estimated YouTube ad video revenue using Random Forest Model.")

    @st.cache_resource
    def train_rf_model():
        path = r"C:\Users\pc\Desktop\miniproject3\youtube_processed.csv.xlsx"
        df = pd.read_excel(path)

        df[["likes", "comments", "watch_time_minutes"]] = df[["likes", "comments", "watch_time_minutes"]].fillna(0)

        df_clean = df.drop(columns=["video_id", "date"], errors="ignore")

        unique_categories = sorted([col.replace("category_", "") for col in df_clean.columns if col.startswith("category_")])
        unique_devices = sorted([col.replace("device_", "") for col in df_clean.columns if col.startswith("device_")])
        unique_countries = sorted([col.replace("country_", "") for col in df_clean.columns if col.startswith("country_")])

        df_clean['engagement_rate'] = (df_clean['likes'] + df_clean['comments']) / (df_clean['views'] + 1e-5)
        df_clean['view_to_sub_ratio'] = df_clean['views'] / (df_clean['subscribers'] + 1e-5)
        df_clean['avg_view_duration_mins'] = df_clean['watch_time_minutes'] / (df_clean['views'] + 1e-5)
        df_clean['comment_to_like_ratio'] = df_clean['comments'] / (df_clean['likes'] + 1e-5)

        X = df_clean.drop(columns=["ad_revenue_usd"])
        y = df_clean["ad_revenue_usd"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        return model, X.columns.tolist(), r2, mae, rmse, unique_categories, unique_devices, unique_countries

    with st.spinner("Training Random Forest Model... Please wait..."):
        (rf_model, model_features, r2_accuracy, mae_val, rmse_val, categories, devices, countries) = train_rf_model()

    st.sidebar.header("Model Evaluation")
    st.sidebar.metric(label="Random Forest R² Accuracy", value=f"{r2_accuracy:.2%}")
    st.sidebar.metric(label="Mean Absolute Error (MAE)", value=f"${mae_val:,.2f}")
    st.sidebar.metric(label="Root Mean Squared Error (RMSE)", value=f"${rmse_val:,.2f}")

    st.subheader("Step 1: Please select your YouTube ad video features")
    col1, col2 = st.columns(2)
    with col1:
        views = st.number_input("Total Views", min_value=0, value=25000, step=1000, format="%d")
        likes = st.number_input("Total Likes", min_value=0, value=1200, step=100, format="%d")
        comments = st.number_input("Total Comments", min_value=0, value=150, step=10, format="%d")
    with col2:
        watch_time = st.number_input("Total Watch Time (minutes)", min_value=0.0, value=75000.0, step=500.0, format="%.1f")
        video_length = st.number_input("Video Duration (minutes)", min_value=0.0, value=12.5, step=0.5, format="%.1f")
        subscribers = st.number_input("Total Subscribers", min_value=0, value=15000, step=500, format="%d")

    st.subheader("Step 2: Please choose video, device, country and upload date context")
    col3, col4, col5 = st.columns(3)
    with col3:
        selected_category = st.selectbox("Video Category", categories)
        
        months_dict = {
            "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
            "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
        }
        selected_month_name = st.selectbox("Upload Month", list(months_dict.keys()), index=5)
        selected_month_num = months_dict[selected_month_name]
        
    with col4:
        selected_device = st.selectbox("Device", devices)
        
        days_dict = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
        }
        selected_day_name = st.selectbox("Upload Day of the Week", list(days_dict.keys()), index=2)
        selected_day_num = days_dict[selected_day_name]
        
    with col5:
        selected_country = st.selectbox("Country", countries)
        
        is_weekend_val = 1 if selected_day_num in [5, 6] else 0
        weekend_label = "Yes" if is_weekend_val == 1 else "No"
        st.text_input("Is it a Weekend?", value=weekend_label, disabled=True)

    if st.button("Estimated YouTube Ad Video Revenue", type="primary"):
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
            "month": selected_month_num,
            "day_of_week": selected_day_num,
            "is_weekend": is_weekend_val,
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
        
        raw_prediction = rf_model.predict(final_input_df)[0]
        calculated_revenue = max(0.0, raw_prediction)

        st.success(f"### Estimated YouTube Ad Video Revenue: **${calculated_revenue:,.2f}**")