import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="InsightFlow Pro", layout="wide")

st.title("📊 InsightFlow: AI Data Explorer & Preprocessor")
st.markdown("Upload, Clean, Engineer, and Visualize your data in one place.")

# --- 1. FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load original data
    df = pd.read_csv(uploaded_file)
    
    # --- 2. SIDEBAR PREPROCESSING TOOLS ---
    st.sidebar.header("🛠️ Preprocessing Tools")
    
    # Missing Value Handling
    if st.sidebar.button("Auto-Clean Missing Values"):
        # Numeric: Mean | Categorical: Unknown
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna("Unknown")
        st.sidebar.success("✅ Missing values handled!")

    # Outlier Detection & Fixing (IQR Method)
    if st.sidebar.checkbox("Detect & Cap Outliers"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Capping (Winsorizing)
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        st.sidebar.warning("⚠️ Outliers capped via IQR method")

    # Feature Engineering (Dates)
    date_col = st.sidebar.selectbox("Select Date Column (if any)", [None] + list(df.columns))
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df['Day_of_Week'] = df[date_col].dt.day_name()
            df['Month'] = df[date_col].dt.month_name()
            st.sidebar.info("📅 Date features extracted!")
        except:
            st.sidebar.error("Invalid date column")

    # Scaling
    if st.sidebar.checkbox("Scale Numeric Data"):
        scaler = StandardScaler()
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            df[num_cols] = scaler.fit_transform(df[num_cols])
            st.sidebar.success("⚖️ Scaling applied (StandardScaler)")

    # --- 3. DATA OVERVIEW ---
    st.subheader("1. Data Preview")
    st.dataframe(df.head(10)) 
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("2. Summary Statistics")
        st.write(df.describe()) 
        
    with col2:
        st.subheader("3. Data Quality Info")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Null Values": df.isnull().sum()
        })
        st.table(info_df)

    # --- 4. SMART VISUALIZATION ---
    st.divider()
    st.subheader("4. Interactive Chart Builder")
    
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    chart_col1, chart_col2 = st.columns([1, 3])
    
    with chart_col1:
        chart_type = st.selectbox("Select Chart Type", ["Bar", "Scatter", "Line", "Histogram", "Boxplot"])
        x_axis = st.selectbox("Select X Axis", options=df.columns)
        y_axis = st.selectbox("Select Y Axis", options=numeric_columns if numeric_columns else df.columns)
        color_tag = st.selectbox("Group by (Color)", options=[None] + list(df.columns))

    with chart_col2:
        if chart_type == "Bar":
            fig = px.bar(df, x=x_axis, y=y_axis, color=color_tag, template="plotly_dark")
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_tag, template="plotly_dark")
        elif chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis, color=color_tag, template="plotly_dark")
        elif chart_type == "Boxplot":
            fig = px.box(df, x=x_axis, y=y_axis, color=color_tag, template="plotly_dark")
        else:
            fig = px.histogram(df, x=x_axis, template="plotly_dark")
            
        st.plotly_chart(fig, use_container_width=True)

    # --- 5. DOWNLOAD CLEANED DATA ---
    st.divider()
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Processed Data", data=csv, file_name="processed_data.csv", mime="text/csv")

else:
    st.info("☝️ Please upload a CSV file to begin analysis.")