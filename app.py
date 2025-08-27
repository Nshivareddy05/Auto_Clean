import streamlit as st
from clean_file import Clean_File
from io import BytesIO
import pandas as pd
import plotly.express as px

# --- Global vars ---
cleaned_data = None
report_ = None
column_names = None
col_wise_null = None
total_null = None
total_count = None
not_null = None
summary = None 
project_name = None

st.set_page_config(page_title="Data Cleaning App", layout="wide")
st.header("SmartSurvey")

# --- Sidebar navigation ---
page = st.sidebar.selectbox("Navigate", ["Upload & Clean", "Report Generation", "About / Creators"])

# --- Page 1: Upload & Clean ---
if page == "Upload & Clean":
    st.title("Upload & Clean Dataset")
    uploaded_file = st.file_uploader("Choose your CSV or Excel file", type=["csv", "xls", "xlsx"])
    project_name = st.text_input("Project Name")

    if uploaded_file and project_name:
        cf = Clean_File(uploaded_file)

        # --- Step 1: Missing value imputation ---
        cleaned_data, report_ = cf.smart_imputer()

        # --- Step 2: Outlier removal (IQR method) ---
        cleaned_data, outlier_report = cf.remove_outliers(method="IQR")
        report_.extend(outlier_report)

        # Save cleaned data to session
        st.session_state["cleaned_data"] = cleaned_data
        st.session_state["project_name"] = project_name

        # --- Success message ---
        st.success(f"✅ Dataset cleaned for project '{project_name}'")

        # --- Preview cleaned data ---
        st.subheader("Cleaned Data Preview")
        st.dataframe(cleaned_data.head())

        # --- Missing Values Visualization ---
        st.subheader("Missing Values (%) by Column")
        missing_perc = cleaned_data.isnull().mean() * 100
        missing_df = missing_perc.reset_index()
        missing_df.columns = ["Column", "Missing %"]
        fig = px.bar(missing_df, x="Column", y="Missing %", text="Missing %")
        st.plotly_chart(fig, use_container_width=True)

        # --- Statistical Summary ---
        st.subheader("Statistical Summary")
        st.write(cleaned_data.describe(include="all"))

        # --- Outlier Cleaning Report ---
        st.subheader("Outlier Cleaning Report")
        if outlier_report:
            for line in outlier_report:
                st.write("- " + line)
        else:
            st.write("No numeric outliers detected.")

        # --- Download Cleaned Data ---
        file_format = st.selectbox("Choose a Download Format : ", ["Csv", "Excel"]).lower()
        if file_format == "csv":
            csv_data = cleaned_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned Dataset as CSV",
                data=csv_data,
                file_name=f"{project_name}_cleaned.csv",
                mime="text/csv"
            )
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                cleaned_data.to_excel(writer, index=False, sheet_name="CleanedData")
            processed_data = output.getvalue()
            st.download_button(
                label="📥 Download Cleaned Dataset as Excel",
                data=processed_data,
                file_name=f"{project_name}_cleaned.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# --- Page 2: Report Generation (placeholder) ---
elif page == "Report Generation":
    st.title("Report Generation")
    st.write("📊 This section will generate detailed reports from cleaned data.")

# --- Page 3: About / Creators ---
elif page == "About / Creators":
    st.title("About / Creators")
    st.markdown("""
    - Project created by: Shiva  
    - Purpose: Data cleaning + report generation  
    - Technologies used: Streamlit, Sklearn, Ollama (Mistral)  
    - Features: Automatic imputation, outlier removal, detailed log report, AI-generated summary
    """)
