import streamlit as st
import pandas as pd
import numpy as np
from modules import data_profiler, data_cleaner, code_generator

# Page config
st.set_page_config(
    page_title="DataPrep Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'cleaning_history' not in st.session_state:
    st.session_state.cleaning_history = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("# 📊 DataPrep")
    st.markdown("### AI-Powered Assistant")
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📁 Upload Data", "🔍 Profile Data", "🛠️ Clean Data", "📥 Export", "💾 Database"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if st.session_state.df is not None:
        st.success(f"✅ Dataset loaded")
        st.caption(f"Rows: {st.session_state.df.shape[0]:,}")
        st.caption(f"Columns: {st.session_state.df.shape[1]}")
    
    st.markdown("---")
    

# Main content routing
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">DataPrep Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent Data Preparation Made Simple")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🚀 Fast")
        st.write("Analyze datasets in seconds, not hours")
    
    with col2:
        st.markdown("#### 🤖 Smart")
        st.write("AI-powered recommendations for data cleaning")
    
    with col3:
        st.markdown("#### 📊 Visual")
        st.write("Beautiful charts and insights")
    
    st.markdown("---")
    
    st.markdown("### How It Works")
    
    steps = [
        ("1️⃣", "Upload", "Upload your CSV or Excel file"),
        ("2️⃣", "Profile", "Get instant data quality analysis"),
        ("3️⃣", "Clean", "Apply smart cleaning strategies"),
        ("4️⃣", "Export", "Download cleaned data + Python code")
    ]
    
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(desc)
    
    st.markdown("---")
    st.info("👈 **Get started by selecting 'Upload Data' from the sidebar**")

elif page == "📁 Upload Data":
    st.header("📁 Upload Your Dataset")
    
    # Multiple upload options
    upload_method = st.radio(
        "Choose upload method:",
        ["📄 File Upload", "🔗 From URL", "💾 From Database"],
        horizontal=True
    )
    
    if upload_method == "📄 File Upload":
        uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Supported formats: CSV, Excel, JSON"
    )
    
    if uploaded_file:
        try:
            # Detect file type and read with proper encoding handling
            if uploaded_file.name.endswith('.csv'):
                # Try multiple encodings for CSV files
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                        st.info("ℹ️ File loaded with Latin-1 encoding")
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='cp1252')
                        st.info("ℹ️ File loaded with Windows-1252 encoding")
                    except Exception:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
                        st.info("ℹ️ File loaded with ISO-8859-1 encoding")
                        
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            
            st.session_state.df = df
            st.success(f"✅ Loaded: {uploaded_file.name}")
            
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Size", f"{memory_mb:.2f} MB")
            with col4:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric("Missing %", f"{missing_pct:.1f}%")
            
            # Preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column info
            with st.expander("📊 Column Details"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count().values,
                    'Null': df.isnull().sum().values,
                    'Unique': df.nunique().values
                })
                st.dataframe(col_info, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("💡 Tip: Try saving your file with UTF-8 encoding or as Excel format")
    
    elif upload_method == "🔗 From URL":
        url = st.text_input("Enter CSV URL:")
        if url and st.button("Load"):
            try:
                df = pd.read_csv(url)
                st.session_state.df = df
                st.success("✅ Data loaded from URL")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    elif upload_method == "💾 From Database":
        st.info("💡 Coming soon: PostgreSQL, MySQL, SQLite support")
        st.code("""
# Example connection (you can implement this)
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:pass@localhost/dbname')
df = pd.read_sql_query('SELECT * FROM table', engine)
        """, language='python')

elif page == "🔍 Profile Data":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        st.stop()
    
    df = st.session_state.df
    
    st.header("🔍 Data Quality Profile")
    
    # Overall quality score
    quality_score = data_profiler.calculate_quality_score(df)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
            <h1 style='font-size: 4rem; margin: 0;'>{quality_score}</h1>
            <h3 style='margin: 0;'>Data Quality Score</h3>
            <p style='margin: 10px 0 0 0;'>{'🟢 Excellent' if quality_score >= 90 else '🟡 Good' if quality_score >= 70 else '🟠 Fair' if quality_score >= 50 else '🔴 Poor'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "❌ Missing Data", "📈 Outliers", "🔄 Duplicates"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Dataset Summary")
            summary = data_profiler.get_basic_summary(df)
            for key, value in summary.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("📈 Data Types Distribution")
            type_counts = df.dtypes.value_counts()
            st.bar_chart(type_counts)
    
    with tab2:
        missing_analysis = data_profiler.analyze_missing_data(df)
        
        if missing_analysis['total_missing'] > 0:
            st.subheader(f"Found {missing_analysis['total_missing']:,} missing values")
            
            # Missing data table
            st.dataframe(missing_analysis['by_column'], use_container_width=True)
            
            # Visualization
            import plotly.express as px
            fig = px.bar(
                missing_analysis['by_column'],
                x='Column',
                y='Missing %',
                title='Missing Data Distribution',
                color='Missing %',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            for col, strategy in missing_analysis['recommendations'].items():
                st.info(f"**{col}**: {strategy}")
        else:
            st.success("✅ No missing values detected!")
    
    with tab3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("Select column:", numeric_cols)
            
            outlier_info = data_profiler.detect_outliers(df, selected_col)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Outliers Found", outlier_info['count'])
            with col2:
                st.metric("Outlier %", f"{outlier_info['percentage']:.2f}%")
            with col3:
                st.metric("Method", "IQR (1.5x)")
            
            # Box plot
            import plotly.express as px
            fig = px.box(df, y=selected_col, title=f'Distribution: {selected_col}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found")
    
    with tab4:
        duplicates = df.duplicated().sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Duplicate Rows", duplicates)
        with col2:
            st.metric("Duplicate %", f"{(duplicates/len(df)*100):.2f}%")
        
        if duplicates > 0:
            if st.button("🔍 Show Duplicate Rows"):
                st.dataframe(df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist()))

elif page == "🛠️ Clean Data":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        st.stop()
    
    # Import and use the advanced cleaning interface
    from modules.advanced_cleaner import enhanced_cleaning_interface
    enhanced_cleaning_interface(st.session_state.df)

elif page == "📥 Export":
    if st.session_state.cleaned_df is None:
        st.warning("⚠️ Please clean your data first!")
        st.stop()
    
    cleaned_df = st.session_state.cleaned_df
    
    st.header("📥 Export Cleaned Data")
    
    # Summary
    st.subheader("📊 Summary")
    
    original_rows = len(st.session_state.df)
    cleaned_rows = len(cleaned_df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original", original_rows)
    with col2:
        st.metric("Cleaned", cleaned_rows)
    with col3:
        improvement = ((original_rows - cleaned_rows) / original_rows * 100)
        st.metric("Removed", f"{improvement:.1f}%")
    
    # Preview
    st.subheader("📋 Preview")
    st.dataframe(cleaned_df.head(20), use_container_width=True)
    
    # Download options
    st.subheader("💾 Download")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📄 CSV",
            csv,
            "cleaned_data.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        from io import BytesIO
        buffer = BytesIO()
        with pd.ExcelWriter(buffer) as writer:
            cleaned_df.to_excel(writer, index=False)
        st.download_button(
            "📊 Excel",
            buffer.getvalue(),
            "cleaned_data.xlsx",
            use_container_width=True
        )
    
    with col3:
        json_str = cleaned_df.to_json(orient='records', indent=2)
        st.download_button(
            "🔗 JSON",
            json_str,
            "cleaned_data.json",
            use_container_width=True
        )
    
    # Generated code
    st.subheader("🐍 Python Code")
    
    code = code_generator.generate_pipeline_code(
        st.session_state.df,
        cleaned_df,
        st.session_state.cleaning_history
    )
    
    st.code(code, language='python')
    
    st.download_button(
        "💾 Download Code",
        code,
        "cleaning_pipeline.py",
        "text/plain"
    )

elif page == "💾 Database":
    st.header("💾 Database Integration")
    st.info("🚧 Database features coming in v1.1")
    
    st.markdown("""
    ### Planned Features:
    - 📥 Import from PostgreSQL, MySQL, SQLite
    - 📤 Export cleaned data to database
    - 🔄 Schedule automated cleaning pipelines
    - 📊 Track data quality over time
    """)
    
    # Placeholder for database connection
    with st.expander("⚙️ Database Settings (Preview)"):
        db_type = st.selectbox("Database Type:", ["PostgreSQL", "MySQL", "SQLite"])
        host = st.text_input("Host:", "localhost")
        port = st.number_input("Port:", value=5432)
        database = st.text_input("Database:")
        username = st.text_input("Username:")
        password = st.text_input("Password:", type="password")
        
        if st.button("Test Connection"):
            st.warning("Feature under development")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("DataPrep Assistant v1.0")
with col2:
    st.caption("Made with Streamlit")
with col3:
    st.caption("© 2025")