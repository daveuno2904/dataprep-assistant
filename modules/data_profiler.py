import pandas as pd
import numpy as np

def calculate_quality_score(df):
    """Calculate overall data quality score (0-100)"""
    scores = []
    
    # Completeness (no missing values)
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    completeness = max(0, 100 - missing_pct)
    scores.append(completeness)
    
    # Uniqueness (no duplicates)
    dup_pct = (df.duplicated().sum() / len(df)) * 100
    uniqueness = max(0, 100 - dup_pct)
    scores.append(uniqueness)
    
    # Consistency
    consistency = 85
    scores.append(consistency)
    
    return round(sum(scores) / len(scores))

def get_basic_summary(df):
    """Get basic dataset summary statistics"""
    return {
        "Total Rows": f"{df.shape[0]:,}",
        "Total Columns": df.shape[1],
        "Memory Usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
        "Numeric Columns": len(df.select_dtypes(include=[np.number]).columns),
        "Categorical Columns": len(df.select_dtypes(include=['object']).columns),
        "DateTime Columns": len(df.select_dtypes(include=['datetime64']).columns)
    }

def analyze_missing_data(df):
    """Analyse missing data patterns"""
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing %', ascending=False)
    
    recommendations = {}
    for _, row in missing_data.iterrows():
        col = row['Column']
        pct = row['Missing %']
        
        if pct > 50:
            recommendations[col] = "Consider dropping this column (>50% missing)"
        elif pct > 20:
            recommendations[col] = "Use median/mode imputation or forward fill"
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                recommendations[col] = "Suggest: Fill with median (numeric data)"
            else:
                recommendations[col] = "Suggest: Fill with mode (categorical data)"
    
    return {
        'total_missing': df.isnull().sum().sum(),
        'by_column': missing_data,
        'recommendations': recommendations
    }

def detect_outliers(df, column):
    """Detect outliers using IQR method"""
    if column not in df.columns:
        return {'count': 0, 'percentage': 0}
    
    data = df[column].dropna()
    
    if not pd.api.types.is_numeric_dtype(data):
        return {'count': 0, 'percentage': 0}
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    return {
        'count': len(outliers),
        'percentage': (len(outliers) / len(data)) * 100 if len(data) > 0 else 0,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

#======================================================
#Smart Recommender 
#======================================================

def show_smart_recommendations(df):
    """
    Display intelligent recommendations in Streamlit
    """
    import streamlit as st
    from modules.recommender import SmartRecommender
    
    st.header("🧠 Intelligent Data Strategy")
    
    with st.spinner("Analysing your data and generating recommendations..."):
        recommender = SmartRecommender(df)
        strategy = recommender.generate_complete_strategy()
    
    # Overall strategy summary
    st.subheader("📋 Recommended Action Plan")
    
    if strategy['priority_order']:
        for i, (action, priority, severity) in enumerate(strategy['priority_order'][:10], 1):
            severity_colors = {
                'CRITICAL': '🔴',
                'HIGH': '🟠',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }
            st.markdown(f"{i}. {severity_colors[severity]} **{action}** (Priority {priority})")
    
    st.markdown("---")
    
    # Detailed imputation recommendations
    if strategy['imputation']:
        st.subheader("💧 Missing Value Strategy")
        
        for rec in strategy['imputation']:
            with st.expander(f"📊 {rec['column']} - {rec['missing_pct']:.1f}% missing"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Recommended Strategy:** `{rec['strategy']}`")
                    st.markdown("**Reasoning:**")
                    for reason in rec['reasoning']:
                        st.markdown(f"- {reason}")
                    
                    if rec['alternatives']:
                        st.caption(f"Alternatives: {', '.join(rec['alternatives'])}")
                
                with col2:
                    # Confidence meter
                    st.metric("Confidence", f"{rec['confidence']}%")
                    
                    # Apply button
                    if st.button(f"Apply", key=f"apply_{rec['column']}"):
                        st.success(f"✅ Will apply {rec['strategy']} to {rec['column']}")
    
    # Outlier recommendations
    if strategy['outliers']:
        st.markdown("---")
        st.subheader("📈 Outlier Handling Strategy")
        
        for rec in strategy['outliers']:
            with st.expander(f"📊 {rec['column']} - {rec['outlier_count']} outliers ({rec['outlier_pct']:.1f}%)"):
                st.markdown(f"**Recommended Strategy:** `{rec['strategy']}`")
                st.markdown("**Reasoning:**")
                for reason in rec['reasoning']:
                    st.markdown(f"- {reason}")
                
                st.metric("Confidence", f"{rec['confidence']}%")
    
    # Feature engineering suggestions
    if strategy['feature_engineering']:
        st.markdown("---")
        st.subheader("⚡ Feature Engineering Opportunities")
        
        high_impact = [fe for fe in strategy['feature_engineering'] if fe['impact'] == 'HIGH']
        
        if high_impact:
            st.info(f"Found {len(high_impact)} high-impact feature engineering opportunities")
            
            for fe in high_impact[:5]:  # Show top 5
                with st.expander(f"✨ {fe['type']}"):
                    st.markdown(f"**New Features:** {', '.join(fe['features'][:3])}")
                    if len(fe['features']) > 3:
                        st.caption(f"...and {len(fe['features']) - 3} more")
                    st.markdown(f"**Why:** {fe['reasoning']}")
                    st.markdown(f"**Expected Impact:** {fe['impact']}")
    
    return strategy
