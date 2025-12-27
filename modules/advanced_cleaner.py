import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats

class AdvancedCleaner:
    """
    Execute advanced cleaning strategies recommended by SmartRecommender
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        self.changes_log = []
        
    def apply_imputation(self, column, strategy, custom_value=None):
        """
        Apply recommended imputation strategy
        """
        before_missing = self.df[column].isnull().sum()
        
        if strategy == 'DROP_COLUMN':
            self.df = self.df.drop(columns=[column])
            self.changes_log.append({
                'action': 'DROP_COLUMN',
                'column': column,
                'details': f'Dropped column with {before_missing} missing values'
            })
            
        elif strategy == 'DROP_ROWS':
            before_rows = len(self.df)
            self.df = self.df.dropna(subset=[column])
            rows_dropped = before_rows - len(self.df)
            self.changes_log.append({
                'action': 'DROP_ROWS',
                'column': column,
                'details': f'Dropped {rows_dropped} rows with missing {column}'
            })
            
        elif strategy == 'MEAN':
            mean_val = self.df[column].mean()
            self.df[column].fillna(mean_val, inplace=True)
            self.changes_log.append({
                'action': 'MEAN_IMPUTATION',
                'column': column,
                'details': f'Filled {before_missing} values with mean: {mean_val:.2f}'
            })
            
        elif strategy == 'MEDIAN':
            median_val = self.df[column].median()
            self.df[column].fillna(median_val, inplace=True)
            self.changes_log.append({
                'action': 'MEDIAN_IMPUTATION',
                'column': column,
                'details': f'Filled {before_missing} values with median: {median_val:.2f}'
            })
            
        elif strategy == 'MODE':
            mode_val = self.df[column].mode()[0]
            self.df[column].fillna(mode_val, inplace=True)
            self.changes_log.append({
                'action': 'MODE_IMPUTATION',
                'column': column,
                'details': f'Filled {before_missing} values with mode: {mode_val}'
            })
            
        elif strategy == 'NEW_CATEGORY':
            self.df[column].fillna('MISSING', inplace=True)
            self.changes_log.append({
                'action': 'NEW_CATEGORY',
                'column': column,
                'details': f'Created "MISSING" category for {before_missing} values'
            })
            
        elif strategy == 'KNN':
            # KNN imputation for numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if column in numeric_cols and len(numeric_cols) > 1:
                imputer = KNNImputer(n_neighbors=5)
                self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
                self.changes_log.append({
                    'action': 'KNN_IMPUTATION',
                    'column': column,
                    'details': f'KNN imputation on {before_missing} values using 5 neighbors'
                })
            else:
                # Fallback to median
                median_val = self.df[column].median()
                self.df[column].fillna(median_val, inplace=True)
                self.changes_log.append({
                    'action': 'MEDIAN_IMPUTATION',
                    'column': column,
                    'details': f'Fell back to median (KNN requires multiple numeric columns)'
                })
                
        elif strategy == 'MULTIPLE_IMPUTATION':
            # MICE (Multiple Imputation by Chained Equations)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if column in numeric_cols:
                imputer = IterativeImputer(random_state=42, max_iter=10)
                self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
                self.changes_log.append({
                    'action': 'MULTIPLE_IMPUTATION',
                    'column': column,
                    'details': f'MICE imputation on {before_missing} values'
                })
            else:
                # Fallback to mode for categorical
                mode_val = self.df[column].mode()[0]
                self.df[column].fillna(mode_val, inplace=True)
                self.changes_log.append({
                    'action': 'MODE_IMPUTATION',
                    'column': column,
                    'details': f'Fell back to mode for categorical column'
                })
                
        elif strategy == 'FORWARD_FILL':
            self.df[column].fillna(method='ffill', inplace=True)
            # Handle any remaining NaN at the start
            self.df[column].fillna(method='bfill', inplace=True)
            self.changes_log.append({
                'action': 'FORWARD_FILL',
                'column': column,
                'details': f'Forward filled {before_missing} missing values'
            })
            
        elif strategy == 'CUSTOM' and custom_value is not None:
            self.df[column].fillna(custom_value, inplace=True)
            self.changes_log.append({
                'action': 'CUSTOM_IMPUTATION',
                'column': column,
                'details': f'Filled {before_missing} values with custom value: {custom_value}'
            })
        
        return self.df
    
    def handle_outliers(self, column, strategy):
        """
        Handle outliers using recommended strategy
        """
        before_rows = len(self.df)
        col_data = self.df[column]
        
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if strategy == 'REMOVE':
            self.df = self.df[~outlier_mask]
            self.changes_log.append({
                'action': 'REMOVE_OUTLIERS',
                'column': column,
                'details': f'Removed {outlier_count} outlier rows'
            })
            
        elif strategy == 'CAP':
            self.df.loc[col_data < lower_bound, column] = lower_bound
            self.df.loc[col_data > upper_bound, column] = upper_bound
            self.changes_log.append({
                'action': 'CAP_OUTLIERS',
                'column': column,
                'details': f'Capped {outlier_count} outliers to IQR bounds'
            })
            
        elif strategy == 'TRANSFORM':
            # Try log transform if all values are positive
            if (col_data > 0).all():
                self.df[f'{column}_log'] = np.log(col_data)
                self.changes_log.append({
                    'action': 'LOG_TRANSFORM',
                    'column': column,
                    'details': f'Applied log transform, created {column}_log'
                })
            else:
                # Box-Cox requires positive values, so add constant if needed
                min_val = col_data.min()
                if min_val <= 0:
                    shift = abs(min_val) + 1
                    col_data_shifted = col_data + shift
                else:
                    col_data_shifted = col_data
                    
                try:
                    transformed, lambda_val = stats.boxcox(col_data_shifted)
                    self.df[f'{column}_transformed'] = transformed
                    self.changes_log.append({
                        'action': 'BOXCOX_TRANSFORM',
                        'column': column,
                        'details': f'Applied Box-Cox transform (λ={lambda_val:.3f})'
                    })
                except:
                    # If Box-Cox fails, fall back to capping
                    self.df.loc[col_data < lower_bound, column] = lower_bound
                    self.df.loc[col_data > upper_bound, column] = upper_bound
                    self.changes_log.append({
                        'action': 'CAP_OUTLIERS',
                        'column': column,
                        'details': f'Transform failed, capped {outlier_count} outliers instead'
                    })
        
        elif strategy == 'WINSORIZE':
            # Cap at 5th and 95th percentiles
            lower_percentile = col_data.quantile(0.05)
            upper_percentile = col_data.quantile(0.95)
            
            capped_count = ((col_data < lower_percentile) | (col_data > upper_percentile)).sum()
            
            self.df.loc[col_data < lower_percentile, column] = lower_percentile
            self.df.loc[col_data > upper_percentile, column] = upper_percentile
            
            self.changes_log.append({
                'action': 'WINSORIZE',
                'column': column,
                'details': f'Winsorized {capped_count} values to 5th-95th percentile range'
            })
        
        return self.df
    
    def create_features(self, column, feature_type):
        """
        Create engineered features
        """
        if feature_type == 'DATETIME_DECOMPOSITION':
            if pd.api.types.is_datetime64_any_dtype(self.df[column]):
                self.df[f'{column}_year'] = self.df[column].dt.year
                self.df[f'{column}_month'] = self.df[column].dt.month
                self.df[f'{column}_day'] = self.df[column].dt.day
                self.df[f'{column}_dayofweek'] = self.df[column].dt.dayofweek
                self.df[f'{column}_is_weekend'] = self.df[column].dt.dayofweek.isin([5, 6]).astype(int)
                
                self.changes_log.append({
                    'action': 'DATETIME_FEATURES',
                    'column': column,
                    'details': 'Created year, month, day, dayofweek, is_weekend features'
                })
                
        elif feature_type == 'LOG_TRANSFORM':
            if pd.api.types.is_numeric_dtype(self.df[column]):
                if (self.df[column] > 0).all():
                    self.df[f'{column}_log'] = np.log(self.df[column])
                    self.changes_log.append({
                        'action': 'LOG_FEATURE',
                        'column': column,
                        'details': f'Created log-transformed feature: {column}_log'
                    })
                    
        elif feature_type == 'POLYNOMIAL':
            if pd.api.types.is_numeric_dtype(self.df[column]):
                self.df[f'{column}_squared'] = self.df[column] ** 2
                self.df[f'{column}_cubed'] = self.df[column] ** 3
                self.changes_log.append({
                    'action': 'POLYNOMIAL_FEATURES',
                    'column': column,
                    'details': f'Created squared and cubed features'
                })
                
        elif feature_type == 'BINNING':
            if pd.api.types.is_numeric_dtype(self.df[column]):
                self.df[f'{column}_binned'] = pd.qcut(self.df[column], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'], duplicates='drop')
                self.changes_log.append({
                    'action': 'BINNING',
                    'column': column,
                    'details': 'Created 5 quantile-based bins'
                })
        
        return self.df
    
    def get_cleaning_summary(self):
        """
        Generate summary of all cleaning operations
        """
        summary = {
            'original_shape': self.original_df.shape,
            'final_shape': self.df.shape,
            'rows_removed': self.original_df.shape[0] - self.df.shape[0],
            'columns_added': self.df.shape[1] - self.original_df.shape[1],
            'operations': len(self.changes_log),
            'changes': self.changes_log
        }
        
        return summary
    
    def compare_quality(self):
        """
        Compare data quality before and after cleaning
        """
        original_missing = self.original_df.isnull().sum().sum()
        final_missing = self.df.isnull().sum().sum()
        
        original_duplicates = self.original_df.duplicated().sum()
        final_duplicates = self.df.duplicated().sum()
        
        # Quality score calculation
        original_score = self._calculate_quality_score(self.original_df)
        final_score = self._calculate_quality_score(self.df)
        
        comparison = {
            'missing_values': {
                'before': original_missing,
                'after': final_missing,
                'improvement': original_missing - final_missing
            },
            'duplicates': {
                'before': original_duplicates,
                'after': final_duplicates,
                'improvement': original_duplicates - final_duplicates
            },
            'quality_score': {
                'before': original_score,
                'after': final_score,
                'improvement': final_score - original_score
            },
            'row_count': {
                'before': len(self.original_df),
                'after': len(self.df),
                'change': len(self.df) - len(self.original_df)
            }
        }
        
        return comparison
    
    def _calculate_quality_score(self, df):
        """
        Calculate overall data quality score
        """
        scores = []
        
        # Completeness
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        completeness = max(0, 100 - missing_pct)
        scores.append(completeness)
        
        # Uniqueness
        dup_pct = (df.duplicated().sum() / len(df)) * 100
        uniqueness = max(0, 100 - dup_pct)
        scores.append(uniqueness)
        
        # Validity (simplified)
        validity = 85  # Placeholder
        scores.append(validity)
        
        return round(sum(scores) / len(scores))


# ============================================
# Integration with Streamlit app.py
# Enhanced Clean Data page
# ============================================

def enhanced_cleaning_interface(df):
    """
    Advanced cleaning interface with smart recommendations
    """
    import streamlit as st
    from modules.recommender import SmartRecommender
    from modules.advanced_cleaner import AdvancedCleaner
    
    st.header("🛠️ Smart Data Cleaning")
    
    # Initialize cleaner
    if 'cleaner' not in st.session_state:
        st.session_state.cleaner = AdvancedCleaner(df)
    
    # Get smart recommendations
    if 'strategy' not in st.session_state:
        with st.spinner("Analyzing your data..."):
            recommender = SmartRecommender(df)
            st.session_state.strategy = recommender.generate_complete_strategy()
    
    strategy = st.session_state.strategy
    cleaner = st.session_state.cleaner
    
    # Show recommended action plan
    st.subheader("📋 Recommended Cleaning Plan")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Issues Found", len(strategy['imputation']) + len(strategy['outliers']))
    with col2:
        critical_issues = sum(1 for _, _, sev in strategy['priority_order'] if sev == 'CRITICAL')
        st.metric("Critical Issues", critical_issues)
    with col3:
        st.metric("Estimated Time", "< 1 minute")
    
    st.markdown("---")
    
    # Tabbed interface for different cleaning operations
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Auto-Clean", "💧 Missing Values", "📈 Outliers", "⚡ Feature Engineering"])
    
    with tab1:
        st.subheader("🎯 One-Click Smart Cleaning")
        st.info("Apply all high-confidence recommendations automatically")
        
        # Show what will be done
        st.markdown("**This will:**")
        for action, priority, severity in strategy['priority_order'][:10]:
            st.markdown(f"- {action}")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 Auto-Clean Now", type="primary", use_container_width=True):
                with st.spinner("Cleaning your data..."):
                    # Apply all high-confidence recommendations
                    for imp in strategy['imputation']:
                        if imp['confidence'] >= 85:
                            cleaner.apply_imputation(imp['column'], imp['strategy'])
                    
                    for out in strategy['outliers']:
                        if out['confidence'] >= 80:
                            cleaner.handle_outliers(out['column'], out['strategy'])
                    
                    # Store cleaned data
                    st.session_state.cleaned_df = cleaner.df
                    st.session_state.cleaning_summary = cleaner.get_cleaning_summary()
                    
                    st.success("✅ Auto-cleaning complete!")
                    st.balloons()
                    
                    # Show summary
                    summary = cleaner.get_cleaning_summary()
                    st.markdown("### 📊 Cleaning Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Operations", summary['operations'])
                    with col2:
                        st.metric("Rows Removed", summary['rows_removed'])
                    with col3:
                        st.metric("Columns Added", summary['columns_added'])
                    
                    # Quality comparison
                    comparison = cleaner.compare_quality()
                    
                    st.markdown("### ⭐ Quality Improvement")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Missing Values", 
                            comparison['missing_values']['after'],
                            delta=-comparison['missing_values']['improvement'],
                            delta_color="inverse"
                        )
                    with col2:
                        st.metric(
                            "Duplicates",
                            comparison['duplicates']['after'],
                            delta=-comparison['duplicates']['improvement'],
                            delta_color="inverse"
                        )
                    with col3:
                        st.metric(
                            "Quality Score",
                            f"{comparison['quality_score']['after']}%",
                            delta=f"+{comparison['quality_score']['improvement']}%"
                        )
    
    with tab2:
        st.subheader("💧 Missing Value Handling")
        
        if strategy['imputation']:
            for imp in strategy['imputation']:
                with st.expander(f"📊 {imp['column']} - {imp['missing_pct']:.1f}% missing - Confidence: {imp['confidence']}%"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Recommended:** `{imp['strategy']}`")
                        st.markdown("**Why:**")
                        for reason in imp['reasoning']:
                            st.markdown(f"- {reason}")
                    
                    with col2:
                        # Allow user to override
                        selected_strategy = st.selectbox(
                            "Strategy:",
                            [imp['strategy']] + imp['alternatives'],
                            key=f"strat_{imp['column']}"
                        )
                        
                        if st.button("Apply", key=f"apply_{imp['column']}"):
                            cleaner.apply_imputation(imp['column'], selected_strategy)
                            st.success(f"✅ Applied {selected_strategy}")
        else:
            st.success("✅ No missing values to handle!")
    
    with tab3:
        st.subheader("📈 Outlier Management")
        
        if strategy['outliers']:
            for out in strategy['outliers']:
                with st.expander(f"📊 {out['column']} - {out['outlier_count']} outliers ({out['outlier_pct']:.1f}%)"):
                    st.markdown(f"**Recommended:** `{out['strategy']}`")
                    st.markdown("**Why:**")
                    for reason in out['reasoning']:
                        st.markdown(f"- {reason}")
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        strategy_choice = st.selectbox(
                            "Strategy:",
                            ['REMOVE', 'CAP', 'TRANSFORM', 'WINSORIZE'],
                            index=['REMOVE', 'CAP', 'TRANSFORM', 'WINSORIZE'].index(out['strategy']),
                            key=f"out_strat_{out['column']}"
                        )
                    with col2:
                        if st.button("Apply", key=f"apply_out_{out['column']}"):
                            cleaner.handle_outliers(out['column'], strategy_choice)
                            st.success(f"✅ Applied {strategy_choice}")
        else:
            st.success("✅ No significant outliers detected!")
    
    with tab4:
        st.subheader("⚡ Feature Engineering")
        
        if strategy['feature_engineering']:
            high_impact = [fe for fe in strategy['feature_engineering'] if fe['impact'] == 'HIGH']
            
            if high_impact:
                st.info(f"Found {len(high_impact)} high-impact opportunities")
                
                for fe in high_impact:
                    with st.expander(f"✨ {fe['type']}"):
                        st.markdown(f"**New Features:** {', '.join(fe['features'][:3])}")
                        st.markdown(f"**Reasoning:** {fe['reasoning']}")
                        st.markdown(f"**Impact:** {fe['impact']}")
                        
                        # Find the column name from features (it's usually in the feature name)
                        # This is a simplified approach
                        if fe['features']:
                            base_col = fe['features'][0].split('_')[0]
                            if base_col in df.columns:
                                if st.button(f"Create Features", key=f"fe_{fe['type']}_{base_col}"):
                                    cleaner.create_features(base_col, fe['type'])
                                    st.success(f"✅ Created {fe['type']} features!")
        else:
            st.info("No immediate feature engineering opportunities identified")