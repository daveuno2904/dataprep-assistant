import pandas as pd 
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats

class SmartRecommender:
    """
    Intelligent recommendation engine for data cleanining, Analyses data patterns and suggests optimal strategies 
    """

    def __init__(self, df):
        self.df = df 
        self.recommendations = []

    def analyse_missing_pattern(self,column):
        """
        Determine if the missing data is MCAR, MAR, or MNAR
        MCAR = Missing completely at Random 
        MAR = Missing at Random
        MNAR = Missing not at Random 
        """

        missing_mask = self.df[column].isnull()
        missing_count = missing_mask.sum()

        if missing_count == 0:
            return None
        
        #Testing the correlations with other columns
        correlations = {}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if col != column:
                
                #Calculate the correlation between missingness and other variables
                corr = np.corrcoef(missing_mask.astype(int), self.df[col].fillna(0))[0,1]
                if abs(corr) > 0.3:
                    correlations[col] = corr
        
        if correlations:
            pattern = "MAR"
            explanation = f"Missing values correlate with: {', '.join(correlations.keys())}"
        else:
            pattern = "MCAR"
            explanation = "Missing values appear random"

        return {
            'pattern': pattern,
            'explanation': explanation,
            'correlations': correlations
        }

    def recommend_imputation_strategy(self, column):
        """
        Recommend the best imputations strategy based on:
        - Data type
        - Missing Pattern
        - Distribution
        - Cardinality 
        """
        col_data = self.df[column]
        missing_pct = (col_data.isnull().sum()/len(col_data)) * 100

        recommendation = {
            'column': column,
            'missing_pct': missing_pct,
            'strategy': None,
            'reasoning': [],
            'confidence': 0,
            'alternatives': []
        }
        
        if missing_pct > 70:
            recommendation['strategy'] = 'DROP_COLUMN'
            recommendation['reasoning'].append(f"Over 70% missing ({missing_pct:.1f}%)")
            recommendation['reasoning'].append("Imputation would create more noise than signal")
            recommendation['confidence'] = 95
            return recommendation
        

        #Analyse missing patterns
        pattern_info = self.analyse_missing_pattern(column)

        #if the columns are numeric

        if pd.api.types.is_numeric_dtype(col_data):
            non_missing = col_data.dropna()

            #checking for distribution 
            skewness = non_missing.skew()

            #checking for outliers 
            Q1 = non_missing.quantile(0.25)
            Q3 = non_missing.quantile(0.75)
            IQR = Q3 - Q1 
            outlier_pct = ((non_missing < Q1 - 1.5 * IQR) | (non_missing > Q3 + 1.5 * IQR)).sum()/len(non_missing) * 100

            #decision tree for numeric imputation 
            if missing_pct < 5:
                if abs(skewness) > 1:
                    recommendation['strategy'] = 'MEDIAN'
                    recommendation['reasoning'].append(f"Small missing % ({missing_pct:.1f}%)")
                    recommendation['reasoning'].append(f"Skewed distribution (skewness={skewness:.2f})")
                    recommendation['reasoning'].append("Median is robust to outliers")
                    recommendation['confidence'] = 90
                    recommendation['alternatives'] = ['MEAN', 'DROP_ROWS']
                else:
                    recommendation['strategy'] = 'MEAN'
                    recommendation['reasoning'].append(f"Small missing % ({missing_pct:.1f}%)")
                    recommendation['reasoning'].append(f"Symmetric distribution (skewness={skewness:.2f})")
                    recommendation['reasoning'].append("Mean preserves distribution")
                    recommendation['confidence'] = 85
                    recommendation['alternatives'] = ['MEDIAN', 'KNN']
                    
            elif missing_pct < 20:
                if pattern_info and pattern_info['pattern'] == 'MAR':
                    recommendation['strategy'] = 'KNN'
                    recommendation['reasoning'].append(f"Moderate missing % ({missing_pct:.1f}%)")
                    recommendation['reasoning'].append("Missing data has pattern (MAR)")
                    recommendation['reasoning'].append("KNN uses relationships in data")
                    recommendation['confidence'] = 85
                    recommendation['alternatives'] = ['MULTIPLE_IMPUTATION', 'MEDIAN']
                else:
                    recommendation['strategy'] = 'MEDIAN'
                    recommendation['reasoning'].append(f"Moderate missing % ({missing_pct:.1f}%)")
                    recommendation['reasoning'].append("Random missingness (MCAR)")
                    recommendation['reasoning'].append("Median is safe default")
                    recommendation['confidence'] = 80
                    recommendation['alternatives'] = ['KNN', 'MEAN']
                    
            else:  # 20-70% missing
                recommendation['strategy'] = 'MULTIPLE_IMPUTATION'
                recommendation['reasoning'].append(f"High missing % ({missing_pct:.1f}%)")
                recommendation['reasoning'].append("Simple imputation would bias results")
                recommendation['reasoning'].append("Multiple imputation accounts for uncertainty")
                recommendation['confidence'] = 75
                recommendation['alternatives'] = ['KNN', 'FLAG_AND_MEDIAN']
        
        #considerations for categorical columns
        else:
            non_missing = col_data.dropna()
            cardinality = non_missing.nunique()
            cardinality_pct = (cardinality / len(non_missing)) * 100
            
            # Get mode frequency
            if len(non_missing) > 0:
                mode_freq = (non_missing == non_missing.mode()[0]).sum() / len(non_missing) * 100
            else:
                mode_freq = 0
            
            if missing_pct < 5:
                if mode_freq > 50:
                    recommendation['strategy'] = 'MODE'
                    recommendation['reasoning'].append(f"Small missing % ({missing_pct:.1f}%)")
                    recommendation['reasoning'].append(f"Clear dominant category ({mode_freq:.1f}%)")
                    recommendation['reasoning'].append("Mode is natural choice")
                    recommendation['confidence'] = 90
                    recommendation['alternatives'] = ['DROP_ROWS', 'NEW_CATEGORY']
                else:
                    recommendation['strategy'] = 'NEW_CATEGORY'
                    recommendation['reasoning'].append(f"Small missing % ({missing_pct:.1f}%)")
                    recommendation['reasoning'].append(f"No dominant category (top: {mode_freq:.1f}%)")
                    recommendation['reasoning'].append("'Missing' as separate category preserves info")
                    recommendation['confidence'] = 85
                    recommendation['alternatives'] = ['MODE', 'DROP_ROWS']
                    
            elif missing_pct < 30:
                if cardinality_pct > 50:
                    recommendation['strategy'] = 'NEW_CATEGORY'
                    recommendation['reasoning'].append(f"Moderate missing % ({missing_pct:.1f}%)")
                    recommendation['reasoning'].append(f"High cardinality ({cardinality} unique values)")
                    recommendation['reasoning'].append("'Missing' preserves information")
                    recommendation['confidence'] = 80
                    recommendation['alternatives'] = ['DROP_ROWS', 'PREDICTIVE']
                else:
                    recommendation['strategy'] = 'PREDICTIVE'
                    recommendation['reasoning'].append(f"Moderate missing % ({missing_pct:.1f}%)")
                    recommendation['reasoning'].append(f"Low cardinality ({cardinality} categories)")
                    recommendation['reasoning'].append("Can predict from other features")
                    recommendation['confidence'] = 75
                    recommendation['alternatives'] = ['MODE', 'NEW_CATEGORY']
                    
            else:  
                # if 30-70% of the data missing
                recommendation['strategy'] = 'NEW_CATEGORY'
                recommendation['reasoning'].append(f"High missing % ({missing_pct:.1f}%)")
                recommendation['reasoning'].append("Too much missing to impute reliably")
                recommendation['reasoning'].append("'Missing' as category is most honest")
                recommendation['confidence'] = 85
                recommendation['alternatives'] = ['DROP_COLUMN', 'INVESTIGATE']
        
        return recommendation
    
    def recommend_outlier_strategy(self, column):
        """
        Intelligent outlier handling recommendations
        """
        col_data = self.df[column].dropna()
        
        if not pd.api.types.is_numeric_dtype(col_data):
            return None
            
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        outlier_pct = (len(outliers) / len(col_data)) * 100
        
        if outlier_pct == 0:
            return None
        
        recommendation = {
            'column': column,
            'outlier_count': len(outliers),
            'outlier_pct': outlier_pct,
            'strategy': None,
            'reasoning': [],
            'confidence': 0
        }
        
        # Check if outliers are extreme
        z_scores = np.abs(stats.zscore(col_data))
        extreme_outliers = (z_scores > 3).sum()
        
        # Check if outliers are one-sided or both
        lower_outliers = (col_data < lower_bound).sum()
        upper_outliers = (col_data > upper_bound).sum()
        
        if outlier_pct < 1:
            recommendation['strategy'] = 'REMOVE'
            recommendation['reasoning'].append(f"Very few outliers ({outlier_pct:.2f}%)")
            recommendation['reasoning'].append("Likely data entry errors or anomalies")
            recommendation['reasoning'].append("Safe to remove without losing much data")
            recommendation['confidence'] = 85
            
        elif outlier_pct < 5:
            if extreme_outliers > len(outliers) * 0.5:
                recommendation['strategy'] = 'REMOVE'
                recommendation['reasoning'].append(f"Small outlier % ({outlier_pct:.1f}%)")
                recommendation['reasoning'].append(f"Many extreme outliers (z-score > 3)")
                recommendation['reasoning'].append("Likely measurement errors")
                recommendation['confidence'] = 80
            else:
                recommendation['strategy'] = 'CAP'
                recommendation['reasoning'].append(f"Small outlier % ({outlier_pct:.1f}%)")
                recommendation['reasoning'].append("Outliers not extremely distant")
                recommendation['reasoning'].append("Capping preserves information while reducing impact")
                recommendation['confidence'] = 85
                
        elif outlier_pct < 10:
            recommendation['strategy'] = 'CAP'
            recommendation['reasoning'].append(f"Moderate outlier % ({outlier_pct:.1f}%)")
            recommendation['reasoning'].append("Removing would lose significant data")
            recommendation['reasoning'].append("Capping at IQR boundaries reduces impact")
            recommendation['confidence'] = 80
            
        else:
            recommendation['strategy'] = 'TRANSFORM'
            recommendation['reasoning'].append(f"High outlier % ({outlier_pct:.1f}%)")
            recommendation['reasoning'].append("Data may have heavy-tailed distribution")
            recommendation['reasoning'].append("Log transform or Box-Cox may normalize distribution")
            recommendation['confidence'] = 75
        
        return recommendation
    
    def recommend_feature_engineering(self, column):
        """
        Suggest feature engineering opportunities
        """
        recommendations = []
        col_data = self.df[column]
        
        # DATETIME features
        if pd.api.types.is_datetime64_any_dtype(col_data):
            recommendations.append({
                'type': 'DATETIME_DECOMPOSITION',
                'features': ['year', 'month', 'day', 'dayofweek', 'hour', 'is_weekend'],
                'reasoning': 'Extract temporal patterns from datetime',
                'impact': 'HIGH'
            })
        
        # NUMERIC features
        elif pd.api.types.is_numeric_dtype(col_data):
            non_missing = col_data.dropna()
            
            # Check if log transform makes sense
            if (non_missing > 0).all():
                skewness = non_missing.skew()
                if abs(skewness) > 1:
                    recommendations.append({
                        'type': 'LOG_TRANSFORM',
                        'features': [f'log_{column}'],
                        'reasoning': f'Highly skewed distribution (skewness={skewness:.2f})',
                        'impact': 'MEDIUM'
                    })
            
            # Suggest binning if many unique values
            if non_missing.nunique() > 100:
                recommendations.append({
                    'type': 'BINNING',
                    'features': [f'{column}_binned'],
                    'reasoning': 'High cardinality - binning may reveal patterns',
                    'impact': 'LOW'
                })
            
            # Suggest polynomial features if appropriate
            if non_missing.nunique() > 10 and len(non_missing) < 10000:
                recommendations.append({
                    'type': 'POLYNOMIAL',
                    'features': [f'{column}_squared', f'{column}_cubed'],
                    'reasoning': 'Capture non-linear relationships',
                    'impact': 'MEDIUM'
                })
        
        # CATEGORICAL features
        else:
            cardinality = col_data.nunique()
            
            if cardinality > 10 and cardinality < 50:
                recommendations.append({
                    'type': 'ONE_HOT_ENCODING',
                    'features': [f'{column}_{val}' for val in col_data.unique()[:5]],
                    'reasoning': 'Moderate cardinality suitable for one-hot encoding',
                    'impact': 'HIGH'
                })
            elif cardinality >= 50:
                recommendations.append({
                    'type': 'TARGET_ENCODING',
                    'features': [f'{column}_encoded'],
                    'reasoning': 'High cardinality - target encoding more efficient',
                    'impact': 'HIGH'
                })
        
        return recommendations
    
    def generate_complete_strategy(self):
        """
        Generate complete data preparation strategy
        """
        strategy = {
            'imputation': [],
            'outliers': [],
            'feature_engineering': [],
            'duplicates': None,
            'data_types': [],
            'priority_order': []
        }
        
        # 1. Check duplicates first
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            strategy['duplicates'] = {
                'count': dup_count,
                'recommendation': 'REMOVE',
                'reasoning': f'{dup_count} exact duplicate rows found',
                'priority': 1
            }
        
        # 2. Analyze each column for missing values
        for col in self.df.columns:
            if self.df[col].isnull().any():
                rec = self.recommend_imputation_strategy(col)
                strategy['imputation'].append(rec)
        
        # 3. Analyze numeric columns for outliers
        for col in self.df.select_dtypes(include=[np.number]).columns:
            rec = self.recommend_outlier_strategy(col)
            if rec:
                strategy['outliers'].append(rec)
        
        # 4. Suggest feature engineering
        for col in self.df.columns:
            recs = self.recommend_feature_engineering(col)
            if recs:
                strategy['feature_engineering'].extend(recs)
        
        # 5. Create priority order
        priority_order = []
        
        if strategy['duplicates']:
            priority_order.append(('Remove duplicates', 1, 'CRITICAL'))
        
        # High confidence imputation
        for imp in strategy['imputation']:
            if imp['confidence'] >= 85:
                priority_order.append((f"Impute {imp['column']}", 2, 'HIGH'))
        
        # Outlier handling
        for out in strategy['outliers']:
            if out['confidence'] >= 80:
                priority_order.append((f"Handle outliers in {out['column']}", 3, 'MEDIUM'))
        
        # Feature engineering
        for fe in strategy['feature_engineering']:
            if fe['impact'] == 'HIGH':
                priority_order.append((f"Engineer {fe['type']}", 4, 'LOW'))
        
        strategy['priority_order'] = priority_order
        
        return strategy

