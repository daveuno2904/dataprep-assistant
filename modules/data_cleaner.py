# modules/data_cleaner.py

import pandas as pd
import numpy as np

def apply_cleaning_steps(df, steps):
    """Apply all cleaning steps to dataframe"""
    cleaned = df.copy()
    
    for step in steps:
        if step[0] == "duplicates":
            cleaned = cleaned.drop_duplicates()
        else:
            col, strategy = step
            
            if "Drop rows" in strategy:
                cleaned = cleaned.dropna(subset=[col])
            
            elif "Fill: Mean" in strategy:
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    cleaned[col].fillna(cleaned[col].mean(), inplace=True)
            
            elif "Fill: Median" in strategy:
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    cleaned[col].fillna(cleaned[col].median(), inplace=True)
            
            elif "Fill: Mode" in strategy:
                cleaned[col].fillna(cleaned[col].mode()[0], inplace=True)
            
            elif "Fill: Forward Fill" in strategy:
                cleaned[col].fillna(method='ffill', inplace=True)
            
            elif "Remove" in strategy and col in cleaned.select_dtypes(include=[np.number]).columns:
                # Remove outliers
                Q1 = cleaned[col].quantile(0.25)
                Q3 = cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                cleaned = cleaned[(cleaned[col] >= lower) & (cleaned[col] <= upper)]
            
            elif "Cap at IQR" in strategy:
                Q1 = cleaned[col].quantile(0.25)
                Q3 = cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                cleaned[col] = cleaned[col].clip(lower, upper)
    
    return cleaned