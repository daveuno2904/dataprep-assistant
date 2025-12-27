import pandas as pd
import numpy as np

def format_large_number(num):
    """Format Large numbers with commas"""
    return f"({num:,}"

def calculate_percentage(part, whole):

    if whole == 0:
        return 0
    return round((part / whole) * 100, 2)

def detect_column_type(series):

    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"
    
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "DateTime"
    
    elif series.unuique() / len(series) < 0.05:
        return "Categorical"
    
    else:
        return "Text"

