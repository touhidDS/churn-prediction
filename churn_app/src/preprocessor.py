"""
Preprocessing: numeric conversion, date handling, missing values, encoding.
Adapted from Churn_Prediction_XAI_with_SQL_Support.ipynb
"""
import re
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder


def get_numeric_column_patterns():
    return {
        'revenue', 'income', 'salary', 'wage', 'profit', 'loss', 'balance',
        'amount', 'price', 'cost', 'fee', 'charge', 'payment', 'debt', 'debit',
        'credit', 'interest', 'rate', 'apr', 'loan', 'mortgage', 'investment',
        'id', 'code', 'number', 'no', 'num', 'account', 'customer_id', 'user_id',
        'transaction', 'invoice', 'order', 'ticket', 'reference', 'ref',
        'ein', 'ssn', 'tax', 'federal',
        'zip', 'zipcode', 'postal', 'pincode', 'pin', 'postcode',
        'percent', 'percentage', 'ownership', 'share', 'ratio',
        'count', 'quantity', 'qty', 'total', 'sum', 'number_of', 'num_of',
        'employees', 'staff', 'headcount', 'volume', 'units',
        'duration', 'minutes', 'seconds', 'calls', 'sms', 'data_usage',
        'bandwidth', 'speed', 'mbps', 'gb', 'mb', 'download', 'upload',
        'sales', 'orders', 'transactions', 'customers', 'users', 'visitors',
        'views', 'clicks', 'impressions', 'conversions', 'leads',
        'age', 'weight', 'height', 'distance', 'length', 'width', 'size',
        'score', 'rating', 'rank', 'grade', 'level', 'tier',
        'year', 'month', 'day', 'quarter', 'week', 'hour', 'minute',
        'patient_id', 'diagnosis_code', 'procedure_code', 'dosage', 'temperature',
        'pressure', 'heartrate', 'glucose',
        'product_id', 'sku', 'barcode', 'upc', 'stock', 'inventory',
        'discount', 'shipping',
        'value', 'metric', 'measure', 'figure', 'stat', 'index'
    }


def is_numeric_column(column_name: str) -> bool:
    column_lower = column_name.lower().strip()
    column_clean = re.sub(r'[^a-z0-9]', '', column_lower)
    for pattern in get_numeric_column_patterns():
        if pattern in column_clean:
            return True
    return False


def aggressive_clean_column(series):
    cleaned = series.astype(str)
    cleaned = (cleaned
        .str.replace(r'[\$€£¥₹₽₩¢]', '', regex=True)
        .str.replace(r'%', '', regex=True)
        .str.replace(r',', '', regex=True)
        .str.replace(r'-', '', regex=True)
        .str.replace(r'\b(GB|MB|KB|TB|kg|km|cm|mm|lb|oz|USD|EUR|CAD)\b', '', regex=True, flags=re.IGNORECASE)
        .str.replace(r'\s+', '', regex=True)
        .str.replace(r'[(){}\[\]]', '', regex=True)
        .str.replace(r'[^0-9.]', '', regex=True)
        .str.strip()
    )
    cleaned = cleaned.replace(['', 'nan', 'None', 'null', 'NULL'], np.nan)
    return cleaned


def smart_convert_to_numeric(df: pd.DataFrame):
    df_converted = df.copy()
    print("🔍 Analyzing columns for numeric conversion...\n")
    conversion_summary = {'converted': [], 'failed': [], 'skipped': []}

    for col in df_converted.columns:
        if df_converted[col].dtype in ['int64', 'float64', 'int32', 'float32', 'Int64']:
            conversion_summary['skipped'].append(col)
            continue

        if is_numeric_column(col):
            print(f"🎯 '{col}' identified as numeric column")
            try:
                original_dtype = df_converted[col].dtype
                sample_before = df_converted[col].head(3).tolist()
                cleaned_series = aggressive_clean_column(df_converted[col])
                converted_series = pd.to_numeric(cleaned_series, errors='coerce')
                original_non_null = df_converted[col].notna() & (df_converted[col].astype(str).str.strip() != '')
                success_rate = converted_series[original_non_null].notna().sum() / original_non_null.sum() if original_non_null.sum() > 0 else 0

                if success_rate > 0.1:
                    df_converted[col] = converted_series
                    non_null_values = df_converted[col].dropna()
                    if len(non_null_values) > 0 and (non_null_values % 1 == 0).all():
                        df_converted[col] = df_converted[col].astype('Int64')
                        final_dtype = 'Int64'
                    else:
                        final_dtype = 'float64'

                    print(f"   ✅ Converted: {original_dtype} → {final_dtype} | Success: {success_rate:.1%}")
                    print(f"   Before: {sample_before} | After: {df_converted[col].head(3).tolist()}\n")
                    conversion_summary['converted'].append(col)
                else:
                    print(f"   ⚠️  Conversion failed ({success_rate:.1%} success) — keeping as {original_dtype}\n")
                    conversion_summary['failed'].append(col)
            except Exception as e:
                print(f"   ❌ Error: {str(e)}\n")
                conversion_summary['failed'].append(col)
        else:
            conversion_summary['skipped'].append(col)

    print("\n" + "=" * 60)
    print("📊 CONVERSION SUMMARY")
    print("=" * 60)
    print(f"✅ Converted: {len(conversion_summary['converted'])} → {conversion_summary['converted']}")
    print(f"❌ Failed:    {len(conversion_summary['failed'])} → {conversion_summary['failed']}")
    print(f"⏭️  Skipped:   {len(conversion_summary['skipped'])}")
    return df_converted


def convert_dates_to_years(df: pd.DataFrame) -> pd.DataFrame:
    print("📅 DATE TO YEARS CONVERSION")
    print("=" * 70)
    df = df.copy()
    current_date = datetime.now()
    date_keywords = ['dob', 'birth', 'date', 'created', 'signup', 'registered',
                     'joined', 'start', 'opened', 'last', 'updated', 'time']

    for col in df.columns:
        col_lower = col.lower().strip()
        if any(keyword in col_lower for keyword in date_keywords):
            print(f"🔍 Found date column: {col}")
            try:
                date_series = pd.to_datetime(df[col], errors='coerce')
                if date_series.notna().sum() > 0:
                    years_diff = (current_date - date_series).dt.days / 365.25
                    new_col_name = f"{col.replace(' ', '_').lower()}_years"
                    df[new_col_name] = years_diff.round(1)
                    df = df.drop(columns=[col])
                    print(f"   ✅ Created: {new_col_name} | Sample: {df[new_col_name].dropna().head(3).tolist()}")
                    print(f"   🗑️  Dropped original: {col}\n")
                else:
                    print(f"   ⚠️  Could not parse dates, skipping\n")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}\n")

    print("=" * 70)
    print(f"✅ DATE CONVERSION COMPLETE! | Shape: {df.shape}")
    return df


def simple_missing_handler(df_with_years, target_column, drop_threshold=0.50):
    print("🔧 Simple Missing Value Handling")
    print("=" * 70)
    df = df_with_years.copy()
    total_missing = df.isnull().sum().sum()
    print(f"Missing values before: {total_missing}\n")

    if total_missing == 0:
        print("✅ No missing values!")
        return df

    columns_to_drop = [col for col in df.columns
                       if col != target_column and df[col].isnull().sum() / len(df) > drop_threshold]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"❌ Dropped {len(columns_to_drop)} columns with >{drop_threshold * 100}% missing: {columns_to_drop}\n")

    if target_column in df.columns:
        target_missing = df[target_column].isnull().sum()
        if target_missing > 0:
            df = df.dropna(subset=[target_column])
            print(f"🗑️  Dropped {target_missing} rows with missing target\n")

    for col in df.select_dtypes(include=['number']).columns:
        if col != target_column and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"✅ {col}: filled with median ({median_val:.2f})")

    for col in df.select_dtypes(include=['object']).columns:
        if col != target_column and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col] = df[col].fillna(mode_val)
            print(f"✅ {col}: filled with mode ('{mode_val}')")

    remaining = df.isnull().sum().sum()
    print(f"\n✅ DONE! Missing before: {total_missing} → After: {remaining} | Shape: {df.shape}")
    return df


def smart_categorical_encoder(df, target_column):
    print("🔧 SMART CATEGORICAL ENCODING")
    print("=" * 70)
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    if len(categorical_cols) == 0:
        print("✅ No categorical columns to encode")
        return df

    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"🔍 {col} | {unique_count} unique values")

        if unique_count == 2:
            vals = df[col].unique()
            positive_vals = ['yes', 'true', 'active', 'paid', 'y', '1', 'on', 'enabled']
            if any(str(v).lower() in positive_vals for v in vals):
                mapping = {val: (1 if str(val).lower() in positive_vals else 0) for val in vals}
            else:
                mapping = {vals[0]: 0, vals[1]: 1}
            df[col] = df[col].map(mapping)
            print(f"   ✅ Binary Encoding: {mapping}\n")

        elif 3 <= unique_count <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = df.drop(columns=[col])
            df = pd.concat([df, dummies], axis=1)
            print(f"   ✅ One-Hot: {unique_count} → {len(dummies.columns)} columns\n")

        elif 11 <= unique_count <= 20:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"   ✅ Label Encoding: {unique_count} categories\n")

        else:
            df = df.drop(columns=[col])
            print(f"   ❌ DROPPED: {unique_count} values (high cardinality)\n")

    if target_column in df.columns and df[target_column].dtype == 'object':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
        print(f"🎯 Target encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

    print(f"✅ ENCODING COMPLETE! Final shape: {df.shape}")
    return df