"""
SQL and CSV file loading.
Adapted from Churn_Prediction_XAI_with_SQL_Support.ipynb
"""
import sqlite3
import re
import os
import pandas as pd


def clean_sql_for_sqlite(sql_content: str) -> str:
    """Clean MySQL/PostgreSQL syntax so SQLite can execute it."""
    print("🧹 Cleaning SQL syntax for compatibility...")

    sql_content = sql_content.replace('`', '"')

    sql_content = re.sub(r'AUTO_INCREMENT=\d+', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'AUTO_INCREMENT', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'ENGINE\s*=\s*\w+', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'DEFAULT CHARSET\s*=\s*\w+', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'COLLATE\s*=\s*\w+', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'CHARACTER SET \w+', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'\bUNSIGNED\b', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'\bZEROFILL\b', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'\bTINYINT\b', 'INTEGER', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'\bSMALLINT\b', 'INTEGER', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'\bMEDIUMINT\b', 'INTEGER', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'\bBIGINT\b', 'INTEGER', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'\bDATETIME\b', 'TEXT', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'\bTIMESTAMP\b', 'TEXT', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'SET NAMES \w+;', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'SET @\w+\s*=.*?;', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'SET @@\w+\s*=.*?;', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'SET SQL_MODE\s*=.*?;', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'SET FOREIGN_KEY_CHECKS\s*=.*?;', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'SET UNIQUE_CHECKS\s*=.*?;', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'^--.*$', '', sql_content, flags=re.MULTILINE)
    sql_content = re.sub(r'/\*.*?\*/', '', sql_content, flags=re.DOTALL)

    print("✅ SQL cleaned successfully!")
    return sql_content


def load_sql_file(filepath: str):
    """Load a .sql dump into an in-memory SQLite DB. Returns (conn, table_names)."""
    print("=" * 60)
    print("📂 LOADING SQL FILE")
    print("=" * 60)

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        sql_content = f.read()

    print(f"📄 File size: {len(sql_content):,} characters")
    sql_content = clean_sql_for_sqlite(sql_content)

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    try:
        cursor.executescript(sql_content)
        conn.commit()
        print("✅ SQL executed successfully in memory!")
    except Exception as e:
        print(f"❌ Error executing SQL: {e}")
        conn.close()
        return None, []

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall()]

    if len(tables) == 0:
        print("❌ No tables found in SQL file!")
        conn.close()
        return None, []

    print(f"\n✅ Found {len(tables)} table(s): {tables}")
    return conn, tables


def sql_table_to_dataframe(conn, table_name: str) -> pd.DataFrame:
    """Convert a specific SQL table to a pandas DataFrame."""
    print(f"\n📊 Loading table: '{table_name}'")
    df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    print(f"✅ Loaded! Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    return df


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    print("=" * 60)
    print("📊 CSV FILE DETECTED — Loading directly...")
    print("=" * 60)
    df = pd.read_csv(filepath)
    print("✅ Dataset Loaded Successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df