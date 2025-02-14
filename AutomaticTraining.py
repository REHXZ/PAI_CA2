import pandas as pd
import numpy as np
import os
import itertools
from AutomaticExperimentTracker import Tracker
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text

# Connection details
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

print(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)

# Create a connection string
connection_string = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# Create an engine
engine = create_engine(connection_string)

# Query data
query = "SELECT * FROM merged_data;"
df = pd.read_sql(query, engine)

# Create an engine
engine = create_engine(connection_string)

# Test the connection
try:
    with engine.connect() as conn:
        print("Connection successful :)!")
except Exception as e:
    print(f"Error connecting to the database: {e}")

def get_last_row_count():
    query = "SELECT row_count FROM last_row_count ORDER BY id DESC LIMIT 1;"
    with engine.connect() as conn:
        result = conn.execute(text(query)).scalar()
    return result or 0  # Default to 0 if no rows exist

def save_last_row_count(count):
    query = """
    INSERT INTO last_row_count (row_count) VALUES (:row_count);
    """
    with engine.connect() as conn:
        conn.execute(text(query), {"row_count": count})
        conn.commit()

def get_current_row_count():
    """Query the database for the current row count."""
    query = "SELECT COUNT(*) FROM merged_data;"
    with engine.connect() as conn:
        result = conn.execute(text(query)).scalar()
    return result

def convert_dtypes(df):
    df['order_value'] = df['order_value'].astype('float32')
    df['refund_value'] = df['refund_value'].astype('float32')
    df['num_items_ordered'] = df['num_items_ordered'].astype(float).round().astype('uint8')
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['first_order_datetime'] = pd.to_datetime(df['first_order_datetime'])
    df[['country_code', 'collect_type', 'payment_method']] = df[['country_code', 'collect_type', 'payment_method']].astype('category')
    return df

def group_payment_methods(payment_method):
    mapping = {
        'CreditCard': ['GenericCreditCard', 'CybersourceCreditCard', 'CybersourceApplePay', 'CreditCard'],
        'DigitalWallet': ['GCash', 'AFbKash', 'JazzCashWallet', 'AdyenBoost', 'PayPal'],
        'BankTransfer': ['XenditDirectDebit', 'RazerOnlineBanking'],
        'PaymentOnDelivery': ['Invoice', 'PayOnDelivery']
    }
    for key, values in mapping.items():
        if payment_method in values:
            return key
    return 'Others'

def date_transformations(df):
    df['days_since_first_order'] = (df['order_date'] - df['first_order_datetime']).dt.days
    df = df.drop(columns=['first_order_datetime'])
    df['order_date_day_of_week'] = df['order_date'].dt.dayofweek
    df['order_date_day'] = df['order_date'].dt.day
    df['order_date_month'] = df['order_date'].dt.month
    df['order_date_year'] = df['order_date'].dt.year
    df = df.drop(columns=['order_date'])
    return df

last_row_count = get_last_row_count()
current_row_count = get_current_row_count()

if current_row_count > last_row_count:
    print("New data detected. Running the script...")

    df = convert_dtypes(df)
    df['payment_method'] = df['payment_method'].map(group_payment_methods)
    df = date_transformations(df)
    df = df.drop(columns=['order_id', 'customer_id'])

    # Split data
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_cols = ['payment_method', 'country_code', 'collect_type']
    numeric_columns = ['order_value', 'refund_value', 'num_items_ordered', 'days_since_first_order',
                    'order_date_day_of_week', 'order_date_day', 'order_date_month', 'order_date_year']

    tracker = Tracker('Automated Training')
    tracker.run_experiments(X_train, y_train, X_test, y_test, categorical_cols, numeric_columns)
    save_last_row_count(current_row_count)

else:
    print("No new data. Exiting...")