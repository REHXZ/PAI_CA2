import pandas as pd
import pandas as pd
import numpy as np
import os
import itertools
from AutomatedTraining import AutomatedTraining
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# import data
df = pd.read_csv('data/merged_data.csv')

# Convert data types for memory efficiency
def convert_dtypes(df):
    df['order_value'] = df['order_value'].astype('float32')
    df['refund_value'] = df['refund_value'].astype('float32')
    df['num_items_ordered'] = df['num_items_ordered'].astype(float).round().astype('uint8')
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['first_order_datetime'] = pd.to_datetime(df['first_order_datetime'])
    df[['country_code', 'collect_type', 'payment_method']] = df[['country_code', 'collect_type', 'payment_method']].astype('category')
    return df

df = convert_dtypes(df)

# Payment method grouping
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

df['payment_method'] = df['payment_method'].map(group_payment_methods)

# Date transformations
def date_transformations(df):
    df['days_since_first_order'] = (df['order_date'] - df['first_order_datetime']).dt.days
    df = df.drop(columns=['first_order_datetime'])
    df['order_date_day_of_week'] = df['order_date'].dt.dayofweek
    df['order_date_day'] = df['order_date'].dt.day
    df['order_date_month'] = df['order_date'].dt.month
    df['order_date_year'] = df['order_date'].dt.year
    df = df.drop(columns=['order_date'])
    return df

df = date_transformations(df)
df = df.drop(columns=['order_id', 'customer_id'])

# Split data
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_cols = ['payment_method', 'country_code', 'collect_type']
numeric_columns = ['order_value', 'refund_value', 'num_items_ordered', 'days_since_first_order',
                   'order_date_day_of_week', 'order_date_day', 'order_date_month', 'order_date_year']

tracker = AutomatedTraining('Automated Training')
tracker.run_experiments(X_train, y_train, X_test, y_test, categorical_cols, numeric_columns)