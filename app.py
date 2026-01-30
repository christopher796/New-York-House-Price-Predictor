import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title=" House Price Predictor", layout="centered")

st.title("🏢New York City House Price Predictor")

# load data
@st.cache_data
def load_data():
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "nyc_housing_base.csv")
    return pd.read_csv(csv_path)

df = load_data()
df = df.drop(["bldgclass"], axis=1)

# Features & Target
X = df.drop("sale_price", axis=1)
y = df["sale_price"]

# Train Test Split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
random_state=0)

# Column selection
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in 
['int64', 'float64']]
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique()
< 10 and X_train_full[cname].dtype == "object"]

# Category options
def get_category_options(df, col):
    return sorted(df[col].dropna().astype(str).unique().tolist())

category_options = {
    col: get_category_options(df, col)
    for col in categorical_cols
}

my_cols = numerical_cols + categorical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Preprocessing
# Preprocess numerical data
numerical_transformer = SimpleImputer(strategy='median')
# Preprocess categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Build a transformer safely
transformers = []
if numerical_cols:
    transformers.append(('num', numerical_transformer, numerical_cols))
if categorical_cols:
    transformers.append(('cat', categorical_transformer, categorical_cols))

# Bundle preprocessing for numerical & categorical data
preprocessors = ColumnTransformer(
    transformers=transformers, remainder='drop'
)

# Define model
model = XGBRegressor(
    n_estimators = 3500,
    learning_rate = 0.02,
    max_depth = 7,
    min_child_weight= 5,
    subsample = 0.8,
    colsample_bytree=0.8,
    reg_alpha= 0.1,
    reg_lambda = 1.0,
    objective= "reg:squarederror",
    random_state= 0,
    n_jobs= 5
)

# Create Pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessors),
    ('model', model)
])

# Train model
@st.cache_resource
def train_model():
    my_pipeline.fit(X_train, y_train)
    return my_pipeline
my_pipeline = train_model()

# User inputs
st.divider()
st.subheader("Enter House Details")

user_data = {}
for col in numerical_cols:
    user_data[col] = st.number_input(
        f"{col}",
        value = float(X[col].median())
    )
for col in categorical_cols:
    user_data[col] = st.selectbox(
        col,
        category_options[col]
    )

user_df = pd.DataFrame([user_data])

# Prediction Button
if st.button("Predict Price"):
    prediction = my_pipeline.predict(user_df)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f} USD")

