import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# LOAD DATA
# save file path
file_path= ("nyc_housing_base.csv")
# read dataset
df = pd.read_csv(file_path)
df = df.drop(["bldgclass"], axis=1)
# Features
X = df.drop(["sale_price"], axis=1)
# Target
y = df["sale_price"]

# Train Test Split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
random_state=0)

# Select categorical & Numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in 
['int64', 'float64']]
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique()
< 10 and X_train_full[cname].dtype == "object"]

# Categorical options
def get_category_options(df, col):
    return sorted(df[col].dropna().astype(str).unique().tolist())

category_options = {
    col: get_category_options(df, col)
    for col in categorical_cols
}

my_cols = numerical_cols + categorical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

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

# Create & Evaluate Pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessors),
    ('model', model)
])

# Fit model
my_pipeline.fit(X_train, y_train)
# Get predictions
predictions = my_pipeline.predict(X_valid)
# Evaluate model
mae = mean_absolute_error(y_valid, predictions)
print(f"\nModel MAE: ${mae:,.2f}USD")

# User Input
print("\n=====ENTER HOUSE DETAILS TO PREDICT PRICE=====")
user_data = {}
for col in my_cols:
    if col in numerical_cols:
        while True:
            try:
                user_data[col] = float(input(f"Enter {col}: "))
                break
            except ValueError:
                print("Please Enter a valid number.")
    else:
        options = category_options[col]
        print(f"\nSelect {col}: ")
        for i, opt in enumerate(options):
            print(f"{i}: {opt}")
        while True:
            try:
                choice = int(input("Enter Option Number: "))
                if 0<= choice<len(options):
                    user_data[col] = options[choice]
                    break
                else:
                    print("Invalid Option Number.")
            except ValueError:
                print("Please Enter A number.")

# Convert to dataframe
user_df = pd.DataFrame([user_data], columns=my_cols)

# Predict
predicted_price= my_pipeline.predict(user_df)[0]

# Output the price prediction
print(f"\nPredicted House Price: {predicted_price:,.2f} USD")
