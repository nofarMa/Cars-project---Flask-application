import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from scipy.stats import uniform
import pickle
from car_data_prep import prepare_data
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
file_path ='Car.csv'
df = pd.read_csv(file_path)
df_processed = prepare_data(df)

# Split the data into features and target
X = df_processed.drop(columns=['Price'])
y = df_processed['Price']
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define numeric and categorical columns   
categorical_features = ['Gear', 'Engine_type', 'manufactor', 'model', 'Prev_ownership', 'Curr_ownership', 'Color','City']
numeric_features = ['Year', 'Hand', 'capacity_Engine', 'Km', 'Pic_num']


# Define the preprocessing for numerical features with Polynomial Features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))])  # Adding polynomial features

# Define the preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),  # Handle missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine the transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


# Define the Elastic Net model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(max_iter=10000, tol=1e-4, random_state=42))])

# Define the hyperparameter search space
param_distributions = {
    'regressor__alpha': uniform(0.01, 20),  # Extended range for Regularization strength
    'regressor__l1_ratio': uniform(0, 1)    # Mix between L1 and L2 regularization
}

# Setup the RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions, n_iter=500, cv=5, scoring='neg_mean_squared_error', random_state=42)

# Fit the RandomizedSearchCV to the data
random_search.fit(X_train, y_train)


best_model = random_search.best_estimator_


# Save the model to a file
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score: ", -random_search.best_score_)
print("Model saved successfully as 'trained_model.pkl'")