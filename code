import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. SYNTHETIC DATA GENERATION (For testing)
# ==========================================
print("Generating synthetic housing data...")
np.random.seed(42)
n_samples = 1000

data = {
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'sqft_living': np.random.normal(2000, 500, n_samples),
    'sqft_lot': np.random.normal(5000, 2000, n_samples),
    'yr_built': np.random.randint(1950, 2023, n_samples),
    'zipcode': np.random.choice(['98001', '98002', '98003', '98004'], n_samples),
    'condition': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], n_samples),
    'view': np.random.choice(['None', 'City', 'Water'], n_samples)
}

df = pd.DataFrame(data)

# Create a realistic price target based on features
base_price = 100000
df['price'] = (
    base_price + 
    (df['bedrooms'] * 50000) + 
    (df['sqft_living'] * 150) + 
    ((2023 - df['yr_built']) * -500) + # Newer homes cost more
    np.where(df['view'] == 'Water', 100000, 0) + 
    np.random.normal(0, 20000, n_samples) # Add some noise
)

X = df.drop('price', axis=1)
y = df['price']

# ==========================================
# 2. ADVANCED PREPROCESSING PIPELINE
# ==========================================
print("Building preprocessing pipeline...")
numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'yr_built']
categorical_features = ['zipcode', 'condition', 'view']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ==========================================
# 3. MODEL SELECTION & PIPELINE
# ==========================================
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# ==========================================
# 4. HYPERPARAMETER TUNING
# ==========================================
print("Starting hyperparameter tuning (this may take a moment)...")
param_distributions = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__learning_rate': [0.05, 0.1],
    'regressor__max_depth': [3, 4, 5],
}

search = RandomizedSearchCV(
    model_pipeline, 
    param_distributions, 
    n_iter=5, # Kept low for quick testing
    cv=3, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1,
    random_state=42
)

# ==========================================
# 5. TRAINING & EVALUATION
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

search.fit(X_train, y_train)
best_model = search.best_estimator_

predictions = best_model.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Best Parameters Found: {search.best_params_}")
print(f"Mean Absolute Error (MAE): ${mean_absolute_error(y_test, predictions):,.2f}")
print(f"R-squared (R2) Score: {r2_score(y_test, predictions):.4f}")

print("\nPlotting feature importance...")
importances = best_model.named_steps['regressor'].feature_importances_

# We need to get the feature names after one-hot encoding
cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_feature_names)

# Sort and plot
indices = np.argsort(importances)[-10:] # Top 10 features
plt.figure(figsize=(10,6))
plt.title('Top 10 Key Drivers of House Price')
plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
plt.yticks(range(len(indices)), [all_feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()


# 6. MODEL PERSISTENCE (SAVING)
#------------------------------------
model_filename = 'advanced_house_model.pkl'
joblib.dump(best_model, model_filename)
print(f"\nModel successfully saved to '{model_filename}'.")
print("You can load this later using: loaded_model = joblib.load('advanced_house_model.pkl')")
