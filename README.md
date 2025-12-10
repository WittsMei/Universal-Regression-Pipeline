# Universal-Regression-Pipeline

### This is a "Mini-AutoML" (Automated Machine Learning) engine.
- This class is not tied to sales or marketing data specifically. It is a general-purpose wrapper that can take any cleaned dataset, test 6 different algorithms, and pick the winner for any continuous target variable (house prices, temperature, stock values, etc.).

### Why is it Generic?
- The "Contestants" are Standard: The self.models dictionary contains the standard toolkit for regression. Whether you are predicting sales or temperature, Random Forest and Linear Regression are valid candidates.
- The Logic is Universal:
  - Cross-Validation (cv=5): It doesn't trust a single run; it verifies consistency.
  - Metric (r2): R-squared is a standard way to measure regression success (1.0 is perfect, 0.0 is randomness) regardless of the data units.
  - Workflow: Train $\rightarrow$ Evaluate $\rightarrow$ Pick Winner $\rightarrow$ Retrain. This is the "Golden Path" for almost all supervised learning.
<img width="2048" height="2048" alt="image" src="https://github.com/user-attachments/assets/91cc7840-660a-4c62-9a17-0fa3d4f0a9ed" />


```python

class UniversalRegressionPipeline:
    # 1. Add 'drop_columns' as an optional argument
    def __init__(self, target_column, drop_columns=None):
        self.target_column = target_column
        self.drop_columns = drop_columns if drop_columns else []
        
        self.models = {
            "linear_regression": LinearRegression(),
            "ridge": Ridge(), "lasso": Lasso(),
            "random_forest": RandomForestRegressor(random_state=42),
            "gradient_boosting": GradientBoostingRegressor(random_state=42),
            "bayesian_ridge": BayesianRidge(),
        }
        self.best_model = None

    def train_best_model(self, df):
        # 2. Dynamically drop the columns specified by the user
        # We always drop target, plus whatever else the user asked for
        cols_to_drop = [self.target_column] + self.drop_columns
        
        X = df.drop(columns=cols_to_drop, errors='ignore') # errors='ignore' prevents crashes
        y = df[self.target_column]
        
        best_score = -np.inf
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            if scores.mean() > best_score:
                best_score = scores.mean()
                self.best_model = model
        
        self.best_model.fit(X, y)
        print(f"The winner is: {self.best_model}")

    def predict(self, df):
        cols_to_drop = [self.target_column] + self.drop_columns
        X = df.drop(columns=cols_to_drop, errors='ignore')
        return self.best_model.predict(X)
```
