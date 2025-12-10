# Universal-Regression-Pipeline

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
