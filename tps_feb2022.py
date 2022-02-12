import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from xgboost import XGBClassifier

seed = 3165
bacteria_data = pd.read_csv('../input/tabular-playground-series-feb-2022/train.csv', index_col='row_id')
le = LabelEncoder()
X = bacteria_data.drop(columns='target')
y = le.fit_transform(bacteria_data.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
xgb_model = XGBClassifier(use_label_encoder=False,
                          learning_rate=0.1,
                          n_estimators=1000,
                          tree_method='gpu_hist',
                          random_state=seed)
pipe = Pipeline([
    ('scaler', RobustScaler()),
    ('model', xgb_model)
])
param_grid = {
    'model__max_depth': [3, 4, 5, 6, 7, 8],
    'model__min_split_loss': [0, 5, 10, 15, 20],
    'model__min_child_weight': [0, 1, 2, 3],
    'model__lambda': [0, 1, 2, 3],
    'model__alpha': [0, 1, 2, 3]
}
grid = GridSearchCV(pipe, param_grid=param_grid,
                    scoring='accuracy', n_jobs=-1,
                    cv=5, refit=True)
grid.fit(X_train, y_train)
print(f'Best Params: {grid.best_params_}')
print(f'Best Score: {grid.best_score_}')
print(f'Train Score: {grid.score(X_train, y_train)}')
print(f'Test Score: {grid.score(X_test, y_test)}')
best_xgbm = grid.best_estimator_['model']
best_xgbm.fit(X,y)
bacteria_test_data = pd.read_csv('../input/tabular-playground-series-feb-2022/test.csv', index_col='row_id')
bacteria_test_data['target'] = le.inverse_transform(best_xgbm.predict(bacteria_test_data))
bacteria_test_data['target'].to_csv('./submission.csv')
