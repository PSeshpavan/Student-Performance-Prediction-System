import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

model = pickle.load(open('artifacts/model.pkl','rb'))
preprocessor = pickle.load(open('artifacts/preprocessor.pkl','rb'))
df = pd.read_csv('artifacts/test.csv')

X = df.drop('math_score', axis=1)
y = df['math_score']

X_trans = preprocessor.transform(X)
preds = model.predict(X_trans)

print(f"R2: {r2_score(y, preds):.4f}")
print(f"MAE: {mean_absolute_error(y, preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, preds)):.4f}")
