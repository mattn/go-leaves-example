import lightgbm as lgb
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
  'objective': 'multiclass',
  'num_class': 3,
}
model = lgb.train(params, lgb_train, valid_sets=lgb_eval)
model.save_model('lg_iris.model')
y_pred = model.predict(X_test)
np.savetxt('iris_pred.tsv', y_pred, delimiter='\t')
np.savetxt('iris_test.tsv', X_test, delimiter='\t')
