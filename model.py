
# %%
# load dataset

from sklearn.datasets import load_iris

dataset = load_iris()

features = dataset.data
labels = dataset.target
labelsNames =list(dataset.target_names)
featureNames = dataset.feature_names
print(featureNames)
print([labelsNames[i] for i in labels[:3]])



# %%

# veri analizi

import pandas as pd

print(type(features))

featuresDf = pd.DataFrame(features)
featuresDf.columns = featureNames

print(type(featuresDf))

print(featuresDf.describe())

print(featuresDf.info())

# %%
# Visualize

featuresDf.plot(x="sepal length (cm)",y="sepal width (cm)",kind="scatter")


# %%
X = features
y = labels

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# %%
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
acc = neigh.score(X_train, y_train)
print("accuracy on train data {:.2}%".format(acc))


# %%

acc = neigh.score(X_test,y_test)
print("accuracy on test data {:.2}%".format(acc))



# %%
from joblib import dump, load

filename ="myFirstSavedModel.joblib"
dump(neigh,filename)


# %%
neigha = load(filename)

acc = neigha.score(X_test,y_test)
print("accuracy on test data {:.2}%".format(acc))


