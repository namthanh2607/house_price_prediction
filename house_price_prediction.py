import pandas as pd
import sklearn
import matplotlib.pyplot as plt

housing = pd.read_csv("Housing.csv")

def select_features(df, features_name):
    return df[features_name].values

def select_label(df, label_name):
    return df[label_name].values

features = select_features(housing,["area","bedrooms","bathrooms","parking","stories"])
labels = select_label(housing,"price")

from sklearn.model_selection import train_test_split

feature_temp, feature_test, label_temp, label_test = train_test_split(
    features, labels, test_size=0.1, random_state=42
)
feature_train, feature_val, label_train, label_val = train_test_split(
    feature_temp, label_temp, test_size=1 / 9, random_state=42
)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(feature_train, label_train)

train_r2_score = model.score(feature_train, label_train)
val_r2_score = model.score(feature_val, label_val)

print(f"R2 score of train set: {train_r2_score}")
print(f"R2 score of validation set: {val_r2_score}")
