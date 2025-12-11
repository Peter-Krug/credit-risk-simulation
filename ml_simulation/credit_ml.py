import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

np.random.seed(7)

# Simulate credit portfolio
n = 1000
df = pd.DataFrame({
    "EaD": np.random.randint(1000, 50000, n),   # loan size
    "PD": np.random.uniform(0.01, 0.10, n),     # default probability
    "LGD": np.random.uniform(0.2, 0.6, n)       # loss ratio
})

# default probabilities
loan_size_weight = .5 # controls the impact of loan size on default probability (bigger loans have higher chance of defaulting)

default_prob = df["PD"] + loan_size_weight * (df["EaD"]/df["EaD"].max())
default_prob = np.clip(default_prob, 0, 1)
df["Default"] = (np.random.rand(n) < default_prob).astype(int)

# Expected & Real Loss
df["Expected_Loss"] = df["PD"] * df["LGD"] * df["EaD"]
df["Real_Loss"] = df["Default"] * df["LGD"] * df["EaD"]

print(df.head())
print("Total Real Loss:", round(df["Real_Loss"].sum(), 2))

# Train/test split
X = df[["EaD", "PD", "LGD"]]
y = df["Default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


# Train Logistic Regression
model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)

# Predictions
y_pred_prob = model.predict_proba(X_test)[:, 1] #Returns 2D Array with probability of defaulting with rows like this: [p(no default), p(default)] for each input triple (EaD,PD,LGD)

# Evaluate AUC
auc = roc_auc_score(y_test, y_pred_prob)    # AUC score measures how well the model discriminates between default and non-default
print("Test AUC:", round(auc, 3))           # AUC > 0.5 indicates performance better than random guessing


# Feature importance
importance = pd.Series(model.coef_[0], index=X.columns)
print("\nFeature Importance:")
print(importance)


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()