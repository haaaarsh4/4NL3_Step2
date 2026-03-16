from model import Model
import pandas as pd
from sklearn.metrics import f1_score, classification_report

train_df = pd.read_csv("train.csv")
val_df   = pd.read_csv("val.csv")
test_df  = pd.read_csv("test_labels.csv")

X_train, y_train = train_df["dialogue"], train_df["name"]
X_val,   y_val   = val_df["dialogue"],   val_df["name"]
X_test,  y_test  = test_df["dialogue"],  test_df["name"]

model = Model()
model.fit(X_train, y_train)

val_preds  = model.predict(X_val)
val_f1     = f1_score(y_val, val_preds, average="weighted")
print(f"Validation F1: {val_f1:.4f}")

test_preds = model.predict(X_test)
test_f1    = f1_score(y_test, test_preds, average="weighted")
print(f"Test F1: {test_f1:.4f}")

print("\nClassification Report (Test):")
print(classification_report(y_test, test_preds))