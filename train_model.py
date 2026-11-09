# train_model.py
import io
import zipfile
import urllib.request

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------
# PART 1 - Retrieve the dataset (Activity A)
# ---------------------------------------------------------------
# NOTE: fetch_ucirepo(id=320) returns 649 rows, which is the
# PORTUGUESE-language course. The lab requires the MATHEMATICS
# records (395 rows), so we read student-mat.csv from the UCI zip.
UCI_ZIP_URLS = [
    "https://archive.ics.uci.edu/static/public/320/student.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
]


def _read_math_from_zip(archive):
    names = archive.namelist()
    if "student-mat.csv" not in names:
        inner = [n for n in names if n.endswith(".zip")][0]
        archive = zipfile.ZipFile(io.BytesIO(archive.read(inner)))
    with archive.open("student-mat.csv") as f:
        return pd.read_csv(f, sep=";")


def load_math_data():
    for url in UCI_ZIP_URLS:
        try:
            # A browser-like User-Agent avoids being blocked by the server
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                archive = zipfile.ZipFile(io.BytesIO(resp.read()))
            return _read_math_from_zip(archive)
        except Exception as err:
            print("WARNING: could not load Math data from", url, "->", err)
    print("Falling back to ucimlrepo (649 rows = Portuguese course).")
    from ucimlrepo import fetch_ucirepo
    student = fetch_ucirepo(id=320)
    return pd.concat([student.data.features, student.data.targets], axis=1)


df = load_math_data()
df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())
assert {"G1", "G2", "G3"}.issubset(df.columns), "G1, G2, G3 must exist"

# ---------------------------------------------------------------
# PART 2 - Preprocess, train, evaluate (Activity B)
# ---------------------------------------------------------------
# Target: 1 = at risk (G3 < 10); 0 = not at risk
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor
X = df.drop(columns=["G3"])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
))

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl")
