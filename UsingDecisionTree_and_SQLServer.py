import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from FromScratchDecisionTree import FromScratchDecisionTree, evaluate_model

#python UsingDecisionTree_and_SQLServer.py

# --- 1. Load data from SQL Server ---
query = """ SELECT CASE WHEN [Gender ID] = 'M' Then 01
                        ELSE 00 END AS IsMale
            , CASE WHEN [Personnel Level ID] IN (03, 04) Then 01
                        ELSE 00 END AS IsUpperLevel
            , CASE WHEN Tenure > 25 Then 01
                        ELSE 00 END AS TenureAbove25Years
            , CASE WHEN Age > 35 Then 01
                        ELSE 00 END AS AgeAbove35Years
            , CASE WHEN [Education Level ID] IN (4, 5, 6) THEN 01
                        ELSE 00 END AS HighEducationLevel
            , CASE WHEN [Status ID] = 00 THEN 00 
                        ELSE 01 END AS LeftCompany 
            FROM DBHR.dbo.v_personnel
            WHERE split = 00
        """ 

server = 'LAPTOP-U55QS3H6\LIVE'
database = 'DBHR'
conn = pyodbc.connect(
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={server};'
    f'DATABASE={database};'
    'Trusted_Connection=yes;'
)
df = pd.read_sql(query, conn)
conn.close()
print("Data loaded:", df.shape)

TEST_SIZE = 0.3    
X = df[["IsMale", "IsUpperLevel", "TenureAbove25Years", "AgeAbove35Years", "HighEducationLevel"]]
y = df["LeftCompany"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=42
)
train_data = X_train.copy()
train_data["LeftCompany"] = y_train
test_data  = X_test.copy()
test_data["LeftCompany"] = y_test

tree = FromScratchDecisionTree(max_depth=8, min_gain=0.001, depth=1)
tree.fit(train_data, "LeftCompany")
y_pred_temp = tree.predict(X_test)
y_pred = [1 if prob[1] < 0.4 else 0 for prob in y_pred_temp]

# print("y_test:", y_test)
# print("y_pred_temp:", y_pred_temp)
# print("y_pred:", y_pred)
# print("Average y_pred_temp:", y_pred_temp[:,1].mean())

accuracy_pred = accuracy_score(y_test, y_pred)
print("Custom Tree Accuracy:", accuracy_pred)
print("Custom Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# evaluate_model(
#     tree,
#     test_data.drop("LeftCompany", axis=1),
#     test_data["LeftCompany"]
# )

clf = DecisionTreeClassifier(
    criterion='gini',  
    max_depth=8,
    random_state=42,
    class_weight="balanced"  # optional: adjust if classes are imbalanced
)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

# print("y_test:", y_test)
# print("preds:", preds)

accuracy = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)
report = classification_report(y_test, preds)

print("Scikit-learn Accuracy:", accuracy)
print("Scikit-learn Confusion Matrix:\n", cm)