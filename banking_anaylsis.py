import pandas as pd

data = pd.read_csv("bank.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

data.drop(['age', 'contact', 'day_of_week'], axis=1, inplace=True)

binary_map = {'yes': 1, 'no': 0}
data['housing'] = data['housing'].map(binary_map)
data['loan'] = data['loan'].map(binary_map)
data['y'] = data['y'].map(binary_map)

data['marital'] = data['marital'].astype('category').cat.codes
data['job'] = data['job'].astype('category').cat.codes
data['education'] = data['education'].astype('category').cat.codes

X = data.drop('y', axis=1)
y = data['y']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

for result in y_pred:
    if result == 1:
        print("This customer will purchase the product")
    else:
        print("This customer will not purchase the product")
