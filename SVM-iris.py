from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset
iris = datasets.load_iris()
features = iris.data
labels = iris.target

# Scale features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Create and train SVM model
clf = SVC(kernel='linear')
clf.fit(features, labels)

# ✅ Accuracy on training data
accuracy = accuracy_score(labels, clf.predict(features))
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Take user input
print("\nEnter flower measurements in cm:")
a = float(input("Sepal length: "))
b = float(input("Sepal width: "))
c = float(input("Petal length: "))
d = float(input("Petal width: "))

# Scale input and predict
user_input = scaler.transform([[a, b, c, d]])
predictions = clf.predict(user_input)
preds = int(predictions)

# Print class name
print(f"\nThe flower class is: {iris.target_names[preds]}")
