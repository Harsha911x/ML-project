import pickle
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Ensure all data entries are of the same length
data_lengths = [len(d) for d in data]
if len(set(data_lengths)) > 1:
    print("Data entries have inconsistent lengths. Filtering out inconsistent entries.")
    # Find the most common length
    common_length = Counter(data_lengths).most_common(1)[0][0]
    # Filter out entries that do not match the common length
    filtered_data_labels = [(d, labels[i]) for i, d in enumerate(data) if len(d) == common_length]
    data, labels = zip(*filtered_data_labels)
    data = list(data)
    labels = list(labels)

# Convert to NumPy arrays
data = np.asarray(data)
labels = np.asarray(labels)

# Check the distribution of samples per class
class_counts = Counter(labels)
print("Class distribution:", class_counts)

# Find classes with fewer than 2 samples
classes_with_few_samples = [cls for cls, count in class_counts.items() if count < 2]

if classes_with_few_samples:
    print(f"Classes with fewer than 2 samples: {classes_with_few_samples}")

    # Manually duplicate samples for minority classes
    for cls in classes_with_few_samples:
        indices = [i for i, label in enumerate(labels) if label == cls]
        for i in indices:
            data = np.vstack([data, data[i]])
            labels = np.append(labels, labels[i])

    # Check class distribution after manual duplication
    class_counts = Counter(labels)
    print("Class distribution after manual duplication:", class_counts)

# Ensure there are no classes with fewer than 2 samples
class_counts = Counter(labels)
print("Updated class distribution:", class_counts)

# Check if there are at least two classes
if len(class_counts) < 2:
    print("Not enough classes to train the model. At least two classes are required.")
else:
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    # Train the model with Gradient Boosting Classifier
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)

    # Predict and evaluate the model
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print('{}% of samples were classified correctly!'.format(score * 100))

    # Save the model
    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
