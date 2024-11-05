import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image

# Function to visualize images from the dataset
def visualize_images(base_path):
    categories = ['healthy', 'parkinson']
    for category in categories:
        plt.figure(figsize=(10, 10))
        image_files = os.listdir(os.path.join(base_path, category))
        for i in range(min(16, len(image_files))):  # Display 16 images
            plt.subplot(4, 4, i + 1)  # 4 rows, 4 columns
            img_path = os.path.join(base_path, category, image_files[i])
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title(category)
        plt.suptitle(f"Images of {category.capitalize()}")
        plt.show()

# Visualize training images
visualize_images("drawings/images/training")

# Function to load data
def load_data(base_path):
    X = []
    y = []
    categories = ['healthy', 'parkinson']
    
    for category in categories:
        category_path = os.path.join(base_path, category)
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize((64, 64))  # Resize to a fixed size
            img_array = np.array(img).flatten()  # Flatten the image to a vector
            X.append(img_array)
            y.append(0 if category == 'healthy' else 1)  # Label: 0 for healthy, 1 for parkinson
    
    return np.array(X), np.array(y)

# Load training data
X_train, y_train = load_data('drawings/images/training')

# Load test data
X_test, y_test = load_data('drawings/images/testing')

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Initialize and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Initialize and train the AdaBoost model
adaboost = AdaBoostClassifier(algorithm='SAMME', random_state=42)
adaboost.fit(X_train, y_train)

# Make predictions with each model on the test dataset
rf_predictions = rf_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)
ada_predictions = adaboost.predict(X_test)

# Calculate accuracy for each model
rf_test_accuracy = accuracy_score(y_test, rf_predictions)
dt_test_accuracy = accuracy_score(y_test, dt_predictions)
ada_test_accuracy = accuracy_score(y_test, ada_predictions)

print(f"Random Forest Test Accuracy: {rf_test_accuracy:.4f}")
print(f"Decision Tree Test Accuracy: {dt_test_accuracy:.4f}")
print(f"AdaBoost Test Accuracy: {ada_test_accuracy:.4f}")

# Compare the models
model_names = ['Random Forest', 'Decision Tree', 'AdaBoost']
test_accuracies = [rf_test_accuracy, dt_test_accuracy, ada_test_accuracy]

plt.bar(model_names, test_accuracies, color=['blue', 'orange', 'green'])
plt.ylabel('Test Accuracy')
plt.title('Model Test Accuracy Comparison')
plt.ylim(0, 1)
plt.show()
