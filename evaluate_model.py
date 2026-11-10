import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

SIZE = 224
MODEL_PATH = "biomedai_cnn.keras"
TEST_DIR = "data/test"

model = tf.keras.models.load_model(MODEL_PATH)

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(SIZE, SIZE),
    color_mode="grayscale",
    batch_size=16,
    shuffle=False
)

# Match the preprocessing used during training
test_data = test_data.map(
    lambda x, y: (x / 255.0, y)
)

loss, accuracy = model.evaluate(test_data, verbose=1)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

y_true = np.concatenate([y.numpy() for x, y in test_data])
scores = model.predict(test_data, verbose=0).ravel()
y_pred = (scores >= 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Normal", "Pneumonia"],
    zero_division=0
))