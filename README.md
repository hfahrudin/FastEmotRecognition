
## Facex: Lightweight, High-Performance Facial Expression Classifier

**facex** is a Python library for detecting faces and classifying emotions in images lightweight, efficient threading and object pooling for concurrent processing making it suitable for high-performance applications.

## Features
- **Face Detection**: Uses Haar cascades to detect faces in images.
- **Emotion Classification**: Predicts emotions like anger, happiness, sadness, and more.
- **Thread-Safe Pooling**: Manages multiple classifiers through a thread-safe object pool.
- **Dedicated Worker Allocation**: Assign dedicated classifiers for specific tasks or users.

## Installation

Install the library via pip:
```bash
pip install facex
```

### Prerequisites
Make sure you have the following:
- Python 3.7 or higher
- Required dependencies (installed automatically with pip):
  - `numpy`
  - `opencv-python`
  - `tensorflow`
  
Additionally, ensure the `assets` directory contains the following files:
- `haarcascade_frontalface_default.xml`
- `model_optimized.tflite`

## Usage

### Example Code
```python
import cv2
import facex

# Initialize the PoolManager
pool_manager = facex.PoolManager(pool_size=3)

# Load a test image
image = cv2.imread("test_image.jpg")

# Predict emotions
predictions = pool_manager.predict(image)

# Display results
for prediction in predictions:
    print(f"Face: {prediction['bbox']}, Emotion: {prediction['emot']}")

# Shutdown the pool manager when done
pool_manager.shutdown()
```

### Key Classes and Methods

#### `facex.EmotionClassifier`
A class responsible for detecting faces and classifying emotions in images.

- **`__init__(model_path, category=None)`**: 
  Initializes the emotion classifier with the provided TensorFlow Lite model and optional emotion categories.
  - **Arguments**:
    - `model_path` (str): Path to the pre-trained TensorFlow Lite model (`model_optimized.tflite`).
    - `category` (list, optional): List of emotion categories (default: `['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprised']`).

- **`detect_faces(input)`**: 
  Detects faces in the input image using OpenCV's Haar Cascade Classifier.
  - **Arguments**:
    - `input` (np.ndarray): The input image (e.g., a frame from a video feed).
  - **Returns**:
    - `faces` (list of tuples): Coordinates of detected faces in the form `(x, y, w, h)`.
    - `procs_input` (np.ndarray): Preprocessed grayscale version of the input image.

- **`detect_emotion(input)`**: 
  Detects the emotion of a given face using the TensorFlow Lite model.
  - **Arguments**:
    - `input` (np.ndarray): The face image (grayscale or resized) to classify.
  - **Returns**:
    - `dict`: A dictionary containing the predicted emotion with associated confidence scores.

- **`predict(input)`**: 
  Detects faces and predicts the emotions for each detected face in the image.
  - **Arguments**:
    - `input` (np.ndarray): The input image (e.g., a frame from a video feed).
  - **Returns**:
    - `list`: A list of dictionaries, each containing:
      - `'bbox'`: Bounding box of the detected face `(x, y, w, h)`.
      - `'emot'`: Emotion prediction for the face (e.g., `{'happy': 0.95, 'sadness': 0.05}`).

#### `facex.PoolManager`
A class for managing a pool of EmotionClassifier instances, enabling thread-safe predictions.

- **`__init__(pool_size=5)`**: 
  Initializes the pool with a given number of `EmotionClassifier` instances.
  - **Arguments**:
    - `pool_size` (int): Number of `EmotionClassifier` instances in the pool (default: `5`).

- **`predict(input_data)`**: 
  Retrieves an `EmotionClassifier` from the pool and uses it to make a prediction on the input data.
  - **Arguments**:
    - `input_data` (np.ndarray): The input image data (e.g., a frame from a video feed).
  - **Returns**:
    - `list`: A list of dictionaries, each containing:
      - `'bbox'`: Bounding box of the detected face `(x, y, w, h)`.
      - `'emot'`: Emotion prediction for the face.

- **`get_worker(user_id)`**: 
  Allocates a dedicated `EmotionClassifier` instance for a specific user.
  - **Arguments**:
    - `user_id` (str): The ID of the user requesting the worker.
  - **Returns**:
    - `EmotionClassifier`: The dedicated `EmotionClassifier` instance for the user.
  - **Raises**:
    - `RuntimeError`: If the user already has a dedicated worker or if no classifiers are available.

- **`release_worker(user_id)`**: 
  Releases the dedicated worker assigned to a user and returns it to the pool.
  - **Arguments**:
    - `user_id` (str): The ID of the user releasing the worker.
  - **Raises**:
    - `KeyError`: If the user does not have a dedicated worker.

- **`shutdown()`**: 
  Shuts down the pool manager and frees all resources, including threads.

## License
**facex** is licensed under the MIT License. See the `LICENSE` file for details.

## Support
If you encounter any issues, feel free to [open an issue](https://github.com/hfahrudin/facex/issues) or reach out via email.
