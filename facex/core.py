
import numpy as np
import pathlib
import os
from concurrent.futures import ThreadPoolExecutor
import threading
from facex.utils import create_interpreter, parse_result, input_resize

class EmotionClassifier:
    """
    A worker for predicting emotions.
    """
    def __init__(self, model_path):
        """
        Initialize the EmotionClassifier with the given model path.
        
        Args:
            model_path (str): Path to the pre-trained model.
        """
        self.model = create_interpreter(model_path)
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

    def predict(self, input):
        """
        Predict the emotion based on the input data.
        
        Args:
            input (np.ndarray): Input data for emotion classification.
        
        Returns:
            dict: Parsed result of the prediction.
        """
        # Preprocess and prepare input data
        input_data = np.array(input_resize(input), dtype=np.float32)

        # Set the input tensor for the model
        self.model.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference
        self.model.invoke()

        # Retrieve and parse the output
        output_data = self.model.get_tensor(self.output_details[0]['index'])
        return parse_result(output_data)


class PoolManager:
    """
    A thread-safe object pool for managing EmotionClassifier instances.
    """
    def __init__(self, pool_size=5):
        """
        Initialize the facex.

        Args:
            pool_size (int): Number of EmotionClassifier instances in the pool.
        """
        self.model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "model_optimized.tflite")
        self.pool_size = pool_size
        self.lock = threading.Lock()
        self.pool = [EmotionClassifier(self.model_path) for _ in range(pool_size)]
        self.executor = ThreadPoolExecutor(max_workers=pool_size)

    def get_classifier(self):
        """
        Acquire a classifier from the pool.
        
        Returns:
            EmotionClassifier: A classifier instance.
        """
        with self.lock:
            if not self.pool:
                raise RuntimeError("All EmotionClassifier instances are in use!")
            return self.pool.pop()

    def release_classifier(self, classifier):
        """
        Release a classifier back into the pool.
        
        Args:
            classifier (EmotionClassifier): The classifier to release.
        """
        with self.lock:
            self.pool.append(classifier)

    def predict(self, input_data):
        """
        Use a classifier from the pool to make a prediction in a thread-safe way.

        Args:
            input_data (np.ndarray): Input data for the classifier.

        Returns:
            dict: Parsed result of the prediction.
        """
        classifier = self.get_classifier()
        try:
            # Perform the prediction using ThreadPoolExecutor
            future = self.executor.submit(classifier.predict, input_data)
            return future.result()
        finally:
            # Ensure the classifier is returned to the pool
            self.release_classifier(classifier)

    def shutdown(self):
        """
        Clean up the executor and free resources.
        """
        self.executor.shutdown(wait=True)

