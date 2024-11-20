import sys
import os
import pytest
import numpy as np
import logging

# Add the parent directory of facex to sys.path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import facex

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@pytest.fixture
def pool_manager():
    # Initialize a PoolManager instance with a small pool size for testing
    return facex.PoolManager(pool_size=3)


@pytest.fixture
def input_data():
    # Generate dummy input data for testing
    val = np.random.random((480, 640, 3)).astype(np.float32)
    return (val * 255).astype(np.uint8)


def test_multi_concurrency(pool_manager, input_data):
    """
    Test concurrent predictions using multiple threads.
    """
    try:
        futures = [pool_manager.predict(input_data) for _ in range(3)]
    except Exception as e:
        logging.error(f"Error during concurrency test: {e}")
        pytest.fail(f"Concurrency test failed due to: {e}")

    # Ensure predictions are returned for all requests
    assert len(futures) == 3
    for result in futures:
        assert isinstance(result, list)  # Assuming parse_result returns a dict


def test_dedicated_worker(pool_manager, input_data):
    """
    Test assigning and using a dedicated worker.
    """
    user_id = "user_123"
    
    try:
        worker = pool_manager.get_worker(user_id)
    except RuntimeError as e:
        logging.error(f"Failed to allocate dedicated worker: {e}")
        pytest.fail(f"Dedicated worker allocation failed due to: {e}")
    
    # Ensure the worker is allocated
    assert worker is not None
    assert isinstance(worker, facex.EmotionClassifier)

    # Make predictions using the dedicated worker
    try:
        prediction = worker.predict(input_data)
        assert isinstance(prediction, list)  # Assuming parse_result returns a dict
    except Exception as e:
        logging.error(f"Prediction failed with dedicated worker: {e}")
        pytest.fail(f"Prediction failed: {e}")

    # Release the worker and ensure it is added back to the pool
    try:
        pool_manager.release_worker(user_id)
        assert len(pool_manager.pool) == 3  # Ensure the pool size is correct after release
    except KeyError as e:
        logging.error(f"Failed to release dedicated worker: {e}")
        pytest.fail(f"Release of dedicated worker failed due to: {e}")


