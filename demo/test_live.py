from facex import PoolManager
import cv2

def main():
    # Initialize PoolManager with 1 worker for simplicity
    pool_manager = PoolManager(pool_size=1)

    # Start video capture from the webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam stream. Press 'q' to quit.")

    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting.")
                break

            # Predict emotions
            try:
                predictions = pool_manager.predict(frame)
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue

            # Display predictions on the frame
            for prediction in predictions:
                x, y, w, h = prediction['bbox']
                emotion = prediction['emot']
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                em = max(emotion, key=emotion.get)

       
                # Add emotion label
                cv2.putText(frame, em, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Show the frame with predictions
            cv2.imshow('Emotion Detection', frame)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
        pool_manager.shutdown()
        print("Webcam stream ended.")

if __name__ == "__main__":
    main()
