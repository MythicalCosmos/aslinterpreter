import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():
    model_path = r"C:\Users\360mc\Stuff\Desktop\SoftwareDev\gesture_recognizer.task"

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = recognizer.recognize(mp_image)

        if result.gestures:
            top_gesture = result.gestures[0][0]
            gesture_name = top_gesture.category_name
            score = top_gesture.score
            cv2.putText(frame, f"{gesture_name} ({score:.2f})", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
