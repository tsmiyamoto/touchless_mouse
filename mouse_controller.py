import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def main():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    screen_width, screen_height = pyautogui.size()

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            results = hands.process(frame)

            # Draw the hand annotations on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # 人差し指(指先)の座標値(x,y)をカメラでキャプチャした画像に合わせる
                    index_finger_tip_x = int(hand_landmarks.landmark[8].x * screen_width)
                    index_finger_tip_y = int(hand_landmarks.landmark[8].y * screen_height)

                    # 人差し指(第二関節)の座標値(x,y)をカメラでキャプチャした画像に合わせる
                    index_finger_pip_x = int(hand_landmarks.landmark[6].x * screen_width)
                    index_finger_pip_y = int(hand_landmarks.landmark[6].y * screen_height)

                    # 人差し指を曲げたとき、ダブルクリックをする
                    if index_finger_tip_y > index_finger_pip_y:
                        pyautogui.doubleClick(index_finger_pip_x, index_finger_pip_y)

                    # 上記外は、カーソルを移動させる
                    else:
                        print(index_finger_pip_x, index_finger_pip_y)
                        pyautogui.moveTo(index_finger_pip_x, index_finger_pip_y)

            cv2.imshow("Hand Detection", frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()


if __name__ == "__main__":
    main()
