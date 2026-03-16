import cv2
import time
from ultralytics import YOLO

# carregar modelo
model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(0)

fall_start_time = None
fall_detected = False

while True:

    ret, frame = camera.read()
    if not ret:
        break

    results = model(frame)

    person_found = False

    for result in results:
        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            # classe 0 = pessoa
            if cls == 0:

                person_found = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                w = x2 - x1
                h = y2 - y1

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                # verificar se corpo está horizontal
                if w > h:

                    if fall_start_time is None:
                        fall_start_time = time.time()

                    fall_duration = time.time() - fall_start_time

                    if fall_duration > 3:
                        fall_detected = True

                else:
                    fall_start_time = None
                    fall_detected = False

    if fall_detected:
        cv2.putText(
            frame,
            "QUEDA DETECTADA",
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0,0,255),
            3
        )

    cv2.imshow("LifeCare Fall Detector", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()