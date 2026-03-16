import cv2
from ultralytics import YOLO

# carregar modelo YOLOv8
model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(0)

while True:

    ret, frame = camera.read()
    if not ret:
        break

    # rodar detecção
    results = model(frame)

    for result in results:
        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # classe 0 = pessoa
            if cls == 0:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                w = x2 - x1
                h = y2 - y1

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                cv2.putText(
                    frame,
                    f"Pessoa {conf:.2f}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

                # lógica simples de queda
                if w > h:
                    cv2.putText(
                        frame,
                        "POSSIVEL QUEDA",
                        (x1, y2+30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3
                    )

    cv2.imshow("LifeCare YOLO Detector", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()