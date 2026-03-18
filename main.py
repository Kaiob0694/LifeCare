import cv2
from ultralytics import YOLO

# Carrega modelo treinado
model = YOLO("runs/detect/train3/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecta objetos
    results = model(frame, conf=0.5)

    # Desenha na tela
    frame = results[0].plot()

    cv2.imshow("Horus - Detector de EPI", frame)

    # ESC para sair
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()