import cv2
import time
import os
from ultralytics import YOLO

# garante que a pasta existe
os.makedirs("fotos", exist_ok=True)

model = YOLO("yolov8n.pt")  # ou seu modelo treinado
cap = cv2.VideoCapture(0)

estado_alerta = True
frames_sem_bone = 0
frames_com_bone = 0
LIMITE = 5
tempo_alerta = None
DURACAO = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    frame = results[0].plot()

    nomes = model.names
    tem_bone = False
    tem_sem_bone = False

    for r in results:
        for c in r.boxes.cls:
            classe = nomes[int(c)].lower()
            if classe in ["hat", "cap", "helmet", "bone"]:
                tem_bone = True
            elif classe in ["no_hat", "sem_bone"]:
                tem_sem_bone = True

    if tem_bone:
        frames_com_bone += 1
        frames_sem_bone = 0
    else:
        frames_sem_bone += 1
        frames_com_bone = 0

    if tem_sem_bone:
        estado_alerta = True
    elif frames_com_bone > LIMITE:
        estado_alerta = False
    elif frames_sem_bone > LIMITE:
        estado_alerta = True

    if estado_alerta:
        if tempo_alerta is None:
            tempo_alerta = time.time()

        restante = DURACAO - int(time.time() - tempo_alerta)

        if restante > 0:
            cv2.putText(frame, f"SEM EPI! ({restante}s)", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            nome_arquivo = f"fotos/foto_alerta_{int(time.time())}.jpg"
            cv2.imwrite(nome_arquivo, frame)
            print(f"📸 Foto salva: {nome_arquivo}")
            tempo_alerta = None
    else:
        cv2.putText(frame, "EPI OK", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        tempo_alerta = None

    cv2.imshow("Horus - Detector de EPI", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()