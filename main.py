import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")
cap = cv2.VideoCapture(0)

estado_alerta = True

frames_sem_bone = 0
frames_com_bone = 0

LIMITE = 5  # quantidade de frames pra confirmar mudança

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

    # 🔥 CONTROLE DE ESTABILIDADE
    if tem_bone:
        frames_com_bone += 1
        frames_sem_bone = 0
    else:
        frames_sem_bone += 1
        frames_com_bone = 0

    # 🔥 REGRA COM "TEMPO"
    if tem_sem_bone:
        estado_alerta = True

    elif frames_com_bone > LIMITE:
        estado_alerta = False

    elif frames_sem_bone > LIMITE:
        estado_alerta = True

    # 🔴 / 🟢 EXIBIÇÃO
    if estado_alerta:
        cv2.putText(frame, "SEM EPI!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3)
    else:
        cv2.putText(frame, "EPI OK", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 3)

    cv2.imshow("Horus - Detector de EPI", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()