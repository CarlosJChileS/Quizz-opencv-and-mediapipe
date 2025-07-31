import cv2
import numpy as np
import time
from quizz_images import QUIZZ
from utils import detectar_respuesta_por_rostro, mostrar_campeones_con_foto

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def letterbox(im, box_size=(560, 720), color=(30, 30, 30)):
    h, w = im.shape[:2]
    box_w, box_h = box_size
    scale = min(box_w / w, box_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    im_resized = cv2.resize(im, (new_w, new_h))
    canvas = np.full((box_h, box_w, 3), color, dtype=np.uint8)
    x0 = (box_w - new_w) // 2
    y0 = (box_h - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = im_resized
    return canvas

def mejorar_contraste_y_brillo(img, alpha=1.35, beta=40):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def ejecutar_quizz():
    puntaje_jugador = {}
    color_jugador = {}

    ancho_total = 1600
    alto_total = 720
    ancho_img_q = int(0.35 * ancho_total)      # 560 px
    ancho_vis_area = int(0.65 * ancho_total)   # 1040 px
    ancho_sep = 20

    for idx_preg, pregunta in enumerate(QUIZZ):
        tiempo_inicio = time.time()
        tiempo_max = 8

        respuestas_por_jugador = {}
        colores_por_id = {}

        img_q0 = cv2.imread(pregunta["image"])
        if img_q0 is None:
            print(f"No se pudo cargar la imagen {pregunta['image']}")
            continue
        img_q = letterbox(img_q0, box_size=(ancho_img_q, alto_total), color=(30,30,30))

        vis_width = ancho_vis_area
        vis_height = alto_total

        while time.time() - tiempo_inicio < tiempo_max:
            success, img = cap.read()
            if not success:
                break
            img = cv2.flip(img, 1)
            img = cv2.resize(img, (vis_width, vis_height))
            img = mejorar_contraste_y_brillo(img)

            vis_area = img.copy()
            respuestas, vis_area, colores_por_id = detectar_respuesta_por_rostro(vis_area)

            cv2.putText(vis_area, f"IMAGEN {idx_preg + 1} de {len(QUIZZ)}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 0), 4)
            cv2.putText(vis_area, "¿Hecha por IA?", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
            cv2.putText(vis_area, "Izquierda=IA | Derecha=REAL" , (30, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

            elapsed = time.time() - tiempo_inicio
            progreso = min(elapsed / tiempo_max, 1.0)
            bar_width = int((1 - progreso) * (vis_width - 100))
            cv2.rectangle(vis_area, (50, 20), (vis_width - 50, 40), (80, 80, 80), -1)
            color_barra = (244, 180, 0)
            cv2.rectangle(vis_area, (50, 20), (50 + bar_width, 40), color_barra, -1)
            cv2.rectangle(vis_area, (50, 20), (vis_width - 50, 40), (255, 255, 255), 2)

            real_count = list(respuestas.values()).count("REAL")
            ia_count = list(respuestas.values()).count("IA")
            cv2.putText(vis_area, f"REAL: {real_count}", (60, vis_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4)
            cv2.putText(vis_area, f"IA: {ia_count}", (vis_width - 300, vis_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)

            separador = np.ones((alto_total, ancho_sep, 3), dtype=np.uint8) * 220
            if img_q.shape[0] != vis_area.shape[0]:
                img_q = cv2.resize(img_q, (img_q.shape[1], vis_area.shape[0]))
            combined = np.hstack((img_q, separador, vis_area))

            cv2.namedWindow("QUIZZ VISUAL", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("QUIZZ VISUAL", ancho_total, alto_total)
            cv2.imshow("QUIZZ VISUAL", combined)
            if cv2.waitKey(1) & 0xFF == 27:
                return

            respuestas_por_jugador = respuestas
            color_jugador.update(colores_por_id)

        # Suma puntaje final de cada jugador
        correcta = pregunta["answer"]
        for jid in color_jugador:
            puntaje_jugador.setdefault(jid, 0)
        for jid, resp in respuestas_por_jugador.items():
            if resp == correcta:
                puntaje_jugador[jid] += 1

    # --- CELEBRACIÓN DE GANADORES (SOPORTA EMPATES) ---
    if puntaje_jugador:
        max_score = max(puntaje_jugador.values())
        ganadores = [jid for jid, score in puntaje_jugador.items() if score == max_score and max_score > 0]
        if ganadores:
            rostros = []
            colores = []
            try:
                _, frame = cap.read()
                import mediapipe as mp
                mp_face = mp.solutions.face_detection
                face_detection = mp_face.FaceDetection(min_detection_confidence=0.55)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_face = face_detection.process(img_rgb)
                h, w, _ = frame.shape
                rostros_en_frame = []
                if results_face.detections:
                    for i, detection in enumerate(results_face.detections):
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        ancho = int(bbox.width * w)
                        alto = int(bbox.height * h)
                        if ancho < 70 or alto < 70:
                            continue
                        rostros_en_frame.append(frame[max(y,0):max(y,0)+alto, max(x,0):max(x,0)+ancho])
                for idx, jid in enumerate(ganadores):
                    color = color_jugador.get(jid, (240, 240, 0))
                    colores.append(color)
                    if idx < len(rostros_en_frame):
                        rostros.append(rostros_en_frame[idx])
                    else:
                        rostros.append(None)
            except Exception as e:
                print("No se pudo obtener los rostros:", e)
                rostros = [None]*len(ganadores)
                colores = [(240,240,0)]*len(ganadores)
            from utils import mostrar_campeones_con_foto
            mostrar_campeones_con_foto(ganadores, colores, rostros)

while True:
    ejecutar_quizz()
    print("\n¿Repetir juego? (s/n): ", end="")
    if input().lower() != "s":
        break

cap.release()
cv2.destroyAllWindows()
