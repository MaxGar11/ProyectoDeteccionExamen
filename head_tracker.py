import numpy as np
import cv2


class HeadTracker:
    def __init__(self):
        # Parámetros para detección de esquinas (Shi-Tomasi)
        self.feature_params = dict(maxCorners=20,  # Reducimos puntos para enfocarnos solo en ojos
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

        # Parámetros para flujo óptico (Lucas-Kanade)
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # --- NUEVO: Cargar clasificadores de Haar ---
        # Nota: cv2.data.haarcascades apunta a la carpeta de instalación de OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Estado interno
        self.prev_gray = None
        self.prev_points = None
        self.mask = None
        self.color = np.random.randint(0, 255, (100, 3))

    def reset(self):
        """Reinicia el tracker"""
        self.prev_gray = None
        self.prev_points = None
        self.mask = None

    def detectar_puntos_ojos(self, frame_gray):
        """
        Detecta el rostro, luego los ojos dentro del rostro,
        y encuentra puntos clave SOLO dentro de los ojos.
        """
        # 1. Detectar rostros
        faces = self.face_cascade.detectMultiScale(frame_gray, 1.3, 5)

        puntos_ojos = []

        for (x, y, w, h) in faces:
            # Region de interes (ROI) del rostro (en gris)
            roi_gray = frame_gray[y:y + h, x:x + w]

            # 2. Detectar ojos dentro del ROI del rostro
            # Limitamos la busqueda a la mitad superior del rostro para evitar falsos positivos (boca/nariz)
            roi_ojos_gray = roi_gray[0:int(h * 0.6), :]
            eyes = self.eye_cascade.detectMultiScale(roi_ojos_gray)

            for (ex, ey, ew, eh) in eyes:
                # ROI específico del ojo
                eye_roi = roi_ojos_gray[ey:ey + eh, ex:ex + ew]

                # 3. Encontrar 'good features' SOLO dentro del ojo
                p = cv2.goodFeaturesToTrack(eye_roi, mask=None, **self.feature_params)

                if p is not None:
                    # 4. Convertir coordenadas locales del ojo a globales de la imagen
                    # Coordenada Global = Coordenada Ojo + Offset Ojo (ex, ey) + Offset Cara (x, y)
                    for punto in p:
                        punto[0][0] += x + ex
                        punto[0][1] += y + ey
                        puntos_ojos.append(punto)

        if len(puntos_ojos) > 0:
            return np.array(puntos_ojos, dtype=np.float32)
        return None

    def handle_frame(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Inicialización (Si no hay puntos, buscamos ojos)
        if self.prev_gray is None or self.prev_points is None or len(self.prev_points) == 0:
            self.prev_gray = frame_gray
            self.mask = np.zeros_like(frame)

            # Usar la detección específica de ojos
            detectados = self.detectar_puntos_ojos(frame_gray)

            if detectados is not None:
                self.prev_points = detectados

            # Si no detecta ojos, retornamos el frame limpio y esperamos al siguiente ciclo
            return frame

        # 2. Calcular Flujo Óptico
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, frame_gray, self.prev_points, None, **self.lk_params
        )

        # Filtrar puntos validos
        if new_points is not None:
            good_new = new_points[status == 1]
            good_old = self.prev_points[status == 1]
        else:
            self.prev_gray = frame_gray
            return frame

        # 3. Dibujar los rastros
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 3, self.color[i].tolist(), -1)

        img_output = cv2.add(frame, self.mask)

        # 4. Actualizar estado
        self.prev_gray = frame_gray.copy()
        self.prev_points = good_new.reshape(-1, 1, 2)

        return img_output