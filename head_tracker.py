import numpy as np
import cv2


class HeadTracker:
    def __init__(self):
        # Parámetros para detección de esquinas (Shi-Tomasi)
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

        # Parámetros para flujo óptico (Lucas-Kanade)
        # winSize: Tamaño de la ventana de búsqueda
        # maxLevel: Niveles de pirámide (para movimientos grandes)
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Estado interno
        self.prev_gray = None  # Frame anterior (escala de grises)
        self.prev_points = None  # Puntos rastreados anteriores
        self.mask = None  # Máscara para dibujar las líneas (estela)
        self.color = np.random.randint(0, 255, (100, 3))  # Colores aleatorios

    def reset(self):
        """Reinicia el tracker (útil al iniciar un nuevo examen)"""
        self.prev_gray = None
        self.prev_points = None
        self.mask = None

    def handle_frame(self, frame):
        """
        Recibe el frame actual BGR, calcula el flujo óptico y
        devuelve el frame dibujado con el rastro del movimiento.
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Inicialización (Primer frame o si se pierden los puntos)
        if self.prev_gray is None or self.prev_points is None or len(self.prev_points) == 0:
            self.prev_gray = frame_gray
            # Detectar características iniciales para rastrear
            self.prev_points = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
            self.mask = np.zeros_like(frame)  # Lienzo limpio para dibujar líneas
            return frame

        # 2. Calcular Flujo Óptico (Lucas-Kanade)
        # Calcula la nueva posición (new_points) de los puntos anteriores
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, frame_gray, self.prev_points, None, **self.lk_params
        )

        # Seleccionar puntos buenos (status=1 significa que se encontró el punto)
        if new_points is not None:
            good_new = new_points[status == 1]
            good_old = self.prev_points[status == 1]
        else:
            # Si falla todo, reiniciamos para el siguiente frame
            self.prev_gray = frame_gray
            return frame

        # 3. Dibujar los rastros (Visualización)
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            # Dibujar línea en la máscara (estela de movimiento)
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)

            # Dibujar punto actual en el frame
            frame = cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)

        img_output = cv2.add(frame, self.mask)

        # 4. Actualizar estado para el siguiente ciclo
        self.prev_gray = frame_gray.copy()
        self.prev_points = good_new.reshape(-1, 1, 2)

        return img_output