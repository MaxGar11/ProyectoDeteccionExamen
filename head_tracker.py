import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from collections import namedtuple, deque
import math

class HeadTracker:
    def __init__(self):
        # Parámetros para detección de esquinas (Shi-Tomasi)
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.2,
                                   minDistance=3,
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

        # Parámetros de Background Subtractor (MOG2) ajustados para mantener objetos casi estáticos (rostro) como foreground más tiempo
        # history grande => modelo se adapta más lento
        self.bg_history = 400
        # varThreshold más bajo => más sensibilidad a pequeñas variaciones (evita que se absorba rápidamente)
        self.bg_var_threshold = 20
        # learningRate fijo más pequeño que 1/history (0.001 < 1/400=0.0025) => ralentiza la incorporación al fondo
        self.bg_learning_rate = 0.001
        self.bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=self.bg_history,
                                       varThreshold=self.bg_var_threshold,
                                       detectShadows=False)

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
        frame = self.extract_skin(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Sustracción de fondo (para resaltar movimiento). Devuelve máscara binaria, la reutilizamos como frame de trabajo.
        #frame_gray = self.bgSubtractor.apply(frame_gray, learningRate=self.bg_learning_rate)
        #Frame difference
        #frame_gray = self.frame_diff(frame_gray)

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

    def background_subtractor(self, frame):
        """Aplica la sustracción de fondo sobre el frame en escala de grises y retorna la máscara."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.bgSubtractor.apply(gray, learningRate=self.bg_learning_rate)

    def frame_diff(self, frame):
        prev_frame = frame
        cur_frame = frame
        next_frame = frame

        diff_frames1 = cv2.absdiff(next_frame, cur_frame)
        diff_frames2 = cv2.absdiff(cur_frame, prev_frame)

        prev_frame = cur_frame
        cur_frame = next_frame
        next_frame = frame

        return cv2.bitwise_and(diff_frames1, diff_frames2)

    def extract_skin(self, frame):
        """Recibe un frame BGR y devuelve el mismo frame con solo las zonas de piel visibles.

        Pasos:
        1. Convertir a HSV.
        2. Generar dos rangos de máscara para tonos de piel (para cubrir variaciones y posible envoltura del tono).
        3. Unir máscaras y limpiar ruido con operaciones morfológicas y blur.
        4. Aplicar la máscara sobre el frame original.

        Retorna: frame BGR con fondo negro donde no hay piel.
        """
        # Convertir a HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rangos típicos de piel en HSV (pueden ajustarse según iluminación/cámara)
        # Primer rango: tonos más rosados / rojizos
        lower1 = np.array([0, 40, 60], dtype=np.uint8)
        upper1 = np.array([15, 255, 255], dtype=np.uint8)
        # Segundo rango: tonos más hacia naranja/amarillo
        lower2 = np.array([15, 40, 60], dtype=np.uint8)
        upper2 = np.array([35, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        skin_mask = cv2.bitwise_or(mask1, mask2)

        # Reducir ruido: apertura morfológica seguida de ligera dilatación para rellenar huecos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

        # Suavizar bordes
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

        # Aplicar máscara sobre el frame original
        skin = cv2.bitwise_and(frame, frame, mask=skin_mask)
        return skin