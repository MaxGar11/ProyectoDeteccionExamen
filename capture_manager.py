import cv2


class CaptureManager:
    """
    Se encarga EXCLUSIVAMENTE de interactuar con el hardware de la cámara.
    """

    def __init__(self, camera_id=0, width=640, height=480):
        self.cap = cv2.VideoCapture(camera_id)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def is_opened(self):
        return self.cap.isOpened()

    def read_frame(self):
        """Retorna: (exito, frame)"""
        return self.cap.read()

    def release(self):
        if self.cap.isOpened():
            self.cap.release()