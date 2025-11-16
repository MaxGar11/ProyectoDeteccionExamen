import cv2


class CaptureManager:
    """
    Componente que encapsula el manejo de la captura de video
    y la ventana de visualización de OpenCV.
    """

    def __init__(self, camera_id=0, width=640, height=480):
        self.window_name = 'Monitoreo de Atencion'
        self.cap = cv2.VideoCapture(camera_id)

        if self.cap.isOpened():
            # Configura la resolución deseada
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"Cámara {camera_id} abierta con éxito.")
        else:
            print(f"Error: No se pudo abrir la cámara con ID {camera_id}")

    def is_opened(self):
        """Verifica si la cámara está conectada y abierta."""
        return self.cap.isOpened()

    def read_frame(self):
        """Lee un solo frame de la cámara."""
        return self.cap.read()

    def show_frame(self, frame):
        """Muestra el frame en la ventana de OpenCV."""
        cv2.imshow(self.window_name, frame)

    def check_events(self):
        """
        Captura los eventos de teclado de la ventana.
        Espera 10ms por una tecla.
        """
        return cv2.waitKey(10) & 0xFF

    def release(self):
        """Libera la cámara y cierra todas las ventanas."""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Recursos de cámara liberados.")