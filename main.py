import cv2
import numpy as np
from capture_manager import CaptureManager

class EyeProctorApp:

    def __init__(self):
        print("Inicializando aplicación...")
        self.examen_activo = False

        # Abstrae el manejo de la cámara y la ventana
        self.cap_manager = CaptureManager()


        # self.tracker = HeadTracker()  (Paso 3)
        # self.logger = TimeLogger()   (Paso 4)

    def teclado_handle(self, key):
        """Maneja la entrada del teclado para controlar el estado del examen."""

        if key == ord('i') and not self.examen_activo:
            # Inicia el examen
            self.examen_activo = True
            print("--- EXAMEN INICIADO ---")

        elif key == ord('p') and self.examen_activo:
            # Detiene el examen
            self.examen_activo = False
            print("--- EXAMEN DETENIDO ---")
            return False  # Devuelve False para detener el bucle

        elif key == 27:  # Tecla ESC
            print("Saliendo de la aplicación...")
            return False  # Devuelve False para detener el bucle

        return True  # Devuelve True para continuar el bucle

    def run(self):

        if not self.cap_manager.is_opened():
            print("Error: No se pudo acceder a la cámara.")
            return

        while True:
            # 1. Capturar el frame desde el manager
            ret, frame = self.cap_manager.read_frame()
            if not ret:
                print("Error: No se pudo leer el frame.")
                break

            # Usamos una copia para dibujar sobre ella sin afectar el original
            display_frame = frame.copy()

            # 2. Lógica de procesamiento y visualización
            if self.examen_activo:


                cv2.putText(display_frame, "EXAMEN ACTIVO - Monitoreando...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Presione 'I' para Iniciar, 'P' para Parar",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 3. Mostrar el frame usando el manager
            self.cap_manager.show_frame(display_frame)

            # 4. Manejar eventos de teclado (simulando botones)
            key = self.cap_manager.check_events()
            if not self.teclado_handle(key):
                # Si _handle_input devuelve False, rompemos el bucle
                break

        self.cap_manager.release()
        print("Aplicación finalizada.")


if __name__ == '__main__':
    app = EyeProctorApp()
    app.run()