import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from capture_manager import CaptureManager
from head_tracker import HeadTracker


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("EyeProctor - Sistema de Monitoreo")
        self.root.geometry("800x650")
        self.root.configure(bg="#f0f0f0")

        self.examen_activo = False

        # Inicializamos Managers y Detectores
        self.cap_manager = CaptureManager()
        self.tracker = HeadTracker()  # Instancia del tracker

        self._setup_ui()
        self._update_video_loop()

    def _setup_ui(self):
        self.header_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        self.header_frame.pack(fill=tk.X)
        self.lbl_status = tk.Label(self.header_frame, text="ESPERANDO INICIO",
                                   font=("Arial", 16, "bold"), fg="white", bg="#2c3e50")
        self.lbl_status.pack(pady=10)

        # --- Video ---
        self.video_frame = tk.Frame(self.root, bg="black", width=640, height=480)
        self.video_frame.pack(pady=20)
        self.lbl_video = tk.Label(self.video_frame, bg="black")
        self.lbl_video.pack()

        # --- Botones ---
        self.controls_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.controls_frame.pack(pady=10)
        btn_style = {"font": ("Arial", 12), "width": 15, "pady": 5}

        self.btn_iniciar = tk.Button(self.controls_frame, text="▶ Iniciar",
                                     bg="#27ae60", fg="white", command=self.iniciar_examen, **btn_style)
        self.btn_iniciar.grid(row=0, column=0, padx=10)

        self.btn_detener = tk.Button(self.controls_frame, text="⏹ Detener",
                                     bg="#c0392b", fg="white", command=self.detener_examen,
                                     state=tk.DISABLED, **btn_style)
        self.btn_detener.grid(row=0, column=1, padx=10)

        self.btn_salir = tk.Button(self.controls_frame, text="Salir",
                                   bg="#7f8c8d", fg="white", command=self.salir, **btn_style)
        self.btn_salir.grid(row=0, column=2, padx=10)

    def _update_video_loop(self):
        """Bucle principal de procesamiento de video"""
        ret, frame = self.cap_manager.read_frame()

        if ret:
            # Espejo (opcional, para que se sienta más natural)
            frame = cv2.flip(frame, 1)

            # --- LÓGICA DE VISIÓN POR COMPUTADORA ---
            if self.examen_activo:
                # 1. Pasamos el frame al tracker para que dibuje el flujo óptico
                frame = self.tracker.handle_frame(frame)

                # 2. Indicador de grabación
                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (45, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)


            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.lbl_video.imgtk = img_tk
            self.lbl_video.configure(image=img_tk)

        self.root.after(10, self._update_video_loop)

    def iniciar_examen(self):
        self.examen_activo = True
        self.tracker.reset()  # Reiniciar puntos de rastreo
        self.lbl_status.config(text="EXAMEN EN CURSO - RASTREANDO", fg="#e74c3c")
        self.btn_iniciar.config(state=tk.DISABLED)
        self.btn_detener.config(state=tk.NORMAL)

    def detener_examen(self):
        self.examen_activo = False
        self.lbl_status.config(text="EXAMEN FINALIZADO", fg="#2ecc71")
        self.btn_iniciar.config(state=tk.NORMAL)
        self.btn_detener.config(state=tk.DISABLED)

    def salir(self):
        self.cap_manager.release()
        self.root.quit()