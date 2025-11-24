import time
import os
from datetime import datetime
import numpy as np

class AttentionReporter:
    """Acumula tiempos de atención y desatención clasificando giros de cabeza.

    Categorías:
      - attention: dentro de umbrales (mirando al frente)
      - left / right: desplazamiento horizontal del centro de los puntos (ojos)
      - up / down: desplazamiento vertical del centro de los puntos
    """
    def __init__(self, threshold_x: int = 40, threshold_y: int = 30):
        self.threshold_x = threshold_x
        self.threshold_y = threshold_y
        self.baseline = None  # (x, y)
        self.current_state = None
        self.current_state_start = None
        self.state_durations = {s: 0.0 for s in ["attention", "left", "right", "up", "down"]}
        self.total_time = 0.0
        self.last_timestamp = None

    def _classify(self, center):
        if self.baseline is None:
            self.baseline = center
            return "attention"

        dx = center[0] - self.baseline[0]
        dy = center[1] - self.baseline[1]

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        # Dentro de umbrales 
        if abs_dx <= self.threshold_x and abs_dy <= self.threshold_y:
            return "attention"

        # Si se exceden ambos, elige el eje dominante
        if abs_dx > self.threshold_x and abs_dy > self.threshold_y:
            if abs_dx >= abs_dy:  # Prioriza horizontal si es mayor o igual
                return "left" if dx < 0 else "right"
            else:
                return "up" if dy < 0 else "down"

        # Sólo excede uno de los ejes
        if abs_dx > self.threshold_x:
            return "left" if dx < 0 else "right"
        if abs_dy > self.threshold_y:
            return "up" if dy < 0 else "down"

        return "attention"

    def update(self, points):
        """Actualizar estado con los puntos rastreados actuales.
        points: ndarray shape (N,1,2) o (N,2). Si None o vacío, no clasifica.
        """
        now = time.time()
        if self.last_timestamp is None:
            self.last_timestamp = now

        # Si no hay puntos, solo acumulamos tiempo total sin cambiar estado.
        if points is None or len(points) == 0:
            # Cierra estado actual hasta now (sin reclasificar)
            if self.current_state is not None:
                self.state_durations[self.current_state] += now - self.last_timestamp
                self.total_time += now - self.last_timestamp
            self.last_timestamp = now
            return

        pts = points.reshape(-1, 2)
        center = np.mean(pts, axis=0)
        new_state = self._classify(center)

        if self.current_state is None or self.current_state_start is None:
            # Primera vez o nueva sesión
            self.current_state = new_state
            self.current_state_start = now
        elif new_state != self.current_state:
            # Cambio de estado: cerrar duración anterior
            elapsed = now - self.current_state_start
            self.state_durations[self.current_state] += elapsed
            self.total_time += elapsed
            self.current_state = new_state
            self.current_state_start = now
        # else misma categoría: continuar acumulando hasta cambio

        self.last_timestamp = now

    def finalize(self, directory: str = ".", prefix: str = "Reporte"):
        """Cerrar estado y escribir reporte TXT con nombre dinámico.

        Formato: ReporteMM/DD/YYYY-(Hora).txt 
        """
        now = time.time()
        if self.current_state is not None and self.current_state_start is not None:
            elapsed = now - self.current_state_start
            self.state_durations[self.current_state] += elapsed
            self.total_time += elapsed
            self.current_state_start = None

        if self.total_time == 0:
            self.total_time = sum(self.state_durations.values())

        def fmt(sec):
            return f"{sec:.2f}s"

        lines = [
            "REPORTE DE ATENCIÓN", "-------------------", "",
            f"Tiempo total: {fmt(self.total_time)}", ""
        ]
        for state in ["attention", "left", "right", "up", "down"]:
            dur = self.state_durations[state]
            pct = (dur / self.total_time * 100.0) if self.total_time > 0 else 0.0
            lines.append(f"{state.upper():<10}: {fmt(dur)} ({pct:.1f}%)")

        # Construir nombre
        dt = datetime.now()
        filename = f"{prefix}{dt.strftime('%m-%d-%Y-(%H-%M-%S)')}.txt"
        file_path = os.path.join(directory, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return file_path

    def reset(self):
        """Resetea toda la información acumulada para iniciar un nuevo examen."""
        self.baseline = None
        self.current_state = None
        self.current_state_start = None
        self.state_durations = {s: 0.0 for s in ["attention", "left", "right", "up", "down"]}
        self.total_time = 0.0
        self.last_timestamp = None
