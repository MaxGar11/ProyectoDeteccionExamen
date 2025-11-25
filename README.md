# Monitoreo de Atención durante Exámenes en Línea

Proyecto de una aplicación de visión por computadora que utiliza la cámara del dispositivo para monitorear la atención del alumno durante la aplicación de un examen, con el objetivo de mejorar la confiabilidad de los instrumentos de evaluación en línea.

- Materia: Visión y Animación por Computadora
- Universidad: Benemérita Universidad Autónoma de Puebla (BUAP)

## Nombres 
- Nombre 1: Maximiliano García Rivera   
- Nombre 2: Juan Felipe Pérez Roldán

## Objetivo del Proyecto
Detectar periodos de atención y no atención del alumno durante un examen en línea. Se considera comportamiento sospechoso si el alumno pasa más del 40% del tiempo del examen sin ver la pantalla del examen. Además, la herramienta debe contabilizar cambios de ventana (vía mouse o teclado) como tiempo de no atención.

## Requisitos del Sistema
- SO: Windows 10/11.
- Cámara web funcional.
- Python 3.8+ (recomendado 3.10 o superior).

## Instalación (Windows / PowerShell)
Se recomienda usar un entorno virtual.

```powershell
# 1) Ir al directorio del proyecto
cd "\ProyectoDeteccionExamen"

# 2) Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Instalar dependencias actuales
pip install opencv-python numpy
```

## Uso
Ejecuta la aplicación desde PowerShell con el entorno virtual activo:

```powershell
python .\main.py
```

Controles dentro de la ventana de la cámara:
- `I`: Iniciar examen (empieza el monitoreo).
- `P`: Parar examen (finaliza el monitoreo actual).
- `ESC`: Salir de la aplicación.

Nota: En la versión actual, el control es por teclado (simula el botón Iniciar/Parar). Una futura versión integrará UI más explícita con botones en pantalla o interfaz gráfica separada.




