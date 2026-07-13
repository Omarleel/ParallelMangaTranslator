# Cola profesional, control de ejecución e instalación automática

## Cambios principales

- Cola FIFO persistente en SQLite (`.pmt_ui_jobs/queue.sqlite3`) con WAL y operaciones atómicas.
- Un único worker de procesamiento para aislar configuración global, caché, logs y memoria GPU entre trabajos.
- Pausa cooperativa que libera el worker; al reanudar solo se repite la página interrumpida.
- Cancelación cooperativa que conserva las páginas terminadas y marca las restantes como canceladas.
- Recuperación de manifiestos después de cierre abrupto, incluyendo detección de estados `processing` abandonados.
- Reintentos configurables por página, con espera exponencial y contador persistente.
- Caché y log separados por trabajo.
- Endpoints y controles UI para pausar, reanudar y cancelar.
- Instalador automático `install_pmt.py`, más wrappers `.bat` y `.sh`.
- Perfiles CPU, CUDA 11.8, CUDA 12.4, CUDA 12.9, ROCm 6.4 y Apple MPS.
- Verificación posterior de PyTorch y del backend GPU; no acepta silenciosamente un perfil GPU que termine usando CPU.
- `ocr.gpu: auto` para CUDA/ROCm.

## Semántica de pausa y cancelación

El control es cooperativo. No es posible interrumpir de forma segura un kernel CUDA o una petición HTTP ya iniciados desde Python. La solicitud se aplica al siguiente checkpoint del pipeline. Una pausa no consume un reintento; una cancelación conserva los resultados completos ya escritos.

## Persistencia

Cada trabajo contiene:

- `manifest.json`: estado, páginas, intentos y opciones del trabajo.
- `.cache/`: caché privada del trabajo.
- `job.log`: log privado del trabajo.
- `outputs/`: resultados y correcciones.

SQLite solo mantiene el orden y la reserva activa de la cola. El manifiesto es la fuente de verdad del progreso de páginas.

## Instalación

```bash
python install_pmt.py --dry-run
python install_pmt.py
```

Perfiles forzados:

```bash
python install_pmt.py --profile cu129
python install_pmt.py --profile cu124
python install_pmt.py --profile cu118
python install_pmt.py --profile rocm64
python install_pmt.py --profile mps
python install_pmt.py --profile cpu
```

## Pruebas añadidas

Las pruebas cubren persistencia FIFO, recuperación de filas `processing`, pausa/cancelación cooperativa, liberación del worker al pausar, recuperación de páginas completas e incompletas, reintentos transitorios, selección del modelo de inpainting y perfiles de hardware.
