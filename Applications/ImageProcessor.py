import logging
import time
import cv2
import os
import numpy as np
import torch
from Applications.Utilities import Utilities
from Applications.CleanManga import CleanManga
from Applications.TranslateManga import TranslateManga
from Applications.FileManager import FileManager

logger = logging.getLogger('debug')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('debug.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

class ImageProcessor:
    def __init__(self, idioma_entrada, idioma_salida, modelo_inpaint, metodo_traduccion="Tradicional", groq_api_key="", lore_manga=""):
        self.file_manager = FileManager()
        self.utilities =Utilities()
        self.clean_manga = CleanManga(modelo_inpaint)
        self.translate_manga = TranslateManga(
            idioma_entrada, 
            idioma_salida, 
            metodo_traduccion=metodo_traduccion, 
            groq_api_key=groq_api_key,
            lore_manga=lore_manga,
        )

    def procesar(self, ruta_carpeta_entrada, ruta_limpieza_salida, ruta_traduccion_salida, lote, transcripcion_queue, traduccion_queue):
        for indice_imagen, archivo in lote.items():
            logger.debug(f"Procesando: {archivo}")
            with open(os.path.join(ruta_carpeta_entrada, archivo), 'rb') as f:
                byte_array = f.read()
            # Convertir a numpy array
            image_nparr = np.frombuffer(byte_array, np.uint8)
            imagen = cv2.imdecode(image_nparr, cv2.IMREAD_COLOR)
            memoria_suficiente = False
            intentos = 0
            while not memoria_suficiente:
                try:
                    # Limpieza
                    transcripcion_queue.put({
                        'agregar_elemento_a_lista': {
                            'Transcripción': {
                                'Página': indice_imagen + 1,
                                'Formato': self.obtener_formato_manga(imagen),
                                'Globos de texto': []
                            }
                        }
                    })
                    mascara_capa, imagen_limpia = self.clean_manga.limpiar_manga(imagen)
                    archivo_limpieza_salida = os.path.join(ruta_limpieza_salida, archivo)
                    cv2.imwrite(archivo_limpieza_salida, imagen_limpia)
                    # Traducción
                    traduccion_queue.put({
                        'agregar_elemento_a_lista': {
                            'Traducción': {
                                'Página': indice_imagen + 1,
                                'Formato': self.obtener_formato_manga(imagen),
                                'Globos de texto': []
                            }
                        }
                    })
                    self.translate_manga.insertar_json_queue(
                        indice_imagen=indice_imagen,
                        transcripcion_queue=transcripcion_queue,
                        traduccion_queue=traduccion_queue
                    )
                    imagen_traducida = self.translate_manga.traducir_manga(imagen, imagen_limpia, mascara_capa)
                    archivo_traduccion_salida = os.path.join(ruta_traduccion_salida, archivo)
                    cv2.imwrite(archivo_traduccion_salida, imagen_traducida)
                    memoria_suficiente = True
                except (torch.cuda.CudaError, RuntimeError) as e:
                    logger.error(f"Error al procesar el archivo {archivo}: {e}")
                    if intentos < 3:
                        imagen = self.reducir_imagen(imagen)
                    torch.cuda.empty_cache()
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error al procesar el archivo {archivo}: {e}")
                finally:
                    intentos += 1
                    
            del imagen
                    
    def obtener_formato_manga(self, imagen):
        # Convierte la imagen a escala de grises
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        # Calcula el valor medio de intensidad de los píxeles en la imagen
        valor_medio = cv2.mean(imagen_gris)[0]
        # Determina el formato de la manga basado en el valor medio
        if valor_medio < 50 or valor_medio > 200:
            formato = "Blanco y negro (B/N)"
        else:
            formato = "Color"
            
        return formato
    
    def reducir_imagen(self, imagen):
        try:
            porcentaje_reduccion = 0.75
            nuevo_alto, nuevo_ancho = [int(dim * porcentaje_reduccion) for dim in imagen.shape[:2]]
            return cv2.resize(imagen, (nuevo_ancho, nuevo_alto))
        except Exception as e:
            print(f"Error al reducir la imagen: {e}")
            return imagen