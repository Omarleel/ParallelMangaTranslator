import os
import torch.multiprocessing as mp
from Applications.JsonGenerator import JsonWriter
from Applications.Utilities import Utilities

class ParallelProcessor:
    def __init__(self):
        self.utilities = Utilities()
    
    def procesar_en_paralelo(self, ruta_carpeta_entrada, ruta_carpeta_salida, process_func, batch_size = 4):
        try:
            archivos = os.listdir(ruta_carpeta_entrada)
            lista_imagenes = [
                archivo for archivo in archivos if archivo.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
            ]
            cantidad_archivos = len(lista_imagenes)
            # Dividir las imágenes en lotes para procesamiento paralelo
            lotes_imagenes =  [lista_imagenes[i:i+batch_size] for i in range(0, cantidad_archivos, batch_size)]
            num_processes = len(lotes_imagenes)
            lotes_imagenes = self.utilities.convertir_a_diccionarios(lotes_imagenes)
            ruta_limpieza_salida = os.path.join(ruta_carpeta_salida, "Limpieza")
            ruta_traduccion_salida = os.path.join(ruta_carpeta_salida, "Traduccion")

            # Inicializar las colas y el proceso para escribir JSON
            transcripcion_queue = mp.Queue()
            traduccion_queue = mp.Queue()
            transcripcion_process = JsonWriter(transcripcion_queue)
            traduccion_process = JsonWriter(traduccion_queue)
            transcripcion_process.start()
            traduccion_process.start()

            transcripcion_queue.put({
                'agregar_entrada' : {
                    'Título' : os.path.basename(ruta_carpeta_entrada),
                    'Páginas': cantidad_archivos,
                }
            })
            traduccion_queue.put({
                'agregar_entrada' : {
                    'Título' : os.path.basename(ruta_carpeta_entrada),
                    'Páginas': cantidad_archivos,
                }
            })
            
            processes = []
            for i in range(num_processes):
                # Seleccionar un lote de imágenes para cada proceso
                lote = lotes_imagenes[i]
                p = mp.Process(
                    target=process_func,
                    args=(
                        ruta_carpeta_entrada,
                        ruta_limpieza_salida,
                        ruta_traduccion_salida,
                        lote,
                        transcripcion_queue,
                        traduccion_queue,
                    ),
                    daemon=True)
                p.start()
                processes.append(p)
            
            for p in processes:
                p.join()

            transcripcion_queue.put({
                'ordenar_por_paginas' : {
                    'tipo' : 'Transcripción',
                }
            })
            traduccion_queue.put({
                'ordenar_por_paginas' : {
                    'tipo' : 'Traducción',
                }
            })
            # Señalizar que la escritura de JSON está completa
            transcripcion_queue.put({
                'guardar_en_archivo' : os.path.join(ruta_limpieza_salida, "Transcripción.json")
            })
            traduccion_queue.put({
                'guardar_en_archivo' : os.path.join(ruta_traduccion_salida, "Traducción.json")
            })

            # Esperar a que los procesos de escritura de JSON terminen
            transcripcion_process.join()
            traduccion_process.join()

            return True
        except Exception as e:
            # Captura cualquier excepción y la imprime
            print(f"Error al procesar imágenes en paralelo: {e}")
            return False