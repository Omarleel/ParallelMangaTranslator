import re
import cv2
import numpy as np
import torch
import logging
import transformers
from PIL import Image, ImageDraw, ImageFont
from manga_ocr import MangaOcr
from paddleocr import PaddleOCR
from Applications.TranslatorManager import TranslatorManager
from Utils.Constantes import COLOR_BLANCO, COLOR_NEGRO, FACTOR_ESPACIO, RUTA_FUENTE, TAMANIO_MINIMO_FUENTE

transformers.logging.set_verbosity_error()

class TranslateManga:
    def __init__(self, idioma_entrada, idioma_salida, metodo_traduccion="Tradicional", groq_api_key="", lore_manga=""):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.idioma_entrada = idioma_entrada
        self.idioma_salida = idioma_salida

        self.metodo_traduccion = metodo_traduccion
        
        self.translator_manager = TranslatorManager(
            idioma_entrada, 
            idioma_salida, 
            metodo=metodo_traduccion, 
            groq_api_key=groq_api_key,
            lore_manga=lore_manga,
        )
        
        self.historial_contexto = []
        
        if self.idioma_entrada == "Japonés":
            self.mocr = MangaOcr()
        else:
            langs_paddle = {
                "Inglés": "en", "Coreano": "korean", 
                "Chino": "ch", "Español": "es"
            }
            lang = langs_paddle.get(self.idioma_entrada, "en")
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    
    def insertar_json_queue(self, indice_imagen, transcripcion_queue, traduccion_queue):
        self.indice_imagen = indice_imagen
        self.transcripcion_queue = transcripcion_queue
        self.traduccion_queue = traduccion_queue
        
    def traducir_manga(self, imagen, imagen_limpia, mascara_capa):
        cuadros_delimitadores, imagenes_interes = self.obtener_areas_interes(imagen, mascara_capa)
        textos = self.obtener_textos(imagenes_interes)
        imagen_traducida = self.incrustar_textos(imagen_limpia, cuadros_delimitadores, textos)
        return imagen_traducida
    
    def obtener_areas_interes(self, imagen, mascara_capa):
        cuadros_delimitadores = []
        imagenes_interes = []
        height, width = mascara_capa.shape
        if self.idioma_entrada == "Japonés":
            kernel_h, kernel_w = round(height * 0.003215), round(width * 0.003215)
        else:
            kernel_h, kernel_w = round(height * 0.005), round(width * 0.03)
        _, mascara_binaria = cv2.threshold(mascara_capa, 127, 255, cv2.THRESH_BINARY)
        mascara_binaria = np.uint8(mascara_binaria)
        # Aplicar una operación de dilatación para fusionar áreas cercanas
        kernel = np.ones((kernel_h, kernel_w),np.uint8)  # Puedes ajustar el tamaño del kernel según tus necesidades
        mascara_dilatada = cv2.dilate(mascara_binaria, kernel, iterations=1)
        # Encuentra los contornos en la máscara de capa dilatada
        contours, _ = cv2.findContours(mascara_dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Recorre todos los contornos
        for contour in contours:
            # Obtiene el rectángulo delimitador de cada contorno
            x, y, w, h = cv2.boundingRect(contour)
            # Recorta la región de interés de la imagen original usando el rectángulo delimitador
            area_interes = imagen[y:y+h, x:x+w]
            cuadros_delimitadores.append( (x, y, w, h) )
            imagenes_interes.append(area_interes)

        return cuadros_delimitadores, imagenes_interes
        
    def obtener_textos(self, imagenes_interes):
        textos = []
        if self.idioma_entrada == "Japonés":
            for imagen_interes in imagenes_interes:
                area_interes_pil = Image.fromarray(cv2.cvtColor(imagen_interes, cv2.COLOR_BGR2RGB))
                text = self.mocr(area_interes_pil)
                textos.append(text)
        else:
            for imagen_interes in imagenes_interes:
                area_interes_pil = Image.fromarray(cv2.cvtColor(imagen_interes, cv2.COLOR_BGR2RGB))
                resultado_paddle = self.paddle_ocr.ocr(
                    img=np.array(area_interes_pil),
                    cls=True
                )
                text = ""
                if not resultado_paddle[0]:
                    pass
                else:
                    for i, line in enumerate(resultado_paddle[0]):
                        linea_actual = line[-1][0]
                        if self.idioma_entrada == "Inglés" or self.idioma_entrada == "Español":
                            if i > 0:
                                text += " " + linea_actual
                            else:
                                text += linea_actual
                        else:
                            if (i+1) < len(resultado_paddle[0]):  # Si no es la última línea
                                text += linea_actual.replace("~", "")
                            else:
                                text += linea_actual
                                
                textos.append(text)      
        return textos
    
    def reemplazar_caracter_especial(self, texto):
        # Diccionario de caracteres especiales y sus equivalentes normales
        caracteres_especiales = {
            "。": ".",
            "·" : ".",
            "？": "?",
            "．": ".",
            "・": ".",
            "！": "!",
            "０": ""
        }
        # Reemplazar caracteres especiales utilizando el diccionario
        for especial, normal in caracteres_especiales.items():
            texto = texto.replace(especial, normal)
        return texto
    
    def suprimir_caracteres_repetidos(self, texto, min_reps=3):
        # El patrón necesita permitir al menos min_reps repeticiones
        patron = r"(.)\1{{{},}}".format(min_reps)
        
        def reemplazo(match):
            # Devuelve solo tres repeticiones del carácter encontrado
            return match.group(1) * 3
        
        # Reemplazamos las repeticiones por solo tres caracteres repetidos seguidos
        texto_modificado = re.sub(patron, reemplazo, texto)
        return texto_modificado
    
    def suprimir_simbolos_y_espacios(self, texto):
        # Recorremos cada carácter de la cadena
        for char in texto:
            # Verificamos si el carácter es alfanumérico (letra o número)
            if char.isalnum(): # o usar .isalpha (mas exhaustivo)
                return texto
        return ""
    
    def traducir_textos(self, textos):
        # 1. Traducir textos brutos
        if self.metodo_traduccion == "LLM":
            textos_traducidos_brutos = self.translator_manager.traducir_textos_llm(textos, self.historial_contexto)
        else:
            textos_traducidos_brutos = []
            for texto in textos:
                traducido = self.translator_manager.traducir_texto(texto)
                if traducido is None or len(traducido) <= 1:
                    traducido = ""
                textos_traducidos_brutos.append(traducido)

        # 2. Limpiar los textos traducidos
        textos_traducidos_limpios = []
        for texto_traducido in textos_traducidos_brutos:
            texto_traducido = self.reemplazar_caracter_especial(texto_traducido).strip()
            texto_traducido = self.suprimir_caracteres_repetidos(texto_traducido)
            texto_traducido = self.suprimir_simbolos_y_espacios(texto_traducido)
            textos_traducidos_limpios.append(texto_traducido)
            
        # 3. Guardar en el historial un "Diccionario" de Original -> Traducción
        if self.metodo_traduccion == "LLM":
            # Creamos una lista con el formato: "Texto Japonés -> Texto Español"
            contexto_bilingue = [
                f"{orig} -> {trad}" 
                for orig, trad in zip(textos, textos_traducidos_limpios) 
                if orig.strip() and trad.strip()
            ]
            
            self.historial_contexto.append(contexto_bilingue)
            if len(self.historial_contexto) > 2:
                self.historial_contexto.pop(0) # Mantiene solo las últimas 2 páginas

        return textos_traducidos_limpios
    
    def obtener_color_texto_borde(self, imagen_limpia, x, y, w, h):
         # Calcula las coordenadas para la región con el margen
        x_margin = max(0, x - 5)
        y_margin = max(0, y - 5)
        w_margin = min(w + 5, imagen_limpia.shape[1] - x_margin)
        h_margin = min(h + 5, imagen_limpia.shape[0] - y_margin)
        # Calcula la región con el margen alrededor de las coordenadas ajustadas
        region_alrededor = imagen_limpia[y_margin:y_margin + h_margin, x_margin:x_margin + w_margin]
        promedio_color = cv2.mean(region_alrededor)[:3]
        if np.mean(promedio_color) < 128:
            color_borde = COLOR_NEGRO
            color_texto = COLOR_BLANCO
        else:
            color_borde = COLOR_BLANCO
            color_texto = COLOR_NEGRO
            
        return color_borde, color_texto
    
    def calcular_ancho_texto(self, texto, fuente):
        try:
            return fuente.getbbox(texto)[2] - fuente.getbbox(texto)[0]
        except Exception as e:
            print(f"Error al calcular ancho del texto: {e}")
            return 0
        
    def calcular_alto_texto(self, texto, fuente):
        try:
            return fuente.getbbox(texto)[3] - fuente.getbbox(texto)[1]
        except Exception as e:
            print(f"Error al calcular alto del texto: {e}")
            return 0
            
    def dividir_en_parrafo(self, palabras, fuente, w):
        parrafo = []
        linea_actual = ""
        for palabra in palabras:
            texto_linea_actual = " ".join((linea_actual, palabra)).strip()
            if self.calcular_ancho_texto(texto_linea_actual, fuente) > w:
                if len(linea_actual) > 0:
                    parrafo.append(linea_actual)
                linea_actual = palabra
            else:
                linea_actual = texto_linea_actual

        if len(linea_actual) > 0:
            parrafo.append(linea_actual)
        
        return parrafo
    
    def obtener_propiedades_fuente(self, ancho_caja, alto_caja, texto):
        numero_de_caracteres = max(len(texto), 1)
        max_size = 100

        # Valores predeterminados
        tamanio_fuente = TAMANIO_MINIMO_FUENTE
        fuente = ImageFont.truetype(RUTA_FUENTE, tamanio_fuente)
        alto_parrafo = 0
        espacio_entre_lineas = min(TAMANIO_MINIMO_FUENTE + 2, tamanio_fuente * FACTOR_ESPACIO)

        draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))  # Un único objeto ImageDraw

        for tamanio_fuente in range(TAMANIO_MINIMO_FUENTE, max_size + 1):
            try:
                fuente = ImageFont.truetype(RUTA_FUENTE, tamanio_fuente)
            except Exception as e:
                print(f"Error al cargar la fuente con tamaño {tamanio_fuente}: {e}")
                break

            espacio_entre_lineas = min(TAMANIO_MINIMO_FUENTE + 2, tamanio_fuente * FACTOR_ESPACIO)
            palabras = texto.split(' ')
            alto_parrafo, ancho_linea_actual, ancho_maximo_linea = 0, 0, 0

            for palabra in palabras:
                text_bbox = draw.textbbox((0, 0), palabra, font=fuente)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                ancho_linea_actual += text_width

                if ancho_linea_actual > ancho_caja:
                    alto_parrafo += text_height + espacio_entre_lineas
                    ancho_linea_actual = text_width
                    ancho_maximo_linea = max(ancho_maximo_linea, ancho_linea_actual)
                else:
                    ancho_maximo_linea = max(ancho_maximo_linea, ancho_linea_actual)

            alto_parrafo += text_height

            if alto_parrafo > alto_caja or ancho_maximo_linea > ancho_caja:
                break

        return fuente, alto_parrafo, espacio_entre_lineas
    
    def ajustar_tam_fuente_ancho(self, parrafo, fuente, espacio_entre_lineas, w):
        for i, linea in enumerate(parrafo):
            anchoLinea = self.calcular_ancho_texto(linea, fuente)
            while anchoLinea > w:
                tamanio_fuente = fuente.size - 1
                tamanio_fuente = max(TAMANIO_MINIMO_FUENTE, tamanio_fuente)
                fuente = ImageFont.truetype(RUTA_FUENTE, tamanio_fuente)
                espacio_entre_lineas = min(TAMANIO_MINIMO_FUENTE + 2, tamanio_fuente * FACTOR_ESPACIO)
                if tamanio_fuente == TAMANIO_MINIMO_FUENTE:
                    break
                anchoLinea = self.calcular_ancho_texto(linea, fuente)

        return fuente, espacio_entre_lineas
    
    def ajustar_tam_fuente_alto(self, parrafo, fuente, espacio_entre_lineas, h):
        excede_altura = True
        while excede_altura:
            altoParrafo = sum([self.calcular_alto_texto(linea, fuente) for linea in parrafo])
            altoParrafo += (len(parrafo) - 1) * espacio_entre_lineas

            if altoParrafo > h:
                tamanio_fuente = fuente.size - 1
                tamanio_fuente = max(TAMANIO_MINIMO_FUENTE, tamanio_fuente)
                if tamanio_fuente == TAMANIO_MINIMO_FUENTE:
                    excede_altura = False
                fuente = ImageFont.truetype(RUTA_FUENTE, tamanio_fuente)
                espacio_entre_lineas = min(TAMANIO_MINIMO_FUENTE + 2, tamanio_fuente * FACTOR_ESPACIO)
            else:
                excede_altura = False

        return fuente, espacio_entre_lineas
    
    def incrustar_textos(self, imagen_limpia, cuadros_delimitadores, textos):
        for (x, y, w, h), texto in zip(cuadros_delimitadores, textos):
            self.transcripcion_queue.put({
                'agregar_a_sublista': {
                    'clave_lista': 'Transcripción',
                    'pagina': self.indice_imagen + 1,
                    'clave_sublista': 'Globos de texto',
                    'elemento_sublista': {
                        'Coordenadas': [[x, y], [x + w, y + h]],
                        'Texto': texto
                    }
                }
            })
            
        # Configura el tamaño y tipo de fuente
        fuente = ImageFont.truetype(RUTA_FUENTE, TAMANIO_MINIMO_FUENTE)
        # Copia la imagen original para no modificarla
        imagen_traducida = Image.fromarray(imagen_limpia)
        textos_traducidos = self.traducir_textos(textos)
        # Crea un objeto ImageDraw para dibujar sobre la imagen
        draw = ImageDraw.Draw(imagen_traducida)
        # Itera sobre cada cuadro delimitador y su texto correspondiente
        for (x, y, w, h), texto_traducido in zip(cuadros_delimitadores, textos_traducidos):
            self.traduccion_queue.put({
                'agregar_a_sublista': {
                    'clave_lista': 'Traducción',
                    'pagina': self.indice_imagen + 1,
                    'clave_sublista': 'Globos de texto',
                    'elemento_sublista': {
                        'Coordenadas': [[x, y], [x + w, y + h]],
                        'Texto': texto_traducido
                    }
                }
            })
            
            palabras = texto_traducido.split(" ")
            fuente, alto_parrafo, espacio_entre_lineas = self.obtener_propiedades_fuente(w, h, texto_traducido)
            parrafo = self.dividir_en_parrafo(palabras, fuente, w)
            # Reajustar el tamaño de la fuente en función de la anchura
            fuente, espacio_entre_lineas = self.ajustar_tam_fuente_ancho(parrafo, fuente, espacio_entre_lineas, w)
            # Reajustar el tamaño de la fuente en función de la altura
            fuente, espacio_entre_lineas = self.ajustar_tam_fuente_alto(parrafo, fuente, espacio_entre_lineas, h)
            color_borde, color_texto = self.obtener_color_texto_borde(imagen_limpia, x, y, w, h)
            margen_superior = (h - alto_parrafo) / 2 if h > alto_parrafo else 0
            desplazamiento_bordes = 5
            y_texto = y + margen_superior
            # Dibuja texto y bordes
            for i, linea in enumerate(parrafo):
                # Asegúrate de que el alto de línea se calcula aquí, dentro del bucle, para cada línea
                alto_linea = self.calcular_alto_texto(linea, fuente)
                text_bbox = draw.textbbox((x, y), linea, font=fuente)
                anch_linea = text_bbox[2] - text_bbox[0]
                x_texto = x + (w - anch_linea) // 2
                
                # Dibujar bordes varias veces para crear el efecto de grosor
                for j in range(desplazamiento_bordes):
                    for dx, dy in [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)]:
                        pos_bordes = (x_texto + j * dx, y_texto + j * dy)
                        draw.text(pos_bordes, linea, font=fuente, fill=color_borde)
                # Dibujar texto encima de los bordes
                draw.text((x_texto, y_texto), linea, font=fuente, fill=color_texto)

                # Incrementa y_texto por el alto de la línea y el espacio entre líneas
                y_texto += alto_linea + espacio_entre_lineas
           
        del draw  
        imagen_traducida = np.array(imagen_traducida)
        return imagen_traducida