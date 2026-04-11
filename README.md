# ParallelMangaTranslator

ParallelMangaTranslator es una herramienta diseñada para simplificar y optimizar el proceso de limpieza y traducción de mangas, especialmente para los aficionados del manga y los grupos de scanlation. Utilizando tecnología OCR (Optical Character Recognition), esta herramienta detecta automáticamente el texto dentro de los mangas escaneados y proporciona funcionalidades para traducir y limpiar las páginas usando GPU CUDA de forma paralela, en ese sentido, ParallelMangaTranslator es una mejora del programa que desarrollé con anterioridad: [MangaTranslate](https://github.com/Omarleel/MangaTranslate).

## Características

- **Procesamiento por Carpetas:** Permite procesar múltiples imágenes al seleccionar o especificar manualmente la ruta de la carpeta que contiene las imágenes. Admite formatos como .jpg, .png, .jpeg, .bmp y .webp. Además, puede descargar y descomprimir automáticamente archivos .zip desde Google Drive. 
- **Detección Automática de Texto:** Utiliza algoritmos OCR avanzados para identificar texto en las páginas de manga, independientemente del estilo de dibujo o letra.
- **Limpieza de Páginas:** Ofrece una herramienta para eliminar los textos de las páginas, facilitando la lectura y la posterior traducción.
- **Camuflaje Avanzado:** Permite camuflar los textos en fondos de páginas a color, adaptándose incluso a fondos irregulares.
- **Traducción Precisa con Google:** Utiliza GoogleTranslator de deep_translator para traducir el manga a varios idiomas, incluyendo japonés, inglés, español, coreano y chino, manteniendo la fidelidad del texto original.
- **Almacenamiento y Organización de Imágenes Procesadas:** Todas las imágenes procesadas se almacenan en la carpeta "Outputs" y se organizan en subcarpetas según la acción realizada sobre ellas, ya sea "Limpieza" o "Traducción".
- **Exportación de Datos en Formato JSON:** Genera archivos .json estructurados que contienen las transcripciones y traducciones de las páginas procesadas, facilitando su análisis o integración con otras aplicaciones.
- **Traducción Contextual con LLM (Nuevo):** Además de la traducción tradicional (Google Translator), ahora integra **Modelos de Lenguaje (LLMs)** a través de Groq. Esto permite traducciones coherentes que entienden el contexto, el *lore* del manga, el tono de los personajes y la jerga, superando las limitaciones de la traducción literal.

##  Requerimientos

Antes de utilizar ParallelMangaTranslator, asegúrate de tener instalados los siguientes requisitos o instalarlos desde requirements.txt:
- Python 3.10.10
- **CUDA**: Esencial para el funcionamiento de algunas bibliotecas OCR que requieren procesamiento en GPU.
- OpenCV: Una biblioteca de procesamiento de imágenes y visión por computadora.
- EasyOCR: Una biblioteca para el reconocimiento óptico de caracteres (OCR) fácil de usar.
- Manga-OCR: Una biblioteca para el reconocimiento óptico de caracteres (OCR) especializada en mangas.
- PaddleOCR: Una biblioteca de OCR basada en PaddlePaddle que admite varios idiomas.
- Deep_Translator: Una biblioteca flexible, gratuita e ilimitada para traducir entre diferentes idiomas de forma sencilla utilizando varios traductores.
- Pillow: Una biblioteca para manipulación de imágenes en Python.
- pydrive2: Una biblioteca de Python que envuelve la API de Google Drive, facilitando las operaciones de carga y descarga de archivos.

## Instalación de CUDA compatible con Torch
Ejecuta los siguientes comandos:
```bash
# Desinstala cualquier versión de Torch que tengas
pip uninstall torch torchvision torchaudio
# Limpia la caché de pip para evitar conflictos
pip cache purge
# Instala la versión específica de Torch compatible con CUDA 12.1:
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

## Contribuciones

Si deseas contribuir al desarrollo de ParallelMangaTranslator, ¡no dudes en hacerlo! Puedes enviar pull requests o reportar problemas en el repositorio del proyecto.

## Pruebas
Para correr el programa, puedes ejecutar el siguiente conjunto de comandos:
```bash
# Clonar el repositorio y acceder a la carpeta del programa
git clone https://github.com/Omarleel/ParallelMangaTranslator
# Accede al proyecto
cd ParallelMangaTranslator
# Instala los requerimientos
pip install -r requirements.txt
# Ejecutar el script
py ParallelMangaTranslator.py
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Omarleel/ParallelMangaTranslator/blob/main/ParallelMangaTranslator.ipynb)