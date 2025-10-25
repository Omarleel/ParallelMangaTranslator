import os, json, tempfile
import torch.multiprocessing as mp

class JsonGenerator:
    def __init__(self):
        self.datos = {}

    def agregar_entrada(self, clave, valor):
        self.datos[clave] = valor

    def agregar_elemento_a_lista(self, clave, elemento):
        self.datos.setdefault(clave, []).append(elemento)

    def agregar_a_sublista(self, clave_lista, pagina, clave_sublista, elemento_sublista):
        if clave_lista not in self.datos or not isinstance(self.datos[clave_lista], list):
            raise ValueError(f"La clave '{clave_lista}' no existe o no es una lista.")
        pagina_data = next((item for item in self.datos[clave_lista] if item.get("Página") == pagina), None)
        if pagina_data is None:
            pagina_data = {"Página": pagina}
            self.datos[clave_lista].append(pagina_data)
        if not isinstance(pagina_data, dict):
            raise ValueError(f"El elemento para la página {pagina} no es un diccionario.")
        pagina_data.setdefault(clave_sublista, []).append(elemento_sublista)

    def ordenar_por_paginas(self, tipo):
        try:
            self.datos[tipo] = sorted(self.datos[tipo], key=lambda x: x["Página"])
            return True
        except Exception as e:
            print(f"Error al ordenar json: {e}")

    def guardar_en_archivo(self, nombre_archivo):
        # Escritura normal (usada por el escritor para hacer dump atómico)
        with open(nombre_archivo, 'w', encoding='utf-8') as archivo:
            json.dump(self.datos, archivo, ensure_ascii=False, indent=4)

class JsonWriter(mp.Process):
    """
    Mensajes soportados (mismos nombres):
      {'agregar_entrada': {...}}
      {'agregar_elemento_a_lista': { 'Clave': {...} }}  # {clave: elemento}
      {'agregar_a_sublista': {'clave_lista':..., 'pagina':..., 'clave_sublista':..., 'elemento_sublista':...}}
      {'ordenar_por_paginas': {'tipo': 'Transcripción' | 'Traducción'}}
      {'guardar_en_archivo': '/ruta/salida.json'}                     # setea ruta y hace checkpoint
      {'guardar_en_archivo': {'path': '/ruta/salida.json', 'finalizar': True}}  # checkpoint y termina
    """
    def __init__(self, result_queue, checkpoint_cada=1):
        super(JsonWriter, self).__init__()
        self.result_queue = result_queue
        self.json_generator = JsonGenerator()
        self.output_path = None
        self.checkpoint_cada = max(1, int(checkpoint_cada))
        self._mods_desde_checkpoint = 0

    def _dump_atomico(self):
        if not self.output_path:
            return
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        tmp_path = self.output_path + ".tmp"
        # dump a .tmp
        self.json_generator.guardar_en_archivo(tmp_path)
        # replace atómico
        os.replace(tmp_path, self.output_path)

    def _maybe_checkpoint(self, force=False):
        if not self.output_path:
            return
        if force or self._mods_desde_checkpoint >= self.checkpoint_cada:
            self._dump_atomico()
            self._mods_desde_checkpoint = 0

    def run(self):
        while True:
            data = self.result_queue.get()
            metodo = next(iter(data))

            if metodo == 'agregar_entrada':
                for clave, valor in data[metodo].items():
                    self.json_generator.agregar_entrada(clave=clave, valor=valor)
                self._mods_desde_checkpoint += 1
                self._maybe_checkpoint()

            elif metodo == 'agregar_elemento_a_lista':
                for clave, elemento in data[metodo].items():
                    self.json_generator.agregar_elemento_a_lista(clave=clave, elemento=elemento)
                self._mods_desde_checkpoint += 1
                self._maybe_checkpoint()

            elif metodo == 'agregar_a_sublista':
                sub = data[metodo]
                self.json_generator.agregar_a_sublista(
                    clave_lista=sub['clave_lista'],
                    pagina=sub['pagina'],
                    clave_sublista=sub['clave_sublista'],
                    elemento_sublista=sub['elemento_sublista']
                )
                self._mods_desde_checkpoint += 1
                self._maybe_checkpoint()

            elif metodo == 'ordenar_por_paginas':
                self.json_generator.ordenar_por_paginas(tipo=data[metodo]['tipo'])
                self._mods_desde_checkpoint += 1
                self._maybe_checkpoint()

            elif metodo == 'guardar_en_archivo':
                # Acepta string o dict {'path':..., 'finalizar': bool}
                payload = data[metodo]
                finalizar = False
                if isinstance(payload, str):
                    self.output_path = payload
                elif isinstance(payload, dict):
                    if 'path' in payload and payload['path']:
                        self.output_path = payload['path']
                    finalizar = bool(payload.get('finalizar', False))
                else:
                    # Tipo no esperado; ignora.
                    pass

                # Fuerza checkpoint inmediato
                self._maybe_checkpoint(force=True)

                if finalizar:
                    break

            else:
                # Mensaje desconocido → terminar (compatibilidad)
                break
