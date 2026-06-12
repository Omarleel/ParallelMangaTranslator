from typing import List
import numpy as np
import cv2
import sys

from parallel_manga_translator.config.constants import COLOR_BLANCO, COLOR_NEGRO, RUTA_MODELO_AOT, RUTA_MODELO_LAMA, RUTA_MODELO_LAMA_LARGE
from .modules import DEFAULT_DEVICE, DEVICE_SELECTOR, TORCH_DTYPE_MAP, BF16_SUPPORTED


class OpenCVInpainter:
    def __init__(self):
        pass
    
    def inpaint(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

class BNInpainter:
    def __init__(self):
        pass
    
    def inpaint(self, img: np.ndarray, resultados: List[np.ndarray], expansion: int = 1) -> np.ndarray:
        height, width = img.shape[:2]
        inpainted_img = img.copy()  # Copia de la imagen original para mantenerla intacta
        for detection in resultados:
            caja = detection[0]
            puntos = np.array(caja, dtype=np.int32).reshape((-1, 1, 2))
            
            # Calcular los límites de la región con margen
            x, y, w, h = cv2.boundingRect(puntos)
            x_margin = max(0, x - expansion)
            y_margin = max(0, y - expansion)
            w_margin = min(w + expansion, width - x_margin)
            h_margin = min(h + expansion, height - y_margin)
            
            # Definir los puntos del contorno con margen
            puntos_margin = np.array([[x_margin, y_margin], [x_margin + w_margin, y_margin], 
                                  [x_margin + w_margin, y_margin + h_margin], [x_margin, y_margin + h_margin]], dtype=np.int32)
            
             # Extraer la región de la imagen
            region_img = img[y_margin:y_margin + h_margin, x_margin:x_margin + w_margin]
            
            # Calcular el promedio de colores
            average_color = np.mean(region_img, axis=(0, 1))
            
            if np.mean(average_color) > 128:
                # Pintar de blanco la región
                inpainted_img = cv2.fillPoly(inpainted_img, [puntos_margin], color=COLOR_BLANCO)
            else:
                # Pintar de negro la región
                inpainted_img = cv2.fillPoly(inpainted_img, [puntos_margin], color=COLOR_NEGRO)
        
        return inpainted_img

class PatchmatchInpainter():

    if sys.platform == 'darwin':
        download_file_list = [{
                'url': 'https://github.com/dmMaze/PyPatchMatchInpaint/releases/download/v1.0/macos_arm64_patchmatch_libs.7z',
                'sha256_pre_calculated': ['843704ab096d3afd8709abe2a2c525ce3a836bb0a629ed1ee9b8f5cee9938310', '849ca84759385d410c9587d69690e668822a3fc376ce2219e583e7e0be5b5e9a'],
                'files': ['macos_libopencv_world.4.8.0.dylib', 'macos_libpatchmatch_inpaint.dylib'],
                'save_dir': 'data/libs',
                'archived_files': 'macos_patchmatch_libs.7z',
                'archive_sha256_pre_calculated': '9f332c888be0f160dbe9f6d6887eb698a302e62f4c102a0f24359c540d5858ea'
        }]
    elif sys.platform == 'win32':
        download_file_list = [{
                'url': 'https://github.com/dmMaze/PyPatchMatchInpaint/releases/download/v1.0/windows_patchmatch_libs.7z',
                'sha256_pre_calculated': ['3b7619caa29dc3352b939de4e9981217a9585a13a756e1101a50c90c100acd8d', '0ba60cfe664c97629daa7e4d05c0888ebfe3edcb3feaf1ed5a14544079c6d7af'],
                'files': ['opencv_world455.dll', 'patchmatch_inpaint.dll'],
                'save_dir': 'data/libs',
                'archived_files': 'windows_patchmatch_libs.7z',
                'archive_sha256_pre_calculated': 'c991ff61f7cb3efaf8e75d957e62d56ba646083bc25535f913ac65775c16ca65'
        }]

    def __init__(self):
        super().__init__()
        from . import patch_match
    
    def _inpaint(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return patch_match.inpaint(img, mask, patch_size=3)

    def is_computational_intensive(self) -> bool:
        return True
    
    def is_cpu_intensive(self) -> bool:
        return True




import torch
from .aot import AOTGenerator, load_aot_model


class AOTInpainter():
    download_file_list = [{
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting.ckpt',
            'sha256_pre_calculated': '878d541c68648969bc1b042a6e997f3a58e49b6c07c5636ad55130736977149f',
            'files': 'models/inpainting/aot_inpainter.ckpt',
    }]

    def __init__(self):
        self.inpaint_size =  int(1024)
        self.model: AOTGenerator = None
        
    async def _load(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = load_aot_model(RUTA_MODELO_AOT, self.device)

    def _unload(self):
        del self.model
        
    def moveToDevice(self, device: str, precision: str = None):
        self.model.to(device)
        self.device = device

    def resize_keepasp(self, im, new_shape=640, scaleup=True, interpolation=cv2.INTER_LINEAR, stride=None):
        shape = im.shape[:2]  # current shape [height, width]

        if new_shape is not None:
            if not isinstance(new_shape, tuple):
                new_shape = (new_shape, new_shape)
        else:
            new_shape = shape

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        if stride is not None:
            h, w = new_unpad
            if h % stride != 0 :
                new_h = (stride - (h % stride)) + h
            else :
                new_h = h
            if w % stride != 0 :
                new_w = (stride - (w % stride)) + w
            else :
                new_w = w
            new_unpad = (new_h, new_w)
            
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=interpolation)
        return im
    
    def inpaint_preprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        img_original = np.copy(img)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        new_shape = self.inpaint_size if max(img.shape[0: 2]) > self.inpaint_size else None

        img = self.resize_keepasp(img, new_shape, stride=None)
        mask = self.resize_keepasp(mask, new_shape, stride=None)

        im_h, im_w = img.shape[:2]
        pad_bottom = 128 - im_h if im_h < 128 else 0
        pad_right = 128 - im_w if im_w < 128 else 0
        mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 127.5 - 1.0
        mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1

        if self.device != 'cpu':
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)
        img_torch *= (1 - mask_torch)
        return img_torch, mask_torch, img_original, mask_original, pad_bottom, pad_right

    @torch.no_grad()
    async def _inpaint(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:

        im_h, im_w = img.shape[:2]
        img_torch, mask_torch, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)
        img_inpainted_torch = self.model(img_torch, mask_torch)
        img_inpainted = ((img_inpainted_torch.detach().cpu().squeeze_(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
        if pad_bottom > 0:
            img_inpainted = img_inpainted[:-pad_bottom]
        if pad_right > 0:
            img_inpainted = img_inpainted[:, :-pad_right]
        new_shape = img_inpainted.shape[:2]
        if new_shape[0] != im_h or new_shape[1] != im_w :
            img_inpainted = cv2.resize(img_inpainted, (im_w, im_h), interpolation = cv2.INTER_LINEAR)
        img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        
        return img_inpainted


from .lama import LamaFourier, load_lama_mpe

class LamaInpainterMPE:
    download_file_list = [{
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting_lama_mpe.ckpt',
            'sha256_pre_calculated': 'd625aa1b3e0d0408acfd6928aa84f005867aa8dbb9162480346a4e20660786cc',
            'files': 'models/inpainting/lama_mpe.ckpt',
    }]

    def __init__(self):
        self.inpaint_size = int(1024)
        self.precision = 'fp32'
        self.model: LamaFourier = None

    async def _load(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = load_lama_mpe(RUTA_MODELO_LAMA, self.device)

    def _unload(self):
        del self.model

    def resize_keep_aspect(self, im, new_shape=640, scaleup=True, interpolation=cv2.INTER_LINEAR, stride=None):
        shape = im.shape[:2]  # current shape [height, width]

        if new_shape is not None:
            if not isinstance(new_shape, tuple):
                new_shape = (new_shape, new_shape)
        else:
            new_shape = shape

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        if stride is not None:
            h, w = new_unpad
            if h % stride != 0 :
                new_h = (stride - (h % stride)) + h
            else :
                new_h = h
            if w % stride != 0 :
                new_w = (stride - (w % stride)) + w
            else :
                new_w = w
            new_unpad = (new_h, new_w)
            
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=interpolation)
        return im
    
    def inpaint_preprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        img_original = np.copy(img)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        new_shape = self.inpaint_size if max(img.shape[0: 2]) > self.inpaint_size else None
        # high resolution input could produce cloudy artifacts
        img = self.resize_keep_aspect(img, new_shape, stride=64)
        mask = self.resize_keep_aspect(mask, new_shape, stride=64)

        im_h, im_w = img.shape[:2]
        longer = max(im_h, im_w)
        pad_bottom = longer - im_h if im_h < longer else 0
        pad_right = longer - im_w if im_w < longer else 0
        mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 255.0
        mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1
        rel_pos, _, direct = self.model.load_masked_position_encoding(mask_torch[0][0].numpy())
        rel_pos = torch.LongTensor(rel_pos).unsqueeze_(0)
        direct = torch.LongTensor(direct).unsqueeze_(0)

        if self.device != 'cpu':
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)
            rel_pos = rel_pos.to(self.device)
            direct = direct.to(self.device)
        img_torch *= (1 - mask_torch)
        return img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right


    @torch.no_grad()
    async def _inpaint(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        im_h, im_w = img.shape[:2]
        img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)
        
        precision = TORCH_DTYPE_MAP[self.precision]
        if self.device in {'cuda'}:
            try:
                with torch.autocast(device_type=self.device, dtype=precision):
                    img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
            except Exception as e:
                print(f'{precision} inference is not supported for this device, use fp32 instead.')
                img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
        else:
            img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)

        img_inpainted = (img_inpainted_torch.detach().cpu().squeeze_(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        if pad_bottom > 0:
            img_inpainted = img_inpainted[:-pad_bottom]
        if pad_right > 0:
            img_inpainted = img_inpainted[:, :-pad_right]
        new_shape = img_inpainted.shape[:2]
        if new_shape[0] != im_h or new_shape[1] != im_w :
            img_inpainted = cv2.resize(img_inpainted, (im_w, im_h), interpolation = cv2.INTER_LINEAR)
        img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        
        return img_inpainted

    def moveToDevice(self, device: str, precision: str = None):
        self.model.to(device)
        self.device = device
        if precision is not None:
            self.precision = precision

class LamaLarge(LamaInpainterMPE):
    download_file_list = [{
            'url': 'https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt',
            'sha256_pre_calculated': '11d30fbb3000fb2eceae318b75d9ced9229d99ae990a7f8b3ac35c8d31f2c935',
            'files': 'data/models/lama_large_512px.ckpt',
    }]

    def __init__(self):
        self.inpaint_size = int(1536) # 512, 768, 1024, 1536, 2048
        self.precision = 'fp32' # or bf16 if BF16_SUPPORTED == 'cuda'

    async def _load(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = load_lama_mpe(RUTA_MODELO_LAMA_LARGE, device='cpu', use_mpe=False, large_arch=True)
        self.moveToDevice(self.device, precision=self.precision )

    def _unload(self):
        del self.model