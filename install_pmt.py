#!/usr/bin/env python3
"""Instalador automático de Parallel Manga Translator.

Detecta NVIDIA/CPU, selecciona un perfil probado (cu129, cu124 o CPU), crea o
actualiza el entorno y verifica que PyTorch pueda usar la GPU. No necesita que el
proyecto esté instalado previamente.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parent
PROFILE_FILES = {
    "cpu": ROOT / "pmt-env-cpu.yml",
    "cu118": ROOT / "pmt-env-cu118.yml",
    "cu124": ROOT / "pmt-env-cu124.yml",
    "cu129": ROOT / "pmt-env-cu129.yml",
    "rocm64": ROOT / "pmt-env-rocm64.yml",
    "mps": ROOT / "pmt-env-mps.yml",
}
DEFAULT_ENV_NAMES = {
    profile: f"ParallelMangaTranslator-{profile}" for profile in PROFILE_FILES
}


@dataclass(frozen=True)
class NvidiaGpu:
    name: str
    driver_version: str = ""
    memory_mib: int = 0


@dataclass(frozen=True)
class HardwareInfo:
    os_name: str
    architecture: str
    nvidia_gpus: tuple[NvidiaGpu, ...]
    max_cuda_version: Optional[float]
    amd_gpus: tuple[str, ...] = ()
    rocm_available: bool = False
    rocm_version: Optional[float] = None
    apple_silicon: bool = False

    @property
    def has_nvidia(self) -> bool:
        return bool(self.nvidia_gpus)


def run(command: Iterable[str], *, check: bool = True, capture: bool = False, env=None) -> subprocess.CompletedProcess:
    command = [str(item) for item in command]
    print("+", " ".join(command))
    return subprocess.run(
        command,
        cwd=str(ROOT),
        check=check,
        text=True,
        capture_output=capture,
        env=env,
    )


def _parse_cuda_version(text: str) -> Optional[float]:
    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", text or "", flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_first_version(text: str) -> Optional[float]:
    match = re.search(r"(\d+)\.(\d+)", text or "")
    if not match:
        return None
    try:
        return float(f"{match.group(1)}.{match.group(2)}")
    except ValueError:
        return None


def _detect_amd_rocm() -> tuple[tuple[str, ...], bool, Optional[float]]:
    if platform.system() != "Linux":
        return (), False, None
    rocm_tools = [shutil.which(name) for name in ("rocminfo", "rocm-smi", "hipconfig")]
    rocm_available = Path("/dev/kfd").exists() or any(rocm_tools)
    names: list[str] = []
    lspci = shutil.which("lspci")
    if lspci:
        try:
            result = subprocess.run([lspci], text=True, capture_output=True, check=True, timeout=10)
            for line in result.stdout.splitlines():
                upper = line.upper()
                if ("VGA" in upper or "DISPLAY" in upper or "3D CONTROLLER" in upper) and (
                    "AMD" in upper or "ATI" in upper
                ):
                    names.append(line.split(": ", 1)[-1].strip())
        except (OSError, subprocess.SubprocessError):
            pass
    rocm_version = None
    hipconfig = shutil.which("hipconfig")
    if hipconfig:
        try:
            result = subprocess.run([hipconfig, "--version"], text=True, capture_output=True, check=True, timeout=10)
            rocm_version = _parse_first_version(result.stdout + result.stderr)
        except (OSError, subprocess.SubprocessError):
            pass
    return tuple(dict.fromkeys(names)), rocm_available, rocm_version


def detect_hardware() -> HardwareInfo:
    gpus: list[NvidiaGpu] = []
    max_cuda = None
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                check=True,
                timeout=15,
            )
            for line in result.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if not parts or not parts[0]:
                    continue
                memory = 0
                if len(parts) > 2:
                    try:
                        memory = int(float(parts[2]))
                    except ValueError:
                        pass
                gpus.append(NvidiaGpu(parts[0], parts[1] if len(parts) > 1 else "", memory))
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            result = subprocess.run([nvidia_smi], text=True, capture_output=True, check=True, timeout=15)
            max_cuda = _parse_cuda_version(result.stdout)
        except (OSError, subprocess.SubprocessError):
            pass

    amd_gpus, rocm_available, rocm_version = _detect_amd_rocm()
    os_name = platform.system()
    architecture = platform.machine()
    return HardwareInfo(
        os_name=os_name,
        architecture=architecture,
        nvidia_gpus=tuple(gpus),
        max_cuda_version=max_cuda,
        amd_gpus=amd_gpus,
        rocm_available=rocm_available,
        rocm_version=rocm_version,
        apple_silicon=(os_name == "Darwin" and architecture.lower() in {"arm64", "aarch64"}),
    )


def is_blackwell_gpu(name: str) -> bool:
    upper = name.upper()
    patterns = (
        r"\bRTX\s*50\d{2}\b",
        r"\bRTX\s*PRO\b.*\bBLACKWELL\b",
        r"\bB100\b",
        r"\bB200\b",
        r"\bGB\d{2,3}\b",
        r"\bBLACKWELL\b",
    )
    return any(re.search(pattern, upper) for pattern in patterns)


def select_profile(info: HardwareInfo, requested: str = "auto") -> tuple[str, list[str]]:
    notes: list[str] = []
    if requested != "auto":
        selected = requested
    elif info.has_nvidia:
        if any(is_blackwell_gpu(gpu.name) for gpu in info.nvidia_gpus):
            selected = "cu129"
            notes.append("Se detectó una GPU Blackwell/RTX 50; se seleccionó el perfil CUDA 12.9.")
        else:
            selected = "cu124"
            notes.append("Se seleccionó CUDA 12.4 como perfil NVIDIA conservador y probado.")
    elif info.apple_silicon:
        selected = "mps"
        notes.append("Se detectó Apple Silicon; se seleccionó PyTorch con backend MPS.")
    elif info.amd_gpus and info.rocm_available:
        selected = "rocm64"
        notes.append("Se detectó una GPU AMD con runtime ROCm; se seleccionó ROCm 6.4.")
    else:
        selected = "cpu"
        if info.amd_gpus and not info.rocm_available:
            notes.append("Se detectó AMD, pero no un runtime ROCm utilizable; se usará CPU.")
        else:
            notes.append("No se detectó un backend GPU compatible; se usará CPU.")

    if selected == "cu129" and info.max_cuda_version is not None and info.max_cuda_version < 12.9:
        if info.max_cuda_version >= 12.4:
            notes.append(
                f"El driver informa CUDA {info.max_cuda_version:.1f}; se baja a cu124. "
                "Actualiza el driver para usar cu129."
            )
            selected = "cu124"
        elif info.max_cuda_version >= 11.8:
            notes.append(f"El driver informa CUDA {info.max_cuda_version:.1f}; se baja a cu118.")
            selected = "cu118"
        else:
            notes.append("El driver NVIDIA es demasiado antiguo para los perfiles incluidos; se usará CPU.")
            selected = "cpu"
    elif selected == "cu124" and info.max_cuda_version is not None and info.max_cuda_version < 12.4:
        if info.max_cuda_version >= 11.8:
            notes.append(f"El driver informa CUDA {info.max_cuda_version:.1f}; se baja a cu118.")
            selected = "cu118"
        else:
            notes.append("El driver NVIDIA es demasiado antiguo para CUDA 11.8; se usará CPU.")
            selected = "cpu"

    if selected == "rocm64" and info.os_name != "Linux":
        notes.append("ROCm está soportado por este instalador únicamente en Linux; se usará CPU.")
        selected = "cpu"
    if selected == "mps" and not info.apple_silicon:
        notes.append("MPS requiere macOS sobre Apple Silicon; se usará CPU.")
        selected = "cpu"

    return selected, notes


def detect_manager(requested: str) -> tuple[str, Optional[str]]:
    if requested == "venv":
        return "venv", None
    candidates = [requested] if requested != "auto" else ["mamba", "conda", "micromamba"]
    for candidate in candidates:
        executable = shutil.which(candidate)
        if executable:
            return candidate, executable
    if requested != "auto":
        raise RuntimeError(f"No se encontró el gestor solicitado: {requested}")
    return "venv", None


def conda_environment_exists(manager: str, executable: str, env_name: str) -> bool:
    result = run([executable, "env", "list", "--json"], capture=True)
    data = json.loads(result.stdout or "{}")
    suffixes = {Path(path).name.lower() for path in data.get("envs", [])}
    return env_name.lower() in suffixes


def install_with_conda(
    manager: str,
    executable: str,
    profile: str,
    env_name: str,
    *,
    dry_run: bool,
) -> None:
    env_file = PROFILE_FILES[profile]
    if not env_file.exists():
        raise FileNotFoundError(env_file)
    exists = False if dry_run else conda_environment_exists(manager, executable, env_name)
    if exists:
        command = [executable, "env", "update", "-n", env_name, "-f", env_file, "--prune"]
    else:
        command = [executable, "env", "create", "-n", env_name, "-f", env_file]
    if dry_run:
        print("+", " ".join(map(str, command)))
        print("+", executable, "run", "-n", env_name, "python", "-m", "pip", "install", "-e", ".", "--no-deps")
        return
    run(command)
    run([executable, "run", "-n", env_name, "python", "-m", "pip", "install", "-e", ".", "--no-deps"])


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def pip_torch_command(profile: str, python_exe: Path) -> list[str]:
    base = [str(python_exe), "-m", "pip", "install"]
    if profile == "cu129":
        return base + [
            "torch==2.8.0+cu129", "torchvision==0.23.0+cu129",
            "--extra-index-url", "https://download.pytorch.org/whl/cu129",
        ]
    if profile == "cu124":
        return base + [
            "torch==2.6.0+cu124", "torchvision==0.21.0+cu124",
            "--extra-index-url", "https://download.pytorch.org/whl/cu124",
        ]
    if profile == "cu118":
        return base + [
            "torch==2.6.0+cu118", "torchvision==0.21.0+cu118",
            "--extra-index-url", "https://download.pytorch.org/whl/cu118",
        ]
    if profile == "rocm64":
        return base + [
            "torch==2.8.0", "torchvision==0.23.0",
            "--index-url", "https://download.pytorch.org/whl/rocm6.4",
        ]
    if profile == "mps":
        return base + ["torch==2.8.0", "torchvision==0.23.0"]
    return base + [
        "torch==2.6.0", "torchvision==0.21.0",
        "--index-url", "https://download.pytorch.org/whl/cpu",
    ]


def install_with_venv(profile: str, env_name: str, *, dry_run: bool, skip_paddle: bool) -> Path:
    venv_dir = ROOT / f".{env_name}"
    python_exe = venv_python(venv_dir)
    commands = [
        [str(python_exe), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        pip_torch_command(profile, python_exe),
        [str(python_exe), "-m", "pip", "install", "-e", str(ROOT)],
    ]
    if profile == "cu129" and not skip_paddle:
        commands.append(
            [
                str(python_exe),
                "-m",
                "pip",
                "install",
                "paddlepaddle-gpu==3.2.2",
                "-i",
                "https://www.paddlepaddle.org.cn/packages/stable/cu129/",
            ]
        )
        commands.append(
            [
                str(python_exe),
                "-m",
                "pip",
                "install",
                "paddleocr==2.10.0",
                "albumentations==1.4.24",
                "albucore==0.0.23",
            ]
        )
    if dry_run:
        print(f"+ crear venv {venv_dir}")
        for command in commands:
            print("+", " ".join(command))
        return python_exe
    if not python_exe.exists():
        print(f"Creando entorno virtual: {venv_dir}")
        venv.EnvBuilder(with_pip=True, clear=False).create(venv_dir)
    for command in commands:
        run(command)
    return python_exe


def verification_code(profile: str) -> str:
    expected_cuda = profile in {"cu118", "cu124", "cu129"}
    expected_rocm = profile == "rocm64"
    expected_mps = profile == "mps"
    return f'''
import json, sys
import torch
cuda_available = bool(torch.cuda.is_available())
mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
payload = {{
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "rocm_build": getattr(torch.version, "hip", None),
    "cuda_available": cuda_available,
    "mps_available": mps_available,
    "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
    "devices": [],
}}
if cuda_available:
    for index in range(torch.cuda.device_count()):
        payload["devices"].append({{
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "memory_gib": round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 2),
        }})
print(json.dumps(payload, ensure_ascii=False, indent=2))
if {expected_cuda!r} and (not cuda_available or getattr(torch.version, "hip", None)):
    raise SystemExit("ERROR: se instaló un perfil CUDA, pero PyTorch no detecta una GPU CUDA")
if {expected_rocm!r} and (not cuda_available or not getattr(torch.version, "hip", None)):
    raise SystemExit("ERROR: se instaló ROCm, pero PyTorch no detecta el backend HIP/ROCm")
if {expected_mps!r} and not mps_available:
    raise SystemExit("ERROR: se instaló MPS, pero torch.backends.mps.is_available() es False")
'''.strip()


def verify(manager: str, executable: Optional[str], env_name: str, python_exe: Optional[Path], profile: str, dry_run: bool) -> None:
    code = verification_code(profile)
    if manager == "venv":
        command = [str(python_exe), "-c", code]
    else:
        command = [str(executable), "run", "-n", env_name, "python", "-c", code]
    if dry_run:
        print("+ verificación de PyTorch/CUDA en el entorno")
    else:
        run(command)


def print_hardware(info: HardwareInfo) -> None:
    print(f"Sistema: {info.os_name} {info.architecture} ({struct.calcsize('P') * 8} bits)")
    if info.nvidia_gpus:
        for index, gpu in enumerate(info.nvidia_gpus):
            memory = f", {gpu.memory_mib / 1024:.1f} GiB" if gpu.memory_mib else ""
            print(f"GPU NVIDIA {index}: {gpu.name} (driver {gpu.driver_version}{memory})")
        print(f"CUDA máxima informada por el driver: {info.max_cuda_version or 'desconocida'}")
    else:
        print("GPU NVIDIA: no detectada")
    if info.amd_gpus:
        for index, name in enumerate(info.amd_gpus):
            print(f"GPU AMD {index}: {name}")
        print(f"ROCm: {'detectado' if info.rocm_available else 'no detectado'}"
              + (f" ({info.rocm_version:.1f})" if info.rocm_version else ""))
    if info.apple_silicon:
        print("Apple Silicon: detectado; backend MPS disponible tras la verificación de PyTorch")


def main() -> int:
    parser = argparse.ArgumentParser(description="Instala automáticamente el entorno correcto para PMT.")
    parser.add_argument("--profile", choices=["auto", "cpu", "cu118", "cu124", "cu129", "rocm64", "mps"], default="auto")
    parser.add_argument("--manager", choices=["auto", "conda", "mamba", "micromamba", "venv"], default="auto")
    parser.add_argument("--environment-name", default="")
    parser.add_argument("--skip-paddle", action="store_true", help="No instalar PaddleOCR GPU en el perfil cu129.")
    parser.add_argument("--dry-run", action="store_true", help="Solo mostrar la selección y los comandos.")
    args = parser.parse_args()

    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("Se requiere un sistema y Python de 64 bits.")

    info = detect_hardware()
    print_hardware(info)
    profile, notes = select_profile(info, args.profile)
    for note in notes:
        print("-", note)
    print(f"Perfil seleccionado: {profile}")

    manager, executable = detect_manager(args.manager)
    env_name = args.environment_name.strip() or DEFAULT_ENV_NAMES[profile]
    print(f"Gestor seleccionado: {manager}")
    print(f"Entorno: {env_name}")

    python_exe = None
    if manager == "venv":
        python_exe = install_with_venv(profile, env_name, dry_run=args.dry_run, skip_paddle=args.skip_paddle)
    else:
        assert executable is not None
        install_with_conda(manager, executable, profile, env_name, dry_run=args.dry_run)

    verify(manager, executable, env_name, python_exe, profile, args.dry_run)
    if manager == "venv":
        if os.name == "nt":
            activation = f".{env_name}\\Scripts\\activate"
        else:
            activation = f"source .{env_name}/bin/activate"
    else:
        activation = f"{manager} activate {env_name}"
    print("\nPlan de instalación generado." if args.dry_run else "\nInstalación completada.")
    print("Activación:", activation)
    print("Interfaz:", "pmt-ui")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Instalación cancelada por el usuario.")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
