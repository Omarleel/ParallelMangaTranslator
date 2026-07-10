from __future__ import annotations

from pathlib import Path


def normalized_page_output_name(source_filename: str, page_index: int) -> str:
    """Return a stable, collision-free output filename for a manga page.

    Output names are based on the page's position in the naturally sorted input
    list instead of extracting the first number from the source filename.  This
    avoids collisions for names such as ``EX2_000.jpg`` and ``EX2_001.jpg``,
    which previously both became ``0002.jpg``.
    """
    if page_index < 0:
        raise ValueError("page_index no puede ser negativo")

    extension = Path(source_filename).suffix.lower()
    if extension == ".webp":
        extension = ".jpg"
    if not extension:
        raise ValueError(f"El archivo no tiene extensión: {source_filename}")

    return f"{page_index + 1:04d}{extension}"
