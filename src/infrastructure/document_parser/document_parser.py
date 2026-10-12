from pathlib import Path

from docling.document_converter import DocumentConverter


class DocumentParser:
    def __init__(self):
        self._converter = DocumentConverter()

    def parse(self, path: Path) -> str:
        result = self._converter.convert(path)
        return result.document.export_to_markdown()
