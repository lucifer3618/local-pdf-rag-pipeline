import os
import logging
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFLoader:
    def load(self, file_path):
        if not os.path.isdir(file_path):
            raise FileNotFoundError(f"PDF directory not found: {file_path}")

        docs = []
        items = sorted(os.listdir(file_path))  # consistent ordering

        for item in items:
            if item.endswith(".pdf"):
                full_path = os.path.join(file_path, item)
                try:
                    reader = PdfReader(full_path)
                except Exception as exc:
                    logger.warning("Failed to read PDF %s: %s", item, exc)
                    continue

                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted

                if not text.strip():
                    logger.warning("No text extracted from %s", item)
                    continue

                docs.append({
                    "id": item.replace(".pdf", ""),
                    "title": item,
                    "content": text,
                    "page_count": len(reader.pages),
                    "path": full_path
                })
                
                logger.info("Loaded %s (%s pages)", item, len(reader.pages))

                logger.info("Loaded %s PDF documents", len(docs))
        return docs