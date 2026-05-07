import os
from pypdf import PdfReader

class PDFLoader:

    def load(self, file_path):
        docs = []
        items = sorted(os.listdir(file_path))  # consistent ordering

        for item in items:
            if item.endswith(".pdf"):
                full_path = os.path.join(file_path, item)
                reader = PdfReader(full_path)

                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted

                if not text.strip():
                    print(f"Warning: No text extracted from {item}")
                    continue

                docs.append({
                    "id": item.replace(".pdf", ""),
                    "title": item,
                    "content": text,
                    "page_count": len(reader.pages),
                    "path": full_path
                })
                print(f"Loaded: {item} ({len(reader.pages)} pages)")

        return docs