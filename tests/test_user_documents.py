"""Локальные тесты этапа загрузки без использования ChromaDB и Ollama."""
import io
import sys
import types
import unittest
from pathlib import Path

from user_documents import extract_text, split_text, MAX_CHUNKS


class DocumentTests(unittest.TestCase):
    def test_txt(self):
        text = extract_text('guide.txt', ('Лучшие места Рима: Колизей, Пантеон и Римский форум. ' * 5).encode('utf-8'))
        self.assertIn('Пантеон', text)

    def test_pdf(self):
        import fitz
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((50, 50), 'Rome attractions: Colosseum, Pantheon and the Roman Forum. More travel information.')
        content = pdf.tobytes()
        pdf.close()
        self.assertIn('Colosseum', extract_text('guide.pdf', content))

    def test_docx(self):
        from docx import Document
        doc = Document()
        doc.add_paragraph('Rome tourist guide Colosseum and Pantheon, together with the Roman Forum.')
        buffer = io.BytesIO()
        doc.save(buffer)
        self.assertIn('Pantheon', extract_text('guide.docx', buffer.getvalue()))

    def test_invalid_and_empty(self):
        with self.assertRaises(ValueError):
            extract_text('test.zip', b'123')
        with self.assertRaises(ValueError):
            extract_text('test.txt', b'')

    def test_overlap(self):
        parts = split_text(' '.join(str(i) for i in range(400)), 100, 20)
        self.assertEqual(len(parts), 5)
        self.assertEqual(parts[0].split()[-20:], parts[1].split()[:20])

    def test_too_many_chunks(self):
        with self.assertRaises(ValueError):
            split_text('text ' * 30000)


if __name__ == '__main__':
    unittest.main()
