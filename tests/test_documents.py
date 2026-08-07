import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.documents import (
    DocumentProcessingError,
    extract_text,
    safe_filename,
    split_text,
    text_looks_corrupted,
)


class DocumentServiceTests(unittest.TestCase):
    def test_split_text_preserves_content_with_overlap(self) -> None:
        text = "规则一。" * 120
        chunks = split_text(text, chunk_size=200, overlap=40)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk for chunk in chunks))
        self.assertTrue(all(len(chunk) <= 200 for chunk in chunks))

    def test_safe_filename_removes_path_components(self) -> None:
        self.assertEqual(safe_filename("../../规则 文档.txt"), "规则_文档.txt")

    def test_invalid_pdf_has_a_document_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.pdf"
            path.write_bytes(b"not a pdf")

            with self.assertRaises(DocumentProcessingError):
                extract_text(path)

    def test_corrupted_question_mark_text_is_rejected(self) -> None:
        self.assertTrue(text_looks_corrupted("???????\n1. ?????????????????"))
        self.assertFalse(text_looks_corrupted("这是正常问题吗？Yes? 单个问号不应被拒绝。"))

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "corrupted.txt"
            path.write_text("???????\n1. ?????????????????", encoding="utf-8")
            with self.assertRaises(DocumentProcessingError):
                extract_text(path)

    def test_document_character_limit_is_enforced(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "large.txt"
            path.write_text("规则" * 20, encoding="utf-8")

            with self.assertRaises(DocumentProcessingError):
                extract_text(path, max_characters=10)


if __name__ == "__main__":
    unittest.main()
