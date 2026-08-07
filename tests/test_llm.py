import unittest
from dataclasses import replace

from app.config import get_settings
from app.schemas import RuntimeConfigRequest
from app.services.llm import (
    LOCAL_EMBEDDING_DIMENSION,
    LLMClient,
    _fallback_answer,
    _normalize_base_url,
    _validate_embeddings,
)


class LLMClientTests(unittest.IsolatedAsyncioTestCase):
    def test_blank_form_value_can_preserve_or_clear_api_key(self) -> None:
        settings = replace(get_settings(), openai_api_key="existing-key", embedding_model="local")
        client = LLMClient(settings)

        client.configure(RuntimeConfigRequest(api_key=None, chat_model="updated-model"))
        self.assertTrue(client.is_configured)
        self.assertEqual(client.api_key, "existing-key")

        client.configure(RuntimeConfigRequest(api_key=""))
        self.assertFalse(client.is_configured)

    async def test_local_embedding_has_stable_dimension(self) -> None:
        settings = replace(get_settings(), openai_api_key="", embedding_model="local")
        client = LLMClient(settings)

        result = await client.embed_texts(["测试规则", "another rule"])

        self.assertEqual(len(result.vectors), 2)
        self.assertTrue(all(len(vector) == LOCAL_EMBEDDING_DIMENSION for vector in result.vectors))
        self.assertEqual(result.warnings, [])

    def test_embedding_response_validation_rejects_mismatched_results(self) -> None:
        with self.assertRaises(ValueError):
            _validate_embeddings([[1.0, 0.0]], expected_count=2)

        with self.assertRaises(ValueError):
            _validate_embeddings([[1.0], [1.0, 0.0]], expected_count=2)

    def test_local_fallback_uses_markdown_structure(self) -> None:
        answer = _fallback_answer("是否符合规则？", [], None, [])

        self.assertIn("## 结论", answer)
        self.assertIn("## 视觉依据", answer)
        self.assertIn("## 文档/规则依据", answer)
        self.assertIn("## 建议", answer)

    def test_base_url_validation(self) -> None:
        self.assertEqual(
            _normalize_base_url("https://api.example.com/v1/"), "https://api.example.com/v1"
        )
        for value in (
            "file:///tmp/model",
            "https://user:secret@example.com/v1",
            "https://api.example.com/v1?token=secret",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _normalize_base_url(value)


if __name__ == "__main__":
    unittest.main()
