import gc
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from app.config import get_settings
from app.services.llm import EmbeddingResult
from app.services.retriever import LocalJsonVectorStore, RetrieverService, _numeric_id


class VariableEmbeddingClient:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension

    async def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        vectors = [[1.0] + [0.0] * (self.dimension - 1) for _ in texts]
        return EmbeddingResult(vectors=vectors, warnings=[])


class RetrieverServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_numeric_ids_include_the_chunk_suffix(self) -> None:
        document_id = uuid.uuid4().hex
        numeric_ids = [_numeric_id(f"{document_id}-{index:04d}") for index in range(5)]

        self.assertEqual(len(set(numeric_ids)), 5)

    def test_local_search_filters_corrupted_records(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = LocalJsonVectorStore(Path(temp_dir) / "chunks.json")
            store.add(
                [
                    {"id": "good", "source": "good.txt", "text": "正常规则", "vector": [1.0, 0.0]},
                    {
                        "id": "bad",
                        "source": "bad.txt",
                        "text": "????????????",
                        "vector": [1.0, 0.0],
                    },
                ]
            )

            citations = store.search([1.0, 0.0], top_k=5)

            self.assertEqual([citation.source for citation in citations], ["good.txt"])

    def test_corrupted_local_store_is_not_silently_overwritten(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "chunks.json"
            path.write_text("{not-json", encoding="utf-8")
            store = LocalJsonVectorStore(path)

            with self.assertRaises(RuntimeError):
                store.add([])

    async def test_milvus_preserves_chunks_switches_dimension_and_deduplicates(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            settings = replace(
                get_settings(),
                root_dir=Path(temp_dir),
                milvus_collection=f"test_chunks_{uuid.uuid4().hex[:8]}",
            )
            settings.ensure_directories()
            embedding = VariableEmbeddingClient(dimension=3)
            retriever = RetrieverService(embedding, settings)  # type: ignore[arg-type]

            chunks = ["alpha rule", "beta rule", "gamma rule"]
            first = await retriever.add_chunks("rules.txt", chunks)
            search = await retriever.search("beta rule", top_k=3)
            duplicate = await retriever.add_chunks("rules-copy.txt", chunks)

            self.assertFalse(first.deduplicated)
            self.assertEqual(first.store_mode, "milvus-lite+local-json")
            self.assertEqual(len(search.citations), 3)
            self.assertTrue(duplicate.deduplicated)
            self.assertEqual(
                len(retriever.local_store.records_for_dimension(embedding.dimension)),
                3,
            )

            embedding.dimension = 4
            second = await retriever.add_chunks("other-rules.txt", ["delta rule"])
            switched_search = await retriever.search("delta rule", top_k=3)

            self.assertEqual(second.store_mode, "milvus-lite+local-json")
            self.assertEqual(second.warnings, [])
            self.assertEqual(len(switched_search.citations), 1)
            self.assertEqual(switched_search.citations[0].source, "other-rules.txt")

            del retriever
            gc.collect()


if __name__ == "__main__":
    unittest.main()
