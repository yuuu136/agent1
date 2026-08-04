import os

from openai import OpenAI


class QwenEmbeddingClient:
    def __init__(
        self,
        api_key_env: str,
        base_url: str,
        model_name: str,
        timeout_seconds: int = 60,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable: {api_key_env}")

        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        vectors = self.embed_texts([query])
        if not vectors:
            return []
        return vectors[0]
