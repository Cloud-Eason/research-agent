"""Verify that every external dependency is reachable."""
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def check(name: str, fn) -> bool:
    try:
        detail = fn()
        print(f"[OK]   {name}: {detail}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {e}")
        return False


def py_version() -> str:
    assert sys.version_info >= (3, 11), sys.version
    return sys.version.split()[0]


def redis_ping() -> str:
    import redis
    r = redis.from_url(os.environ["REDIS_URL"])
    r.ping()
    return f"redis {r.info()['redis_version']}"


def milvus_ping() -> str:
    from pymilvus import MilvusClient
    c = MilvusClient(uri=os.environ["MILVUS_URI"])
    return f"server {c.get_server_version()}, collections={c.list_collections()}"


def deepseek_call() -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url=os.environ["LLM_BASE_URL"])
    resp = client.chat.completions.create(
        model=os.environ["LLM_MODEL"],
        messages=[{"role": "user", "content": "Reply with: pong"}],
        max_tokens=5,
    )
    return f"{resp.choices[0].message.content!r}, tokens={resp.usage.total_tokens}"


def tavily_call() -> str:
    from tavily import TavilyClient
    res = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search("LangGraph", max_results=1)
    return res["results"][0]["url"]


def embed_model() -> str:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vec = emb.encode(["hello"])
    rr = CrossEncoder("BAAI/bge-reranker-base")
    score = rr.predict([("what is redis", "Redis is an in-memory database")])
    return f"dim={vec.shape[1]}, rerank_score={float(score[0]):.3f}"


if __name__ == "__main__":
    results = [
        check("Python", py_version),
        check("Redis", redis_ping),
        check("Milvus", milvus_ping),
        check("DeepSeek", deepseek_call),
        check("Tavily", tavily_call),
        check("Embedding+Reranker", embed_model),
    ]
    sys.exit(0 if all(results) else 1)