from tools.base import Tool
from typing import Dict, Any

import os

try:
    from rag.index import ChromaIndex
except Exception:  # noqa: BLE001
    ChromaIndex = None  # type: ignore

from rag.rerank import rerank_hits


class Retrieval(Tool):
    name = "retrieval"
    description = "Retrieves documents from the knowledge base."

    def __init__(self):
        # Lazy-init to avoid downloading embedding models at import time.
        self.index = None
        self._init_error: str | None = None

    def _get_index(self):
        if self.index is not None:
            return self.index
        if self._init_error:
            return None
        if not ChromaIndex:
            self._init_error = "Chroma is not available"
            return None
        try:
            self.index = ChromaIndex()
            return self.index
        except Exception as exc:  # noqa: BLE001
            self._init_error = str(exc)
            return None

    def run(self, tool_input: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        query = tool_input.get("query", "")
        if isinstance(query, list):
            query = " ".join(map(str, query))
        query = str(query)
        try:
            top_k = int(tool_input.get("top_k", 10))
        except Exception:
            top_k = 10

        threshold = 0.4

        if not query.strip():
            return {"status": "error", "output": "Query is required"}
        try:
            index = self._get_index()
            if not index:
                return {
                    "status": "error",
                    "output": self._init_error or "Chroma is not available",
                }
            hits = index.query(query, top_k)
            if isinstance(hits, dict) and "error" in hits:
                return {"status": "error", "output": hits["error"]}
            filtered = []
            for h in hits:
                dist = h.get("distance", 1.0)
                if dist <= threshold:
                    filtered.append(h)
            if not filtered:
                return {
                    "status": "error",
                    "output": f"No relevant context found (threshold {threshold})",
                }

            # Cross-encoder rerank on the filtered set.
            rerank_top_n = int(tool_input.get("rerank_top_n", min(5, len(filtered))))
            rerank_model = tool_input.get("rerank_model", "BAAI/bge-reranker-v2-m3")
            reranked = rerank_hits(
                query, filtered, top_n=rerank_top_n, model_name=rerank_model
            )
            base_hits = reranked.hits

            expand = bool(tool_input.get("expand_neighbors", True))
            if not expand:
                return {
                    "status": "ok",
                    "output": base_hits,
                    "rerank_backend": reranked.backend,
                }

            radius = int(tool_input.get("neighbor_radius", 1))
            window_max_chars = int(tool_input.get("window_max_chars", 800))

            def make_id(src: str, cid: int) -> str:
                return f"{os.path.basename(src)}::chunk-{cid}"

            needed_ids: set[str] = set()
            for h in base_hits:
                meta = h.get("metadata") or {}
                src = meta.get("source")
                cid = meta.get("chunk_id")
                if isinstance(src, str) and isinstance(cid, int):
                    for d in range(-radius, radius + 1):
                        ncid = cid + d
                        if ncid >= 0:
                            needed_ids.add(make_id(src, ncid))

            fetched = index.get_by_ids(sorted(needed_ids)) if needed_ids else []
            if isinstance(fetched, dict) and "error" in fetched:
                return {
                    "status": "ok",
                    "output": base_hits,
                    "rerank_backend": reranked.backend,
                }

            lookup: dict[tuple[str, int], dict] = {}
            for doc in fetched:
                meta = doc.get("metadata") or {}
                src = meta.get("source")
                cid = meta.get("chunk_id")
                if isinstance(src, str) and isinstance(cid, int):
                    lookup[(src, cid)] = doc

            windows = []
            for h in base_hits:
                meta = h.get("metadata") or {}
                src = meta.get("source")
                cid = meta.get("chunk_id")
                if not (isinstance(src, str) and isinstance(cid, int)):
                    windows.append(h)
                    continue
                cids = [cid + d for d in range(-radius, radius + 1) if cid + d >= 0]
                docs = [lookup.get((src, c)) for c in cids]
                docs = [d for d in docs if d]
                docs.sort(key=lambda d: (d.get("metadata") or {}).get("chunk_id", 0))
                text = "\n".join(str(d.get("text", "")) for d in docs).strip()
                if len(text) > window_max_chars:
                    text = text[:window_max_chars]

                out_meta = dict(meta)
                out_meta["center_chunk_id"] = cid
                out_meta["chunk_ids"] = [
                    (d.get("metadata") or {}).get("chunk_id") for d in docs
                ]
                nh = dict(h)
                nh["text"] = text
                nh["metadata"] = out_meta
                windows.append(nh)

            return {
                "status": "ok",
                "output": windows,
                "rerank_backend": reranked.backend,
            }
        except Exception as e:
            return {"status": "error", "output": str(e)}
