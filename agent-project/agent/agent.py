"""High-level agent loop that coordinates planner and executor."""

import json

from agent.executor import Executor
from agent.memory import Memory
from agent.planner import Planner, Plan
from agent.reactor import Reactor
from utils.logger import get_logger
from config import Settings


class Agent:
    def __init__(self, settings: Settings):
        self.logger = get_logger()
        self.memory = Memory()
        self.planner = Planner(settings=settings, logger=self.logger)
        self.executor = Executor(settings=settings, logger=self.logger)
        self.reactor = Reactor(settings=settings, logger=self.logger)
        self.settings = settings

    def _finalize(self, content: str, trace_id: str) -> str:
        self.memory.add_agent_message(content, trace_id=trace_id)
        self.memory.maybe_summarize(self.planner.llm, trace_id=trace_id)
        return content

    def _summarize_context(self, question: str, context_chunks: list, trace_id: str) -> str:
        """Summarize retrieved chunks to reduce length and avoid verbatim copying."""
        parts = []
        for i, c in enumerate(context_chunks):
            text = (c.get("text", "") or "")[:300]
            meta = c.get("metadata", {}) or {}
            src = meta.get("source", "unknown")
            center = meta.get("center_chunk_id", meta.get("chunk_id", "?"))
            chunk_ids = meta.get("chunk_ids", None)
            extra = ""
            if isinstance(chunk_ids, list) and chunk_ids:
                extra = f", chunk_ids:{chunk_ids}"
            parts.append(f"[{i+1}] (source:{src}, center_chunk:{center}{extra}) {text}")
        snippets = "\n".join(parts)
        prompt = (
            f"用户问题：{question}\n"
            "请用自己的话总结以下片段，提取与用户问题最相关的要点，限制在180字以内。"
            "禁止逐字复制原文，单句话中不得有超过12个连续原文字符。"
            "保留引用编号（如[1][2]），以便标注来源。\n"
            f"{snippets}\n"
            "输出：直接给出最终回答（含引用编号）。"
        )
        resp = self.planner.llm.chat_plain(prompt, trace_id=trace_id)
        return str(resp)

    def _retrieval_refs_for_memory(self, *, query: str, tool_result: dict) -> str:
        rerank_backend = tool_result.get("rerank_backend")
        hits = tool_result.get("output") or []
        refs = []
        if isinstance(hits, list):
            for h in hits[:10]:
                meta = (h or {}).get("metadata") or {}
                refs.append(
                    {
                        "source": meta.get("source"),
                        "center_chunk_id": meta.get("center_chunk_id", meta.get("chunk_id")),
                        "chunk_ids": meta.get("chunk_ids"),
                    }
                )
        payload = {
            "query": query,
            "rerank_backend": rerank_backend,
            "hits": refs,
        }
        # Keep it compact and stable in history.
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def run(self, user_input: str) -> str:
        trace_id = self.logger.new_trace_id()
        self.memory.add_user_message(user_input, trace_id=trace_id)
        self.memory.maybe_summarize(self.planner.llm, trace_id=trace_id)

        if self.settings.mode == "react":
            return self.reactor.run(self.memory, self.planner.llm, trace_id=trace_id)

        for step_idx in range(self.executor.max_steps):
            plan = self.planner.decide(self.memory, trace_id=trace_id)
            self.logger.info(
                "planner.step",
                extra={"trace_id": trace_id, "step": step_idx, "plan": plan},
            )

            if plan.action == "final":
                return self._finalize(plan.output or "", trace_id=trace_id)

            tool_result = self.executor.execute(plan, trace_id=trace_id)

            # 检索工具成功：用上下文生成最终回答, 失败则回退到搜索
            if plan.tool_name == "retrieval":
                # Only store references in memory to avoid stuffing chunks into the prompt.
                q = str((plan.tool_input or {}).get("query", user_input))
                try:
                    refs = self._retrieval_refs_for_memory(query=q, tool_result=tool_result)
                    self.memory.add_tool_message_compact(
                        content=refs, tool_name="retrieval", trace_id=trace_id
                    )
                except Exception:
                    self.memory.add_tool_message(tool_result, "retrieval", trace_id=trace_id)

                if tool_result.get("status") != "ok":
                    search_result = self.executor.execute(
                        Plan(
                            action="tool",
                            tool_name="search",
                            tool_input={"query": user_input},
                        ),
                        trace_id=trace_id,
                    )
                    self.memory.add_tool_message(
                        search_result, "search", trace_id=trace_id
                    )
                    if search_result.get("status") == "ok":
                        out = search_result.get("output", "")
                        self.memory.add_agent_message(out, trace_id=trace_id)
                        return out
                else:
                    context = tool_result.get("output", [])
                    summary = self._summarize_context(user_input, context, trace_id=trace_id)
                    self.logger.info(
                        "rag.summary",
                        extra={"trace_id": trace_id, "summary": summary},
                    )
                    return self._finalize(summary, trace_id=trace_id)

            # 先记录其他工具输出（允许完整输出，因为通常比较短）
            self.memory.add_tool_message(
                tool_result, plan.tool_name or "unknown", trace_id=trace_id
            )

            # 其他工具成功：直接返回结果
            if tool_result.get("status") == "ok":
                final_output = tool_result.get("output", "")
                return self._finalize(str(final_output), trace_id=trace_id)

        return "Reached max steps without conclusion."
