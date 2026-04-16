"""ReAct-style runner: think-act-observe loop without separate planner/executor."""

import json

from agent.prompt import build_react_prompt
from agent.parser import parse_plan
from config import Settings


class Reactor:
    def __init__(self, settings: Settings, logger):
        self.settings = settings
        self.logger = logger
        self.max_steps = settings.max_steps
        self.tools = settings.tool_registry

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
        payload = {"query": query, "rerank_backend": rerank_backend, "hits": refs}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _summarize_context(self, *, question: str, context_chunks: list, llm, trace_id: str) -> str:
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
        return str(llm.chat_plain(prompt, trace_id=trace_id))

    def _last_user_text(self, memory) -> str:
        for m in reversed(getattr(memory, "messages", []) or []):
            if m.get("role") == "user":
                return str(m.get("content", ""))
        return ""

    def run(self, memory, llm, trace_id: str) -> str:
        for step in range(self.max_steps):
            prompt = build_react_prompt(memory.messages, self.tools.keys())
            plan_dict = llm.chat(prompt, trace_id=trace_id)
            if self.logger:
                self.logger.info(
                    "react.plan", extra={"trace_id": trace_id, "step": step, "plan": plan_dict}
                )
            plan = parse_plan(plan_dict)

            if plan.action == "final":
                memory.add_agent_message(plan.output or "", trace_id=trace_id)
                try:
                    memory.maybe_summarize(llm, trace_id=trace_id)
                except Exception:
                    pass
                return plan.output or ""

            if plan.action == "tool":
                tool = self.tools.get(plan.tool)
                if not tool:
                    memory.add_agent_message(f"Unknown tool {plan.tool}", trace_id=trace_id)
                    return f"Unknown tool {plan.tool}"
                result = tool.run(plan.tool_input or {}, trace_id=trace_id)
                if plan.tool == "retrieval":
                    q = str((plan.tool_input or {}).get("query", ""))
                    try:
                        refs = self._retrieval_refs_for_memory(query=q, tool_result=result)
                        memory.add_tool_message_compact(
                            content=refs, tool_name="retrieval", trace_id=trace_id
                        )
                    except Exception:
                        memory.add_tool_message(result, plan.tool or "unknown", trace_id=trace_id)
                else:
                    memory.add_tool_message(result, plan.tool or "unknown", trace_id=trace_id)
                try:
                    memory.maybe_summarize(llm, trace_id=trace_id)
                except Exception:
                    pass
                # short-circuit on success
                if result.get("status") == "ok":
                    if plan.tool == "retrieval":
                        context = result.get("output", [])
                        out = self._summarize_context(
                            question=self._last_user_text(memory),
                            context_chunks=context if isinstance(context, list) else [],
                            llm=llm,
                            trace_id=trace_id,
                        )
                        memory.add_agent_message(out, trace_id=trace_id)
                        try:
                            memory.maybe_summarize(llm, trace_id=trace_id)
                        except Exception:
                            pass
                        return out

                    out = str(result.get("output", ""))
                    memory.add_agent_message(out, trace_id=trace_id)
                    try:
                        memory.maybe_summarize(llm, trace_id=trace_id)
                    except Exception:
                        pass
                    return out
                continue

        return "Reached max steps without conclusion."
