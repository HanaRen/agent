"""High-level agent loop that coordinates planner and executor."""

from agent.executor import Executor
from agent.memory import Memory
from agent.planner import Planner
from utils.logger import get_logger
from config import Settings


class Agent:
    def __init__(self, settings: Settings):
        self.logger = get_logger()
        self.memory = Memory()
        self.planner = Planner(settings=settings, logger=self.logger)
        self.executor = Executor(settings=settings, logger=self.logger)

    def run(self, user_input: str) -> str:
        trace_id = self.logger.new_trace_id()
        self.memory.add_user_message(user_input, trace_id=trace_id)

        for step_idx in range(self.executor.max_steps):
            plan = self.planner.decide(self.memory, trace_id=trace_id)
            self.logger.info(
                "planner.step",
                extra={"trace_id": trace_id, "step": step_idx, "plan": plan},
            )

            if plan.action == "final":
                self.memory.add_agent_message(plan.output, trace_id=trace_id)
                return plan.output

            tool_result = self.executor.execute(plan, trace_id=trace_id)
            self.memory.add_tool_message(tool_result, trace_id=trace_id)

        return "Reached max steps without conclusion."
