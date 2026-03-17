"""CLI entry point for the suggestion agent."""

from agent.agent import Agent
from config import settings
from tools import registry


def main():
    settings.tool_registry = registry
    agent = Agent(settings=settings)
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        response = agent.run(user_input)
        print(f"Agent: {response}")


if __name__ == "__main__":
    main()
