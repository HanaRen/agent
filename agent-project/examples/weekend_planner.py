from agent.agent import Agent
from config import settings


def main():
    agent = Agent(settings=settings)
    query = "帮我计划一个上海周末亲子活动，预算500元。"
    print(agent.run(query))


if __name__ == "__main__":
    main()
