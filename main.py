from agents.agent import run_agent


def main():
    result = run_agent()
    print(result["messages"][-1].content_blocks)


if __name__ == "__main__":
    main()
