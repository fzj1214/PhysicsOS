from physicsos.agents import create_openai_compatible_model, create_physicsos_agent


def main() -> None:
    # Required:
    #   PHYSICSOS_OPENAI_API_KEY
    # Optional:
    #   PHYSICSOS_OPENAI_BASE_URL=https://api.tu-zi.com/v1
    #   PHYSICSOS_OPENAI_MODEL=gpt-5.4
    #   PHYSICSOS_OPENAI_USE_RESPONSES_API=false
    model = create_openai_compatible_model()
    agent = create_physicsos_agent(model=model)
    print(f"agent={type(agent).__name__}")


if __name__ == "__main__":
    main()

