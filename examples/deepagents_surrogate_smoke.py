from langchain_core.language_models.fake_chat_models import FakeListChatModel

from physicsos.agents import create_physicsos_agent
from physicsos.backends.surrogate_runtime import list_surrogate_models


def main() -> None:
    # FakeListChatModel verifies graph construction only. It does not implement
    # tool-calling, so use a real tool-capable model for end-to-end invocation.
    agent = create_physicsos_agent(model=FakeListChatModel(responses=["ok"]))
    print(f"agent={type(agent).__name__}")
    print("surrogate_models=" + ",".join(model.id for model in list_surrogate_models()))


if __name__ == "__main__":
    main()

