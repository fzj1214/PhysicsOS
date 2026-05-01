from physicsos.tools.surrogate_tools import ListSurrogateModelsInput, list_available_surrogates


def main() -> None:
    result = list_available_surrogates(ListSurrogateModelsInput())
    for model in result.models:
        has_checkpoint = result.checkpoint_status[model.id]
        checkpoint = model.checkpoint.uri if model.checkpoint else "none"
        print(f"{model.id}: runner={model.runner} checkpoint_exists={has_checkpoint} checkpoint={checkpoint}")


if __name__ == "__main__":
    main()

