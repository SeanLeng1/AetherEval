"""Model-specific environment for SGLang's isolated worker process."""


def reward_model_environment(architectures):
    if "GPT2ForSequenceClassification" not in (architectures or ()):
        return {}
    return {
        "SGLANG_EXTERNAL_MODEL_PACKAGE": __name__,
    }
