from stable_eureka.llm_generator_base import LLMGeneratorBase


class OpenAIGenerator(LLMGeneratorBase):
    def __init__(self, model: str):
        super().__init__(model)

        # TODO: check if the model is valid or pulled, get the API key

    def generate(self, prompt, k, logger, temperature):
        ...
