import os
from openai import OpenAI
from stable_eureka.llm_generator_base import LLMGeneratorBase

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)


class OpenAIGenerator(LLMGeneratorBase):
    def __init__(self, model: str):
        super().__init__(model)

        if OPENAI_API_KEY is None:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable")

        self.openai = OpenAI()


    def generate(self, prompt, k, logger, temperature):
        ...
