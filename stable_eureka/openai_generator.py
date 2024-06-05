import os
from openai import OpenAI
from stable_eureka.llm_generator_base import LLMGeneratorBase
import time


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)


class OpenAIGenerator(LLMGeneratorBase):
    def __init__(self, model: str):
        super().__init__(model)

        if OPENAI_API_KEY is None:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable")

        self.openai_client = OpenAI()

    def generate(self, prompt, k, logger, temperature):
        init_t = time.time()
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=temperature,
            n=k
        )
        end_t = time.time()

        responses = [choice.message.content for choice in response.choices]
        if logger is not None:
            logger.info(f"Total generation completed in {end_t - init_t:.2f} seconds")
            for i, response in enumerate(responses):
                logger.info(f"Response {i + 1}: {response}")
                logger.info("+---------------------------------+")

        return responses
