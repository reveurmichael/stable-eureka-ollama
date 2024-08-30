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
        responses = None
        for attempt in range(3):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=temperature,
                    n=k
                )

                responses = [choice.message.content for choice in response.choices]
                break

            except Exception as e:
                if logger is not None:
                    logger.warn(f"Error in generation: {e}, attempt {attempt + 1}/3")
                time.sleep(1)

            if not responses:
                raise ValueError("Error in generation after 3 attempts")

            end_t = time.time()

            if logger is not None:
                logger.info(f"Total generation completed in {end_t - init_t:.2f} seconds")
                for i, response in enumerate(responses):
                    logger.info(f"Response {i + 1}: {response}")
                    logger.info("+---------------------------------+")

        return responses
