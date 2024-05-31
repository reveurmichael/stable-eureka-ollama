from stable_eureka.llm_generator_base import LLMGeneratorBase
import ollama
import time


class OllamaGenerator(LLMGeneratorBase):
    def __init__(self, model: str):
        super().__init__(model)

        # TODO: check if the model is valid or pulled

    def generate(self, prompt, k, logger, temperature):

        options = ollama.Options(
            temperature=temperature
        )

        responses = []
        for i in range(k):
            init_t = time.time()
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=False,
                options=options
            )
            end_t = time.time()
            responses.append(response)
            if logger is not None:
                logger.info(f"Generation {i + 1}/{k} completed in {end_t - init_t:.2f} seconds")
                logger.info(f"Response: {response['message']['content']}")
                logger.info("+---------------------------------+")

        return responses  # TODO: this should be generic, not dependant of ollama
