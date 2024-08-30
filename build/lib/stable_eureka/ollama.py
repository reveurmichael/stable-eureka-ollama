from stable_eureka.ollama_generator import OllamaGenerator
import logging

def call_ollama_generator(model: str, prompt: str, k: int, temperature: float):
    # 设置日志记录
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 创建 OllamaGenerator 实例
    generator = OllamaGenerator(model)

    # 调用 generate 方法
    responses = generator.generate(prompt, k, logger, temperature)

    # 打印生成的响应
    for i, response in enumerate(responses):
        print(f"Response {i + 1}: {response}")

# 示例调用
if __name__ == "__main__":
    model_name = "llama3:instruct"  # 替换为您要使用的模型名称
    user_prompt = "请给我一个关于机器学习的简要介绍。"  # 用户输入的提示
    num_responses = 3  # 生成的响应数量
    temp = 0.7  # 温度设置

    call_ollama_generator(model_name, user_prompt, num_responses, temp)