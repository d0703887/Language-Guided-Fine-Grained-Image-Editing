from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.llms import Ollama
import json


SYSTEM_PROMPT = """You are an AI assistant that processes user instructions for image editing tasks.
Given a user's prompt, extract and format the necessary inputs for two models: Grounding DINO and a diffusion model.

Your output must be a JSON object with the following two fields:

1. "dino_input": list of objects to detect or mask.
   - Format:
     - Single object: ["a cat."]
     - Multiple objects: [["a face.", "a car."]]
   - Rules:
     - Lowercase
     - Singular
     - Start with "a "
     - End with "."

2. "diffusion_input": a descriptive sentence of the desired edited image region.
   - Focus on what should appear in the masked area.
   - Natural language description, e.g., "a fluffy orange cat being held by a man."

Example Input:
User: "Change the dog that the man is holding into a fluffy orange cat."

Example Output:
{{
  "dino_input": ["a dog."],
  "diffusion_input": "a fluffy orange cat being held by a man."
}}

Output only the JSON. Do not include any explanations or markdown.

User: {user_input}
"""

def load_llm(model: str):
    return Ollama(model=model)


def parse_json_output(text: str):
    try:
        return json.loads(text.strip())

    except Exception as e:
        print("[Warning] Failed to parse output as JSON.")
        return {"error": "Failed to parse JSON", "raw_output": text}


def load_chain(llm, lambda_function):
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    chain = prompt | llm | RunnableLambda(lambda_function)
    return chain


def parse_user_prompt(user_prompt: str, chain):
    result = chain.invoke({"user_input": user_prompt})
    return result["dino_input"], result["diffusion_input"]
