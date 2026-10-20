import json
import time

from deepeval.models import DeepEvalBaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI


class GeminiJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="gemini-2.5-flash", retries=5):
        self.model_name = model_name
        self.retries = retries
        self.model = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    def load_model(self):
        return self.model

    def _call(self, prompt, schema=None):
        for attempt in range(self.retries):
            try:
                if schema is None:
                    return self.model.invoke(prompt).content
                try:
                    return self.model.with_structured_output(schema).invoke(prompt)
                except Exception:
                    text = self.model.invoke(
                        prompt + "\nReturn only valid JSON."
                    ).content
                    text = text.strip().removeprefix("```json").removesuffix("```").strip()
                    return schema(**json.loads(text))
            except Exception as e:
                if attempt == self.retries - 1:
                    raise
                time.sleep(15 * (attempt + 1) if "429" in str(e) else 2)

    def generate(self, prompt, schema=None):
        return self._call(prompt, schema)

    async def a_generate(self, prompt, schema=None):
        return self._call(prompt, schema)

    def get_model_name(self):
        return self.model_name