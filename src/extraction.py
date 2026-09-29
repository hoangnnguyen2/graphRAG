from .schemas import GraphExtraction
from google.genai import types
from google import genai

class Extraction:
    def __init__(self, API_KEY, MODEL_NAME):
        self.client = genai.Client(api_key=API_KEY)
        self.MODEL_NAME = MODEL_NAME
        
    def extract(self, text):
        prompt = f"""
You are an expert Knowledge Graph engineer. Your task is to extract all significant entities and their explicit semantic relationships from the provided text according to the required schema.

### EXTRACTION RULES:
1. **Deduplication**: Ensure `entity_name` is canonical, normalized, and strictly lowercase.
2. **Coverage**: Capture all active subjects, groups, or domain concepts. Do NOT leave valid interacting entities isolated.
3. **Integrity**: `source` and `target` in relationships MUST strictly match the `entity_name` of an entity defined in the entities list.
4. **Strict Factuality**: Extract only what is explicitly supported by the text. Do not hallucinate or extrapolate.

TEXT:
{text}"""
        response = self.client.models.generate_content(
            model=self.MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GraphExtraction,
                temperature=0
            )
        )

        if response.parsed:
            return response.parsed
        return GraphExtraction.model_validate_json(response.text or "{}")
