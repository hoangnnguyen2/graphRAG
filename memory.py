from google.genai import types
from google import genai

class ChatMemoryManager:
    def __init__(self, api_key, model_name, max_history_turns: int = 4):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_history_turns = max_history_turns
        # Lưu trữ theo session_id: {session_id: [{"role": ..., "content": ...}]}
        self.sessions = {}

    def get_history(self, session_id: str = "default") -> list:
        return self.sessions.get(session_id, [])

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})

    def clear(self, session_id: str = "default"):
        if session_id in self.sessions:
            self.sessions[session_id] = []

    def format_history_for_prompt(self, session_id: str = "default") -> str:
        history = self.get_history(session_id)[-self.max_history_turns:]
        if not history:
            return ""
        return "--- RECENT CONVERSATION HISTORY ---\n" + "\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in history]
        ) + "\n\n"

    def condense_question(self, user_query: str, session_id: str = "default") -> str:
        history = self.get_history(session_id)[-self.max_history_turns:]
        if not history:
            return user_query

        history_str = "\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in history[-self.max_history_turns:]]
        )

        prompt = f"""Based on the conversation history below and the user's latest query, rewrite the query into a STANDALONE, FULLY MEANINGFUL question (replace pronouns like 'he', 'it', 'there', etc. with the specific entities mentioned previously).
ONLY RETURN THE REWRITTEN QUESTION.
DO NOT ANSWER THE QUESTION AND DO NOT ADD ANY EXPLANATIONS.

CONVERSATION HISTORY:{history_str}
NEW QUERY: {user_query}

STANDALONE QUESTION:"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        condensed = response.text.strip() if response.text else ""
        return condensed or user_query
