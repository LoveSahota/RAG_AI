import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"


def keyword_score(query: str, text: str) -> int:
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    return len(query_words.intersection(text_words))


def retrieve_relevant_chunks(question: str, chunks: list[dict], top_k: int = 4) -> list[dict]:
    scored = []

    for chunk in chunks:
        score = keyword_score(question, chunk["content"])
        scored.append({
            "content": chunk["content"],
            "score": score
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def build_rag_prompt(question, relevant_chunks, history):
    context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])

    history_text = ""
    for msg in history[-5:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""
You are AnswerAI, an intelligent assistant that answers questions ONLY from the provided document context.

Instructions:
- Answer the question in a detailed and well-structured manner.
- Use simple and clear explanations.
- answer should be lenghty but within the context provided.
- Break the answer into sections:
   Explanation
   Key Points
   Conclusion
- Do NOT include any information that is not present in the document context.
- If the answer is not found in the document, respond with:
  "The answer was not found in the uploaded document."

Conversation History:
{history_text}

Document Context:
{context}

Question:
{question}

Answer:
"""
    return prompt


def ask_ai(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        data = response.json()
        return data.get("response", "No response received from model.")
    except Exception as e:
        return f"Ollama error: {str(e)}"