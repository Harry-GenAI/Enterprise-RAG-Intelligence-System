from logger import logger


def rewrite_query(llm, history, question):

    prompt = f"""
    You are Query Rewriting Assistant.

    Your job is convert the user query into a clear, standalone question
    optimized for semantic search in knowledge base.

    Rules:
    - Remove ambiguity
    - Expand vague terms
    - Ensure gramatically correct English
    - User formal language
    - Keep meaning same
    - Don't answer the question

    Chat History:
    {history}

    User Question:
    {question}

    Rewritten Query:
    """

    response = llm(prompt)
    return response.strip()
        
