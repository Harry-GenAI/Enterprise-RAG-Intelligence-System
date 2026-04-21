import json
import boto3
from logger import logger

# Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

def call_llm(prompt: str, mode: str = "answer"):
    """
    mode:
    - "rewrite" → fast + cheap (Haiku)
    - "answer"  → smart + reasoning (Sonnet)
    """

    if mode == "rewrite":
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        logger.info("Calling Haiku (Query Rewriter)")
    else:
        model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        logger.info("Calling Claude 3.5 Sonnet v2 (Answer Generation)")

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "temperature": 0.3
    }

    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())

    return result["content"][0]["text"].strip()
