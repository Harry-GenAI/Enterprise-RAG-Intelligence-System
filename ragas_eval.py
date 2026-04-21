from datasets import Dataset
from ragas import evaluate
import asyncio
from rag import retrieve_context
from prompts import build_prompt
from llmservice import generate_reply
from dotenv import load_dotenv

#for langsmith tracing
load_dotenv()

#Inputs
questions = [
    "What is refund processing time?",
    "Can employees misuse company systems?",
    "Is MFA required?",
    "Can confidential data be shared?",
    "How often should passwords be rotated?"
]

ground_truths = [
    "Refund processing time is 5-7 business days.",
    "Employees must use company systems responsibly.",
    "Multi-factor authentication is mandatory.",
    "Confidential data must not be shared externally.",
    "Passwords must be rotated every 90 days."
]

#Ragas Pipeline
async def run_ragas():
    answers=[]
    contexts = []
    
    for query in questions:
        #get contexts
        context, _, _, _ = await asyncio.to_thread(retrieve_context, query, None)

        #get prompt
        prompt = build_prompt("", context, query)

        #get answer
        answer = await generate_reply(prompt)

        #collect generated answers and contexts
        answers.append(answer)
        contexts.append([context])
        
        
    #Dataset
    dataset = Dataset.from_dict({
        "question":questions,
        "ground_truth":ground_truths,
        "answer":answers,
        "contexts":contexts

    })

    #Evaluate
    results = evaluate(dataset)

    print("\n📚Results:\n")
    print(results)
    #print("\n\n", contexts, "\n\n", answers)

#main
if __name__ == "__main__":
    asyncio.run(run_ragas())