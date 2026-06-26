import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# The model will automatically pick up the LangSmith environment variables
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("Sending test query to LLM...")
response = llm.invoke("What is the difference between standard RAG and Agentic RAG in one sentence?")
print("\nResponse:")
print(response.content)
print("\n✅ Check your LangSmith dashboard! You should see this trace under the 'rag-research-assistant' project.")