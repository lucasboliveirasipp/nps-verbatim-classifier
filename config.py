# Carrega variáveis do arquivo .env mais próximo (pasta do projeto)
def _carregar_env():
    from pathlib import Path
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for pasta in Path(__file__).resolve().parents:
        if (pasta / ".env").exists():
            load_dotenv(pasta / ".env")
            break


_carregar_env()

from langchain_openai import AzureChatOpenAI
from typing import TypedDict
from pymongo import MongoClient
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

os.environ["API_VERSION"] = "2024-02-15-preview"
os.environ["GPT4"] = "gpt-4o-mini"
os.environ["GPT_VERBATINS"] = "1-mini-2025-04-14-nps-verbatins"
os.environ["GPT_VERBATINS_API_VERSION"] = "2024-12-01-preview"

###### Model ######
llm_modelo = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    openai_api_version=os.getenv("GPT_VERBATINS_API_VERSION"),
    deployment_name=os.getenv("GPT_VERBATINS"),
    openai_api_type="azure",
    temperature=0,
    api_key=os.getenv("GPT_VERBATINS_API_KEY"),
)

sentiment_client = TextAnalyticsClient(
    endpoint=os.getenv("AZURE_LANGUAGE_ENDPOINT", ""), 
    credential=AzureKeyCredential(os.getenv("AZURE_LANGUAGE_KEY", ""))
)


gpt_4_mini = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    openai_api_version="2025-01-01-preview",
    deployment_name="gpt-4o-mini",
    openai_api_type="azure",
    temperature=0,
    top_p=0,
)

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    openai_api_version="2025-01-01-preview",
    deployment_name="gpt-4o-mini",
    openai_api_type="azure",
    temperature=0,
)

gpt_4_1_mini = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    openai_api_version="2024-08-01-preview",
    deployment_name="gpt-4.1-mini",
    openai_api_type="azure",
    temperature=0,
)


class DBConnector:
    def __init__(self, connection_string: str, database_name: str):
        self.client = MongoClient(connection_string)
        self.database = self.client[database_name]

    def get_db(self):
        return self.database


MONGO = DBConnector(
    "mongodb://localhost:27017", "feedbacks"
).get_db()
