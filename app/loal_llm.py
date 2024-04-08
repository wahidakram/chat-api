from langchain_community.llms import OpenLLM

from config import config

# Load the OpenLLM model
server_url = config.SERVER_URL


def load_openllm():
    openllm = OpenLLM(server_url=server_url)
    return openllm
