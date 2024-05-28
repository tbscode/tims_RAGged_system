from dataclasses import dataclass
import openai
import os

@dataclass
class Backends:
    OPENAI = "openai"
    DEEPINFRA = "deepinfra"
    GROQ = "groq"

@dataclass
class BackendConfig:
    api_key: str
    name: str
    base_url: str = None

@dataclass
class ModelBackend:
    model: str
    supports_json: bool
    supports_tools: bool
    supports_functions: bool
    client_config: BackendConfig
    
BACKENDS = {
    Backends.OPENAI: BackendConfig(
        name=Backends.OPENAI,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=None
    ),
    Backends.DEEPINFRA: BackendConfig(
        name=Backends.DEEPINFRA,
        api_key=os.getenv("DEEPINFRA_API_KEY"),
        base_url="https://api.deepinfra.com/v1/openai"
    ),
    Backends.GROQ: BackendConfig(
        name=Backends.GROQ,
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
}

MODELS = [
    ModelBackend(
        model="llama3-70b-8192",
        supports_json=False,
        supports_tools=True,
        supports_functions=True,
        client_config=BACKENDS[Backends.GROQ]
    ),
    ModelBackend(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        supports_json=False,
        supports_tools=False,
        supports_functions=False,
        client_config=BACKENDS[Backends.DEEPINFRA]
    ),
    ModelBackend(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        supports_json=False,
        supports_tools=False,
        supports_functions=False,
        client_config=BACKENDS[Backends.DEEPINFRA]
    ),
    ModelBackend(
        model="databricks/dbrx-instruct",
        supports_json=False,
        supports_tools=False,
        supports_functions=False,
        client_config=BACKENDS[Backends.DEEPINFRA]
    ),
    ModelBackend(
        model="cognitivecomputations/dolphin-2.6-mixtral-8x7b",
        supports_json=False,
        supports_tools=False,
        supports_functions=False,
        client_config=BACKENDS[Backends.DEEPINFRA]
    ),
    ModelBackend(
        model="gpt-3.5-turbo",
        supports_json=False,
        supports_tools=True,
        supports_functions=True,
        client_config=BACKENDS[Backends.OPENAI]
    ),
    ModelBackend(
        model="gpt-4-turbo",
        supports_json=False,
        supports_tools=True,
        supports_functions=True,
        client_config=BACKENDS[Backends.OPENAI]
    ),
    ModelBackend(
        model="gpt-4o",
        supports_json=False,
        supports_tools=True,
        supports_functions=True,
        client_config=BACKENDS[Backends.OPENAI]
    )
]

def get_client_for_model(
    model: str,
    async_client: bool = False
):
    model = get_model(model)
    if async_client:
        client = create_async_client(model.client_config)
    else:
        client = create_client(model.client_config)
    return client

def create_async_client(
    backend: BackendConfig
):
    client = openai.AsyncOpenAI(
        api_key=backend.api_key,
        base_url=backend.base_url
    )
    return client

def create_client(
    backend: BackendConfig
):
    client = openai.OpenAI(
        api_key=backend.api_key,
        base_url=backend.base_url
    )
    return client

def get_model(model_name):
    for model in MODELS:
        if model.model == model_name:
            return model
    return None