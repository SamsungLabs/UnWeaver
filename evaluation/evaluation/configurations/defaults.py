from .literals import EmbedderMode

#### INDEX CONFIG


DEFAULT_KEY: str = "somekey-4321"


DEFAULT_REMOTE_MLFLOW_URI: str = "http://IP_ADDRESS:PORT/"
DEFAULT_EXPERIMENT_NAME: str = "UnweaverRAGEval"

DEFAULT_NUM_REPEATS = 3
DEFAULT_METRICS = [
    "factual_correctness",
]


# LlmConfig
LLM_BASE_URLS: list[str] = ["http://IP_ADDRESS:PORT/v1"]
LLM_API_KEYS: list[list[str]] = [[DEFAULT_KEY]]
LLM_MODEL: str = "openai/gpt-oss-120b"
LLM_TIMEOUT: int = 300
LLM_MAX_RETRIES = 10
LLM_MAX_WORKERS = 128


# EmbeddingConfig
EMBEDDER_MAX_ASYNC: int = 8
EMBEDDER_MODE: EmbedderMode = "remote"
EMBEDDER_BASE_URLS: list[str] = ["http://IP_ADDRESS:PORT"]
EMBEDDER_API_KEYS: list[list[str]] = [[DEFAULT_KEY]]
EMBEDDER_MODEL: str = "Qwen/Qwen3-Embedding-4B"
EMBEDDER_DIM: int = 2560
EMBEDDER_MAX_TOKEN_SIZE: int = 32768
EMBEDDER_BATCH_SIZE: int = 1
EMBEDDER_TIMEOUT: int = 300
