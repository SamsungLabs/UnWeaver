from .literals import ChunkingStrategy, RetrievalMethod, SystemName, VectorStorageName

DEFAULT_KEY = "somekey-4321"

# MLflowConfig
DEFAULT_REMOTE_MLFLOW_URI: str = "http://IP_ADDRESS:PORT/"
DEFAULT_EXPERIMENT_NAME_INDEX: str = "Unweaver Indexing"
DEFAULT_EXPERIMENT_NAME_QUERY: str = "Unweaver Querying"


#### GENERAL CONFIG

USE_CACHE_LLM: bool = True
USE_CACHE_EMBEDDER: bool = False
USE_WHICH_SYSTEM: SystemName = "unweaver"

VECTOR_STORAGE_NAME: VectorStorageName = "lancedb"

# LLMConfig
LLM_MAX_ASYNC: int = 8
LLM_BASE_URLS: list[str] = [
    "http://IP_ADDRESS:PORT1/v1",
    "http://IP_ADDRESS:PORT2/v1",
    "http://IP_ADDRESS:PORT3/v1",
    "http://IP_ADDRESS:PORT4/v1",
]
LLM_API_KEYS: list[list[str]] = [
    [DEFAULT_KEY],
    [DEFAULT_KEY],
    [DEFAULT_KEY],
    [DEFAULT_KEY],
]
LLM_MODEL: str = "openai/gpt-oss-120b"
LLM_TIMEOUT: int = 300
LLM_MAX_TOKEN_SIZE: int = 32768
LLM_POSTPROCESS_THINKING: bool = True
LLM_ENABLE_THINKING: bool = False

# TokenizerConfig
TOKENIZER_MODEL: str = "openai/gpt-oss-120b"

# EmbeddingConfig
EMBEDDER_MAX_ASYNC: int = 8
EMBEDDER_BASE_URLS: list[str] = [
    "http://IP_ADDRESS:PORT1",
    "http://IP_ADDRESS:PORT2",
    "http://IP_ADDRESS:PORT3",
    "http://IP_ADDRESS:PORT4",
]
EMBEDDER_API_KEYS: list[list[str]] = [
    [DEFAULT_KEY],
    [DEFAULT_KEY],
    [DEFAULT_KEY],
    [DEFAULT_KEY],
]
EMBEDDER_MODEL: str = "Qwen/Qwen3-Embedding-4B"
EMBEDDER_DIM: int = 2560
EMBEDDER_MAX_TOKEN_SIZE: int = 40960
EMBEDDER_BATCH_SIZE: int = 32
EMBEDDER_TIMEOUT: int = 300

# PromptPathsConfig
EXTRACTION_PROMPT = "prompts/extraction.txt"
QUERY_PROMPT = "prompts/query.txt"
SUMMARIZATION_PROMPT = "prompts/summarization.txt"
NO_DATA_RESPONSE = "prompts/no_data_response.txt"

#### QUERY CONFIG

MOCK_RETRIEVAL: bool = False
RETRIEVAL_METHODS: list[RetrievalMethod] = ["unweaver"]

# NaiveQueryConfig
NAIVE_RETRIEVE_TOP_K: int = 5
NAIVE_CHUNKS_MAX_TOKEN_SIZE: int = 4000
NAIVE_CHUNKS_TABLE_FORMAT: str = "csv"

# UnweaverQueryConfig
UNWEAVER_RETRIEVE_TOP_K_ENTS: int = 10
UNWEAVER_RETRIEVE_TOP_K_CHUNKS: int = 5
UNWEAVER_CHUNKS_MAX_TOKEN_SIZE: int = 4000
UNWEAVER_CHUNKS_TABLE_FORMAT: str = "csv"


#### INDEX CONFIG

# ChunkingConfig
CHUNK_STRATEGY: ChunkingStrategy = "by_token_size"
CHUNK_MAX_TOKEN_SIZE: int = 1500
CHUNK_OVERLAP_TOKEN_SIZE: int = 128

# ExtractionConfig
SUMMARIZATION_THRESHOLD: int = 4000
