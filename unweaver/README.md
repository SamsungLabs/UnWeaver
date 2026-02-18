# UnWeaver

UnWeaver is a novel Retrieval-Augmented Generation (RAG) system that implements the approach described in the paper "UnWeaving the knots of GraphRAG - turns out VectorRAG is almost enough". It provides an alternative to graph-based knowledge representations for RAG applications.

## Overview

UnWeaver implements a chunk-based retrieval approach with entity extraction and summarization capabilities. The system supports multiple retrieval methods and can be configured for various use cases.

## Project Structure

```
unweaver/
├── configs/                      # Configuration files
│   ├── custom.json              # Custom configuration (user-editable)
│   └── dumped_default_config.json # Default configuration dump
├── prompts/                      # Prompt templates
│   ├── extraction.txt           # Entity extraction prompt
│   ├── no_data_response.txt     # Response when no data is found
│   ├── query.txt                # Query generation prompt
│   └── summarization.txt        # Summarization prompt
├── unweaver/                     # Main package
│   ├── __init__.py
│   ├── base.py                  # Base classes and utilities
│   ├── content_handlers.py      # Content processing handlers
│   ├── index.py                 # Indexing module (entry point)
│   ├── query.py                 # Query module (entry point)
│   ├── rag.py                   # RAG pipeline implementation
│   ├── usage_monitors.py        # Usage monitoring and tracking
│   ├── utils.py                 # Utility functions
│   ├── chunkers/                # Text chunking strategies
│   ├── configurations/          # Configuration management
│   ├── embedders/               # Embedding model implementations
│   ├── llms/                    # LLM implementations
│   ├── retrievers/              # Retrieval strategies
│   ├── storage/                 # Storage backends (vector DBs)
│   └── tokenizers/              # Tokenizer implementations
├── util_scripts/                 # Utility scripts
│   └── dump_default_config.sh   # Script to dump default config
├── pyproject.toml               # Poetry dependencies
├── poetry.lock                  # Poetry lock file
├── run.sh                       # Main execution script
└── README.md                    # This file
```

## Entry Points

UnWeaver provides two main entry points for indexing and querying:

### 1. Indexing Module

**Module**: `unweaver.index`

**Purpose**: Index documents and create vector embeddings for retrieval

**Usage**:
```bash
poetry run python -m unweaver.index \
  <data_path> \
  <index_path> \
  --config <config_path>
```

**Arguments**:
- `data_path`: Path to the directory containing preprocessed documents
- `index_path`: Path where the index will be stored
- `--config`: Path to the configuration file (can be specified multiple times)
- `--extra`: Extra configuration values in `key:value` format

**What it does**:
1. Reads documents from the data path
2. Chunks documents according to the configured strategy
3. Extracts entities and creates summaries
4. Generates embeddings for chunks and entities
5. Stores everything in the specified vector database

### 2. Query Module

**Module**: `unweaver.query`

**Purpose**: Query the indexed documents and generate answers

**Usage**:
```bash
poetry run python -m unweaver.query \
  <questions_path> \
  <index_path> \
  --run_name <run_name> \
  --config <config_path>
```

**Arguments**:
- `questions_path`: Path to the JSON file containing questions
- `index_path`: Path to the index created by the indexing module
- `--run_name`: Name for this query run (used for result organization)
- `--config`: Path to the configuration file (can be specified multiple times)
- `--extra`: Extra configuration values in `key:value` format

**What it does**:
1. Loads questions from the questions file
2. Retrieves relevant chunks/entities using configured retrieval methods
3. Generates answers using the LLM
4. Stores results in the index directory

### 3. Run Script

**Script**: `run.sh`

**Purpose**: Automate indexing and querying for multiple datasets

**Usage**:
```bash
./run.sh
```

**What it does**:
- Iterates through configured datasets (covidqa, emanual, techqa)
- Runs indexing for each dataset
- Runs querying for each dataset
- Stores all results in respective index directories

## Configuration

UnWeaver uses a hierarchical JSON configuration system. The main configuration file is `configs/custom.json`.

### Configuration Structure

The configuration is divided into three main sections:

#### 1. General Settings (`general`)

```json
{
  "general": {
    "use_cache_llm": true,
    "use_cache_embedder": false,
    "use_which_system": "unweaver",
    "vector_storage_name": "lancedb",
    "remote_mlflow_uri": "http://mlflow-server:port/",
    "experiment_name_index": "Unweaver Indexing",
    "experiment_name_query": "Unweaver Querying",
    "llm": { ... },
    "tokenizer": { ... },
    "embedder": { ... },
    "prompts": { ... }
  }
}
```

**Key parameters**:
- `use_cache_llm`: Enable/disable LLM response caching (requires MongoDB)
- `use_cache_embedder`: Enable/disable embedding caching (requires MongoDB)
- `use_which_system`: Which RAG system to use ("unweaver" or "naive")
- `vector_storage_name`: Vector database backend ("lancedb")
- `remote_mlflow_uri`: MLflow tracking server URI
- `experiment_name_index`: MLflow experiment name for indexing runs
- `experiment_name_query`: MLflow experiment name for query runs

#### 2. LLM Configuration (`general.llm`)

```json
{
  "llm": {
    "max_async": 128,
    "base_urls": ["http://llm-server:port/v1"],
    "api_keys": [["your-api-key"]],
    "model": "openai/gpt-oss-120b",
    "timeout": 300,
    "max_token_size": 32768,
    "postprocess_thinking": true,
    "enable_thinking": false
  }
}
```

**Key parameters**:
- `max_async`: Maximum number of concurrent LLM requests
- `base_urls`: List of LLM API endpoints
- `api_keys`: API keys for each endpoint
- `model`: Model name/identifier
- `timeout`: Request timeout in seconds
- `max_token_size`: Maximum token size for responses
- `postprocess_thinking`: Enable thinking post-processing
- `enable_thinking`: Enable thinking mode

#### 3. Embedder Configuration (`general.embedder`)

```json
{
  "embedder": {
    "max_async": 32,
    "base_urls": ["http://embedder-server:port"],
    "api_keys": [["your-api-key"]],
    "model": "Qwen/Qwen3-Embedding-4B",
    "embedding_dim": 2560,
    "max_token_size": 40960,
    "batch_size": 32,
    "timeout": 300
  }
}
```

**Key parameters**:
- `max_async`: Maximum number of concurrent embedding requests
- `base_urls`: List of embedder API endpoints
- `api_keys`: API keys for each endpoint
- `model`: Embedding model name
- `embedding_dim`: Dimension of embeddings
- `max_token_size`: Maximum token size for input
- `batch_size`: Batch size for embedding generation
- `timeout`: Request timeout in seconds

#### 4. Query Configuration (`query`)

```json
{
  "query": {
    "mock_retrieval": false,
    "retrieval_methods": ["unweaver"],
    "naive": {
      "retrieve_top_k": 5,
      "chunks_max_token_size": 4000,
      "chunks_table_format": "csv"
    },
    "unweaver": {
      "retrieve_top_k_ents": 10,
      "retrieve_top_k_chunks": 5,
      "chunks_max_token_size": 4000,
      "chunks_table_format": "csv"
    }
  }
}
```

**Key parameters**:
- `mock_retrieval`: Use mock retrieval for testing
- `retrieval_methods`: List of retrieval methods to use
- `naive`: Configuration for naive chunk-based retrieval
  - `retrieve_top_k`: Number of chunks to retrieve
  - `chunks_max_token_size`: Maximum token size for chunks
  - `chunks_table_format`: Format for chunk tables ("csv", "json")
- `unweaver`: Configuration for UnWeaver retrieval
  - `retrieve_top_k_ents`: Number of entities to retrieve
  - `retrieve_top_k_chunks`: Number of chunks to retrieve
  - `chunks_max_token_size`: Maximum token size for chunks
  - `chunks_table_format`: Format for chunk tables

#### 5. Index Configuration (`index`)

```json
{
  "index": {
    "chunker": {
      "strategy": "by_token_size",
      "max_token_size": 1500,
      "overlap_token_size": 128
    },
    "extraction": {
      "summarization_threshold": 4000
    }
  }
}
```

**Key parameters**:
- `chunker`: Text chunking configuration
  - `strategy`: Chunking strategy ("by_token_size", "by_sentence", etc.)
  - `max_token_size`: Maximum token size per chunk
  - `overlap_token_size`: Token overlap between chunks
- `extraction`: Entity extraction configuration
  - `summarization_threshold`: Token threshold for summarization

### Configuration Override

You can override configuration values in multiple ways:

1. **Multiple config files**: Specify multiple `--config` arguments (later files override earlier ones)
2. **Extra values**: Use `--extra key:value` to override specific values
3. **Short prefixes**: Use short prefixes for top-level sections (e.g., `g.llm.model` for `general.llm.model`)

Example:
```bash
poetry run python -m unweaver.index \
  data/ \
  index/ \
  --config configs/custom.json \
  --extra g.llm.model:new-model \
  --extra q.retrieve_top_k:10
```

## Output Structure

After running UnWeaver, the index directory will contain:

```
index_<dataset_name>/
├── config.json                 # Index configuration snapshot
├── RUN_ID                      # Unique run identifier
├── indexing_logs.log           # Indexing logs
├── vdb_chunks/                 # Vector database for chunks
├── vdb_entities/               # Vector database for entities
└── query_results/              # Query results
    ├── <run_name>_results.json      # Query results
    └── <run_name>_query_config.json # Query configuration
```

## Retrieval Methods

UnWeaver supports multiple retrieval methods:

### 1. UnWeaver Retrieval

The novel approach described in the paper:
- Extracts entities from documents
- Retrieves relevant entities and chunks
- Uses entity information to improve retrieval quality

### 2. Naive Retrieval

Traditional chunk-based retrieval (VectorRAG):
- Retrieves chunks based on semantic similarity
- No entity extraction or summarization

## Dependencies

Key dependencies include:
- `lancedb`: Vector database for storage
- `openai`: LLM API client
- `sentence-transformers`: Embedding models
- `mlflow`: Experiment tracking
- `pymongo`: MongoDB client (for caching)

See `pyproject.toml` for the complete list of dependencies.

## Troubleshooting

### MongoDB Connection Issues

If using caching with MongoDB, ensure:
- MongoDB is running and accessible
- Environment variables are set:
  - `MONGO_USER`: MongoDB username
  - `MONGO_PASSWORD`: MongoDB password
  - `MONGO_URL`: MongoDB URL and port

### LLM/Embedder API Issues

- Check that API endpoints are accessible
- Verify API keys are correct
- Adjust timeout values if requests are timing out
- Reduce `max_async` if experiencing rate limiting

### Memory Issues

- Reduce `max_async` values for LLM and embedder
- Reduce `batch_size` for embedder
- Use smaller `max_token_size` values

## See Also

- [Root README](../README.md) - Project overview
- [Evaluation README](../evaluation/README.md) - Evaluation framework documentation
