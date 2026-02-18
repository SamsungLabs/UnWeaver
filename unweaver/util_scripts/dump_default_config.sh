#!/bin/bash
cd "$(dirname "$0")/.."

poetry run python -c 'import json;import unweaver.configurations as cf;default_config = cf.RAGConfig();print(json.dumps(default_config.model_dump(), indent=2));'
