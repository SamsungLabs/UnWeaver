#!/bin/bash
cd "$(dirname "$0")/.."

poetry run python -c 'import json;import evaluation.configurations as cf;default_config = cf.EvalConfig();print(json.dumps(default_config.model_dump(), indent=2));'
