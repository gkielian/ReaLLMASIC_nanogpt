#!/bin/bash


python3 get_gemma7b_tokens.py
python3 create_json.py -i output.txt -o output.json
python3 analyze_cjk_distribution.py output.json

