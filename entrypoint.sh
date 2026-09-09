#!/bin/bash
set -euo pipefail
python scripts/fetch_space_index.py
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
