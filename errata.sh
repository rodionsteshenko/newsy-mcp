#! /bin/zsh
clear
ruff format . && ruff check --fix .
rm -rf .chromadb
uv run src/pdf_chef/cli.py ingest ~/Desktop/errata_1.pdf
# uv run src/pdf_chef/cli.py ingest ~/Desktop/errata_2.pdf
# uv run src/pdf_chef/cli.py ingest ~/Desktop/errata_3.pdf
