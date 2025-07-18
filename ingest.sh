#! /bin/zsh
clear
ruff format . && ruff check --fix .
rm -rf .chromadb
uv run src/pdf_chef/cli.py ingest ~/Downloads/Errata.pdf
uv run src/pdf_chef/cli.py ingest ~/Downloads/Datasheet.pdf
uv run src/pdf_chef/cli.py ingest ~/Downloads/Reference_Manual.pdf
