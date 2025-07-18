#!/bin/bash

# Demo script for PDF page validation tools
# Usage: ./demo-page-validation.sh <pdf_filename> <page_number>

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <pdf_filename> <page_number> [output_dir]"
    echo "Example: $0 sample.pdf 1 ./images"
    exit 1
fi

PDF_FILENAME="$1"
PAGE_NUMBER="$2"
OUTPUT_DIR="${3:-./page_images}"

echo "===================================="
echo "PDF Chef Page Validation Demo"
echo "===================================="
echo "PDF: $PDF_FILENAME"
echo "Page: $PAGE_NUMBER"
echo "Output directory: $OUTPUT_DIR"
echo "===================================="

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Step 1: Validate the PDF ingestion status
echo -e "\n\033[1m1. Validating PDF ingestion status\033[0m"
python -m pdf_chef validate-pdf "$PDF_FILENAME"

# Step 2: Get full page content with image
echo -e "\n\033[1m2. Retrieving full page content with image\033[0m"
python -m pdf_chef get-page "$PDF_FILENAME" "$PAGE_NUMBER" --save-image --output-dir "$OUTPUT_DIR"

# Step 3: Get page chunks
echo -e "\n\033[1m3. Retrieving page chunks\033[0m"
python -m pdf_chef get-page "$PDF_FILENAME" "$PAGE_NUMBER" --chunks

# Step 4: Search within the page with a query
echo -e "\n\033[1m4. Searching within the page\033[0m"
read -p "Enter a search term for page $PAGE_NUMBER: " SEARCH_QUERY
python -m pdf_chef get-page "$PDF_FILENAME" "$PAGE_NUMBER" --chunks --query "$SEARCH_QUERY"

echo -e "\n\033[1mDemo completed!\033[0m"
echo "If images were saved, they can be found in: $OUTPUT_DIR"
echo "====================================" 