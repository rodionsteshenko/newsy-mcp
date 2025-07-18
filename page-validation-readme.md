# PDF Chef Page Validation Tools

This set of tools helps diagnose and validate PDF page content retrieval in the PDF Chef application.

## Problem

When the application is unable to return page content or image data, these tools can help diagnose issues with:
- PDF ingestion process
- Vector store data storage
- Page content and image retrieval

## Tool Commands

### 1. Get Page Content

Retrieve and validate page content from an ingested PDF.

```bash
pdf-chef get-page <filename> <page_number> [options]
```

Options:
- `--save-image`: Save the page image to a file
- `--output-dir <dir>`: Directory to save the page image
- `--chunks`: Retrieve individual chunks for the page
- `--query <text>`: Optional search query to find relevant chunks within the page

Examples:
```bash
# Get full page content and verify if image data is available
pdf-chef get-page sample.pdf 1

# Get page content and save the image
pdf-chef get-page sample.pdf 1 --save-image --output-dir ./images

# Get all chunks associated with the page
pdf-chef get-page sample.pdf 1 --chunks

# Search for specific content within a page
pdf-chef get-page sample.pdf 1 --chunks --query "important topic"
```

### 2. Validate PDF Ingestion

Validate the ingestion status of a PDF in the database to check if all pages and content are properly stored.

```bash
pdf-chef validate-pdf <filename>
```

Example:
```bash
pdf-chef validate-pdf sample.pdf
```

This will output:
- Number of searchable chunks
- Pages referenced in chunks
- Full page entries available
- Pages with images available
- Missing pages
- Original PDF file status

## Demo Script

A demo script is provided to showcase all functions in sequence:

```bash
./demo-page-validation.sh <pdf_filename> <page_number> [output_dir]
```

Example:
```bash
./demo-page-validation.sh sample.pdf 1 ./images
```

The script will:
1. Validate PDF ingestion status
2. Retrieve full page content with image
3. Get all page chunks
4. Search within the page using a user-provided query

## Troubleshooting

If pages or images are not available, try re-ingesting the PDF:

```bash
pdf-chef ingest /path/to/your/document.pdf
```

For debugging the data storage, you can directly inspect the Chroma database in the configured location. 