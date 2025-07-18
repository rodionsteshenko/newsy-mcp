import base64
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP, Image
from rich.console import Console

from pdf_chef.config import Config
from pdf_chef.pdf_processor import PDFProcessor
from pdf_chef.vector_store import VectorStore

# Initialize console for rich output, redirected to stderr
console = Console(file=sys.stderr)

# Load configuration
config = Config.get_config()

# Initialize MCP server
mcp = FastMCP("pdf-chef")

# Initialize components
vector_store = VectorStore(
    db_directory=config["chroma_db_directory"],
    embedding_model=config["embedding_model"],
    openai_api_key=config.get("openai_api_key"),
)
pdf_processor = PDFProcessor()


@mcp.tool()
def get_config() -> Dict[str, Any]:
    """
    Get the current configuration settings.

    Returns:
        Dict[str, Any]: The current configuration settings
    """
    try:
        return Config.get_config()
    except Exception as e:
        return {"error": f"Error getting configuration: {str(e)}"}


@mcp.tool()
def list_ingested_pdfs() -> List[Dict[str, Any]]:
    """
    List all ingested PDF documents in the vector store.

    Returns:
        List[Dict[str, Any]]: List of PDF documents with their IDs and filenames
    """
    try:
        # Query collection to get all unique sources
        results = vector_store.collection.get(include=["metadatas"])

        if not results or "metadatas" not in results or not results["metadatas"]:
            return []

        # Extract unique filenames from metadata and generate IDs
        files_dict = {}
        for metadata in results["metadatas"]:
            if metadata and "file" in metadata:
                filename = metadata["file"]
                if filename not in files_dict:
                    # Generate an ID from the filename
                    doc_id = _generate_document_id(filename)
                    files_dict[filename] = {"id": doc_id, "filename": filename}

        # Return as a list of dictionaries
        return sorted([info for info in files_dict.values()], key=lambda x: x["filename"])
    except Exception as e:
        console.print(f"Error listing PDFs: {str(e)}")
        return []


def _generate_document_id(filename: str) -> int:
    """
    Generate a simple incrementing document ID.

    Args:
        filename: The PDF filename (unused, kept for compatibility)

    Returns:
        An integer document ID
    """
    # Use the vector store's ID generation for consistency
    return int(vector_store.generate_document_id(filename))


@mcp.tool()
def ingest_pdf(
    pdf_path: str, extract_images: bool = False, images_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ingest a PDF document into the vector database.

    Args:
        pdf_path (str): Path to the PDF file
        extract_images (bool, optional): Whether to extract images from the PDF. Defaults to False.
        images_dir (str, optional): Directory to save extracted images. Required if extract_images is True.

    Returns:
        Dict[str, Any]: Results of the ingestion process
    """
    try:
        # Validate inputs
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file not found: {pdf_path}"}

        if extract_images and not images_dir:
            return {"error": "images_dir must be provided when extract_images is True"}

        # Process PDF
        documents = pdf_processor.load_pdf(pdf_path)

        if not documents:
            return {"error": "No content could be extracted from the PDF"}

        # Store in vector database
        vector_store.add_documents(documents)

        # Extract images if requested
        image_info = None
        if extract_images:
            image_info = pdf_processor.extract_images(pdf_path, images_dir)

        # Count searchable chunks vs. page data
        searchable_count = sum(
            1
            for doc in documents
            if hasattr(doc, "metadata") and doc.metadata.get("searchable", False)
        )
        page_count = sum(
            1
            for doc in documents
            if hasattr(doc, "metadata") and doc.metadata.get("is_full_page", False)
        )

        # Get the filename and generate a document ID
        filename = os.path.basename(pdf_path)
        doc_id = _generate_document_id(filename)

        return {
            "success": True,
            "document_chunks": searchable_count,
            "page_count": page_count,
            "total_documents": len(documents),
            "pdf_path": pdf_path,
            "filename": filename,
            "id": doc_id,
            "images_extracted": len(image_info) if image_info else 0,
        }
    except Exception as e:
        return {"error": f"Error ingesting PDF: {str(e)}"}


@mcp.tool()
def similarity_search(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Search for documents similar to the query.

    Args:
        query (str): Search query
        k (int, optional): Number of results to return. Defaults to 10.

    Returns:
        List[Dict[str, Any]]: List of document chunks that match the query
    """
    try:
        results = vector_store.similarity_search(query, k)

        # Format results with page information and document ID
        for result in results:
            if "metadata" in result:
                metadata = result["metadata"]
                # Include primary page if available
                if "primary_page" in metadata:
                    result["page"] = metadata["primary_page"]
                # Include all pages that this chunk spans
                if "pages_list" in metadata:
                    # Convert comma-separated string back to list of integers
                    pages_list = [int(p) for p in metadata["pages_list"].split(",") if p]
                    result["pages"] = pages_list
                # Add document ID
                if "file" in metadata:
                    result["document_id"] = _generate_document_id(metadata["file"])

        return results
    except Exception as e:
        console.print(f"Error in similarity search: {str(e)}")
        return []


@mcp.tool()
def get_pdf_metadata(document_id: int) -> Dict[str, Any]:
    """
    Get metadata about a specific ingested PDF.

    Args:
        document_id (int): ID of the PDF document

    Returns:
        Dict[str, Any]: Metadata about the PDF
    """
    try:
        # First, get the filename from the document ID
        filename = _get_filename_from_id(document_id)

        if not filename:
            return {"error": f"Document not found with ID: {document_id}"}

        # Query pages collection to get all unique pages
        page_results = vector_store.pages_collection.get(
            where={"document_id": document_id}, include=["metadatas"]
        )

        # Query main collection for chunks
        chunk_results = vector_store.collection.get(
            where={"document_id": document_id}, include=["metadatas"]
        )

        if (
            not page_results or "metadatas" not in page_results or not page_results["metadatas"]
        ) and (
            not chunk_results or "metadatas" not in chunk_results or not chunk_results["metadatas"]
        ):
            return {"error": f"PDF not found: {filename}"}

        # Count chunks
        total_chunks = (
            len(chunk_results["metadatas"]) if chunk_results and "metadatas" in chunk_results else 0
        )

        # Get unique pages from page collection
        pages = set()
        if page_results and "metadatas" in page_results:
            for metadata in page_results["metadatas"]:
                if "page" in metadata:
                    pages.add(metadata["page"])

        # If no pages found in page collection, try to get from chunks
        if not pages and chunk_results and "metadatas" in chunk_results:
            for metadata in chunk_results["metadatas"]:
                if "pages_list" in metadata:
                    # Convert comma-separated string back to list of integers
                    pages_list = [int(p) for p in metadata["pages_list"].split(",") if p]
                    pages.update(pages_list)

        return {
            "id": document_id,
            "filename": filename,
            "total_chunks": total_chunks,
            "total_pages": len(pages),
            "pages": sorted(list(pages)),
        }
    except Exception as e:
        return {"error": f"Error getting PDF metadata: {str(e)}"}


def _get_filename_from_id(document_id: int) -> Optional[str]:
    """
    Get the filename corresponding to a document ID.

    Args:
        document_id: The document ID to look up

    Returns:
        The filename if found, None otherwise
    """
    # List all PDFs and find the one with matching ID
    try:
        # First check if we can find the document ID in the metadata
        results = vector_store.collection.get(
            where={"document_id": document_id}, include=["metadatas"]
        )

        if results and "metadatas" in results and results["metadatas"]:
            # Return the filename from the first matching document
            return results["metadatas"][0].get("file")

        # If not found in main collection, try the pages collection
        results = vector_store.pages_collection.get(
            where={"document_id": document_id}, include=["metadatas"]
        )

        if results and "metadatas" in results and results["metadatas"]:
            # Return the filename from the first matching document
            return results["metadatas"][0].get("file")

        # If not found using where clause, try checking all documents
        # This is a fallback for older databases
        console.print(
            f"Document ID {document_id} not found in direct query, checking all documents..."
        )
        # Check all documents as a fallback (should not be needed after migration)
        all_docs = list_ingested_pdfs()
        for doc in all_docs:
            if doc.get("id") == document_id:
                return doc.get("filename")

        return None
    except Exception as e:
        console.print(f"Error getting filename from ID: {str(e)}")
        return None


def _save_base64_to_temp_file(base64_str: str, extension: str = "png") -> str:
    """
    Save a base64 string to a temporary file.

    Args:
        base64_str: Base64 encoded string
        extension: File extension

    Returns:
        Path to the temporary file
    """
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}")
    temp_path = temp_file.name

    # Decode and write to file
    img_data = base64.b64decode(base64_str)
    with open(temp_path, "wb") as f:
        f.write(img_data)

    return temp_path


@mcp.tool()
def get_page_with_image(document_id: int, page_number: int) -> Tuple[Dict[str, Any], Image]:
    """
    Get both text and image data for a specific page of a PDF.

    Args:
        document_id (int): ID of the PDF document
        page_number (int): Page number to retrieve

    Returns:
        Tuple[Dict[str, Any], Image]: Page data (text and tables) and the page image
    """
    try:
        console.print(f"[bold blue]Retrieving page {page_number} for document ID {document_id}[/]")

        # First, get the filename from the document ID
        filename = _get_filename_from_id(document_id)

        if not filename:
            error_msg = f"[bold red]ERROR:[/] Document ID {document_id} not found in database"
            console.print(error_msg)
            return (
                {"error": f"Document not found with ID: {document_id}"},
                Image(path=""),  # Empty image
            )

        console.print(f"[bold green]Found document:[/] {filename}")

        # First try to get from vector store (for already ingested PDFs)
        page_data = vector_store.get_full_page(document_id, page_number)

        if page_data:
            console.print(f"[bold green]Found page {page_number} in vector store[/]")

            # Extract image data and save to temp file
            if "metadata" in page_data and "image" in page_data["metadata"]:
                image_b64 = page_data["metadata"]["image"]
                temp_image_path = _save_base64_to_temp_file(image_b64)
                console.print("[bold green]✓ Image data available[/]")

                # Get tables data if available
                tables = []
                if (
                    "metadata" in page_data
                    and "tables" in page_data["metadata"]
                    and page_data["metadata"]["tables"]
                ):
                    tables = json.loads(page_data["metadata"]["tables"])
                    if tables:
                        console.print(f"[bold green]Found {len(tables)} tables in the page[/]")

                # Return text data and image
                return (
                    {
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page_number,
                        "text": page_data["content"],
                        "tables": tables,
                        "success": True,
                    },
                    Image(path=temp_image_path),
                )
            else:
                console.print("[bold yellow]WARNING: No image data available for this page[/]")

                # Get tables data if available
                tables = []
                if (
                    "metadata" in page_data
                    and "tables" in page_data["metadata"]
                    and page_data["metadata"]["tables"]
                ):
                    tables = json.loads(page_data["metadata"]["tables"])

                return (
                    {
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page_number,
                        "text": page_data["content"],
                        "tables": tables,
                        "success": True,
                        "error": "No image available",
                    },
                    Image(path=""),  # Empty image
                )
        else:
            console.print(
                f"[bold yellow]Page {page_number} not found in vector store, trying original PDF...[/]"
            )

        # If not found in vector store, try to get from original PDF
        # Find the PDF path from metadata
        results = vector_store.collection.get(
            where={"document_id": document_id}, include=["metadatas"], limit=1
        )

        if not results or "metadatas" not in results or not results["metadatas"]:
            error_msg = f"[bold red]ERROR:[/] No metadata found for document ID {document_id}"
            console.print(error_msg)
            return (
                {"error": f"PDF not found: {filename}"},
                Image(path=""),  # Empty image
            )

        pdf_path = results["metadatas"][0].get("source")

        if not pdf_path or not os.path.exists(pdf_path):
            error_msg = f"[bold red]ERROR:[/] Original PDF file not found at path: {pdf_path}"
            console.print(error_msg)
            return (
                {"error": "Original PDF file not found"},
                Image(path=""),  # Empty image
            )

        console.print(f"[bold blue]Loading directly from PDF file:[/] {pdf_path}")

        # Get page data directly from PDF
        page_data = pdf_processor.get_page_data(pdf_path, page_number)

        if "error" in page_data:
            error_msg = f"[bold red]ERROR:[/] {page_data['error']}"
            console.print(error_msg)
            return (
                page_data,
                Image(path=""),  # Empty image
            )

        console.print("[bold green]✓ Successfully retrieved page data from PDF file[/]")

        # Save image to temp file
        temp_image_path = _save_base64_to_temp_file(page_data["image"])
        console.print("[bold green]✓ Image saved to temporary file[/]")

        return (
            {
                "document_id": document_id,
                "filename": filename,
                "page_number": page_number,
                "text": page_data["text"],
                "tables": page_data.get("tables", []),
                "success": True,
            },
            Image(path=temp_image_path),
        )
    except Exception as e:
        error_msg = f"[bold red]ERROR getting page with image:[/] {str(e)}"
        console.print(error_msg)
        import traceback

        console.print(traceback.format_exc())
        return (
            {"error": f"Error getting page with image: {str(e)}"},
            Image(path=""),  # Empty image
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
