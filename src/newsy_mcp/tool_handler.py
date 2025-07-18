import base64
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

from pdf_chef.config import Config
from pdf_chef.pdf_processor import PDFProcessor
from pdf_chef.tools import PDF_CHEF_TOOL_DEFINITIONS
from pdf_chef.vector_store import VectorStore

# Initialize console for rich output, redirected to stderr
console = Console(file=sys.stderr)

# Debug output control flags
SHOW_RAW_FUNCTION_CALLS = False
SHOW_TOOL_CALLS = True


class PDFChefToolHandler:
    """Handles PDF Chef tool calls from the LLM."""

    def __init__(self):
        """Initialize the PDF Chef tool handler."""
        # Load configuration
        self.config = Config.get_config()

        # Initialize components
        self.vector_store = VectorStore(
            db_directory=self.config["chroma_db_directory"],
            embedding_model=self.config["embedding_model"],
            openai_api_key=self.config.get("openai_api_key"),
        )
        self.pdf_processor = PDFProcessor()

        # Tool definitions
        self.tool_definitions = PDF_CHEF_TOOL_DEFINITIONS

    def _get_filename_from_id(self, document_id: int) -> Optional[str]:
        """
        Get the filename corresponding to a document ID.

        Args:
            document_id: The document ID to look up

        Returns:
            The filename if found, None otherwise
        """
        try:
            # First check if we can find the document ID in the metadata
            results = self.vector_store.collection.get(
                where={"document_id": document_id}, include=["metadatas"]
            )

            if results and "metadatas" in results and results["metadatas"]:
                # Return the filename from the first matching document
                return results["metadatas"][0].get("file")

            # If not found in main collection, try the pages collection
            results = self.vector_store.pages_collection.get(
                where={"document_id": document_id}, include=["metadatas"]
            )

            if results and "metadatas" in results and results["metadatas"]:
                # Return the filename from the first matching document
                return results["metadatas"][0].get("file")

            # If not found using where clause, try checking all documents
            all_docs = self._handle_list_ingested_pdfs(None)
            for doc in all_docs:
                if doc.get("id") == document_id:
                    return doc.get("filename")

            return None
        except Exception as e:
            console.print(f"Error getting filename from ID: {str(e)}")
            return None

    def _generate_document_id(self, filename: str) -> int:
        """
        Generate a simple incrementing document ID.

        Args:
            filename: The PDF filename (unused, kept for compatibility)

        Returns:
            An integer document ID
        """
        # Delegate to vector store for consistent ID generation
        return self.vector_store.generate_document_id(filename)

    def _save_base64_to_temp_file(self, base64_str: str, extension: str = "png") -> str:
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

    def _handle_get_config(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle get_config tool call.

        Args:
            args: Tool call arguments

        Returns:
            Configuration data
        """
        console.print("[bold blue]======== GET CONFIG ========[/]")
        try:
            return Config.get_config()
        except Exception as e:
            return {"error": f"Error getting configuration: {str(e)}"}

    def _handle_list_ingested_pdfs(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Handle list_ingested_pdfs tool call.

        Args:
            args: Tool call arguments

        Returns:
            List of ingested PDFs
        """
        console.print("[bold blue]======== LIST INGESTED PDFS ========[/]")
        try:
            # Query collection to get all unique sources
            results = self.vector_store.collection.get(include=["metadatas"])

            if not results or "metadatas" not in results or not results["metadatas"]:
                return []

            # Extract unique filenames from metadata and generate IDs
            files_dict = {}
            for metadata in results["metadatas"]:
                if metadata and "file" in metadata:
                    filename = metadata["file"]
                    if filename not in files_dict:
                        # Generate a short ID from the filename
                        doc_id = self._generate_document_id(filename)
                        files_dict[filename] = {"id": doc_id, "filename": filename}

            # Return as a list of dictionaries
            return sorted([info for info in files_dict.values()], key=lambda x: x["filename"])
        except Exception as e:
            console.print(f"Error listing PDFs: {str(e)}")
            return []

    def _handle_ingest_pdf(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle ingest_pdf tool call.

        Args:
            args: Tool call arguments

        Returns:
            Result of the ingestion process
        """
        pdf_path = args.get("pdf_path", "")
        extract_images = args.get("extract_images", False)
        images_dir = args.get("images_dir")

        console.print(
            f"[bold blue]======== INGEST PDF ========[/]\nPath: {pdf_path}\nExtract images: {extract_images}"
        )

        try:
            # Validate inputs
            if not os.path.exists(pdf_path):
                return {"error": f"PDF file not found: {pdf_path}"}

            if extract_images and not images_dir:
                return {"error": "images_dir must be provided when extract_images is True"}

            # Process PDF
            documents = self.pdf_processor.load_pdf(pdf_path)

            if not documents:
                return {"error": "No content could be extracted from the PDF"}

            # Store in vector database
            self.vector_store.add_documents(documents)

            # Extract images if requested
            image_info = None
            if extract_images:
                image_info = self.pdf_processor.extract_images(pdf_path, images_dir)

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
            doc_id = self._generate_document_id(filename)

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

    def _handle_similarity_search(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Handle similarity_search tool call.

        Args:
            args: Tool call arguments

        Returns:
            Search results
        """
        query = args.get("query", "")
        k = args.get("k", 10)

        try:
            results = self.vector_store.similarity_search(query, k)

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
                        result["document_id"] = self._generate_document_id(metadata["file"])

            return results
        except Exception as e:
            console.print(f"Error in similarity search: {str(e)}")
            return []

    def _handle_get_pdf_metadata(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle get_pdf_metadata tool call.

        Args:
            args: Tool call arguments

        Returns:
            PDF metadata
        """
        document_id = args.get("document_id", "")

        # Convert to int if it's a string
        if isinstance(document_id, str) and document_id.isdigit():
            document_id = int(document_id)

        console.print(
            f"[bold blue]======== GET PDF METADATA ========[/]\nDocument ID: {document_id}"
        )

        try:
            # First get the filename from document ID
            filename = self._get_filename_from_id(document_id)
            if not filename:
                return {"error": f"Document ID not found: {document_id}"}

            # Query pages collection to get all unique pages
            page_results = self.vector_store.pages_collection.get(
                where={"document_id": document_id}, include=["metadatas"]
            )

            # Query main collection for chunks
            chunk_results = self.vector_store.collection.get(
                where={"document_id": document_id}, include=["metadatas"]
            )

            if (
                not page_results or "metadatas" not in page_results or not page_results["metadatas"]
            ) and (
                not chunk_results
                or "metadatas" not in chunk_results
                or not chunk_results["metadatas"]
            ):
                return {"error": f"Document not found: {document_id}"}

            # Count chunks
            total_chunks = (
                len(chunk_results["metadatas"])
                if chunk_results and "metadatas" in chunk_results
                else 0
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

    def _handle_get_page_with_image(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle get_page_with_image tool call.

        Args:
            args: Tool call arguments

        Returns:
            Page data with image
        """
        document_id = args.get("document_id", "")
        page_number = args.get("page_number", 1)

        # Convert to int if it's a string
        if isinstance(document_id, str) and document_id.isdigit():
            document_id = int(document_id)

        console.print(
            f"[bold blue]======== GET PAGE WITH IMAGE ========[/]\nDocument ID: {document_id}, Page: {page_number}"
        )

        try:
            # First get the filename from document ID
            filename = self._get_filename_from_id(document_id)
            if not filename:
                error_msg = f"[bold red]ERROR:[/] Document ID {document_id} not found"
                console.print(error_msg)
                return {"error": f"Document ID not found: {document_id}"}

            console.print(f"[bold green]Found document:[/] {filename}")

            # First try to get from vector store (for already ingested PDFs)
            page_data = self.vector_store.get_full_page(document_id, page_number)

            if page_data:
                # Get tables data if available
                tables = []
                if (
                    "metadata" in page_data
                    and "tables" in page_data["metadata"]
                    and page_data["metadata"]["tables"]
                ):
                    tables = json.loads(page_data["metadata"]["tables"])

                # Create a result with image data included as base64
                result = {
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": page_number,
                    "text": page_data["content"],
                    "tables": tables,
                    "success": True,
                }

                # Include image if available
                if "metadata" in page_data and "image" in page_data["metadata"]:
                    result["image"] = page_data["metadata"]["image"]
                    console.print("[bold green]✓ Image data received in response[/]")
                else:
                    console.print("[bold yellow]WARNING:[/] No image available for this page")
                    result["error"] = "No image available"

                return result
            else:
                console.print(
                    "[bold yellow]Page data not found in vector store, trying original PDF...[/]"
                )

            # If not found in vector store, try to get from original PDF
            # Find the PDF path from metadata
            results = self.vector_store.collection.get(
                where={"document_id": document_id}, include=["metadatas"], limit=1
            )

            if not results or "metadatas" not in results or not results["metadatas"]:
                error_msg = f"[bold red]ERROR:[/] No metadata found for document ID {document_id}"
                console.print(error_msg)
                return {"error": f"Document not found: {document_id}"}

            pdf_path = results["metadatas"][0].get("source")

            if not pdf_path or not os.path.exists(pdf_path):
                error_msg = f"[bold red]ERROR:[/] Original PDF file not found at {pdf_path}"
                console.print(error_msg)
                return {"error": "Original PDF file not found"}

            console.print(f"[bold green]Trying to load from original PDF:[/] {pdf_path}")

            # Get page data directly from PDF
            page_data = self.pdf_processor.get_page_data(pdf_path, page_number)

            if "error" in page_data:
                error_msg = f"[bold red]ERROR:[/] {page_data['error']}"
                console.print(error_msg)
                return page_data

            console.print(f"[bold green]✓ Successfully retrieved page {page_number} from PDF[/]")
            return {
                "document_id": document_id,
                "filename": filename,
                "page_number": page_number,
                "text": page_data["text"],
                "tables": page_data.get("tables", []),
                "image": page_data["image"],
                "success": True,
            }
        except Exception as e:
            error_msg = f"[bold red]ERROR getting page with image:[/] {str(e)}"
            console.print(error_msg)
            import traceback

            console.print(traceback.format_exc())
            return {"error": f"Error getting page with image: {str(e)}"}

    def handle_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single tool call.

        Args:
            tool_call: Tool call data

        Returns:
            Tool call result
        """
        function = tool_call["function"]
        name = function["name"]
        args = json.loads(function["arguments"]) if function.get("arguments") else {}

        # Debug: Print the raw function arguments
        if SHOW_RAW_FUNCTION_CALLS:
            console.print(
                Panel(
                    str(function).strip(),
                    title="[bold magenta]Raw Function Call[/bold magenta]",
                    border_style="magenta",
                )
            )

        # Route to the appropriate handler based on function name
        if name == "get_config":
            result = self._handle_get_config(args)
        elif name == "list_ingested_pdfs":
            result = self._handle_list_ingested_pdfs(args)
        elif name == "ingest_pdf":
            result = self._handle_ingest_pdf(args)
        elif name == "similarity_search":
            result = self._handle_similarity_search(args)
        elif name == "get_pdf_metadata":
            result = self._handle_get_pdf_metadata(args)
        elif name == "get_page_with_image":
            result = self._handle_get_page_with_image(args)
        elif name == "chat_with_pdf":
            result = self._handle_chat_with_pdf(args)
        else:
            error_message = f"Unknown command: {name}"
            result = {"error": error_message}

        return {
            "tool_call_id": tool_call["id"],
            "function": {"name": name},
            "output": json.dumps(result),
        }

    def handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle multiple tool calls from the LLM.

        Args:
            tool_calls: List of tool calls from the LLM

        Returns:
            List of tool results
        """
        tool_results = []

        for tool_call in tool_calls:
            if SHOW_TOOL_CALLS:
                console.print(
                    Panel(
                        f"Tool: {tool_call['function']['name']}",
                        title="[bold blue]Tool Call[/bold blue]",
                        border_style="blue",
                    )
                )

            result = self.handle_tool_call(tool_call)
            tool_results.append(result)

        return tool_results
