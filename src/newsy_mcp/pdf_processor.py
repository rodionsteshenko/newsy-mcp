"""PDF processor module for loading and chunking PDFs."""

import base64
import json
import os
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from pdf_chef.table import extract_tables_from_pdf, tables_to_markdown

console = Console()


class PDFProcessor:
    """Process PDF files for embedding and storage."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the PDF processor.

        Args:
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between text chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def generate_document_id(self, filename: str) -> int:
        """Generate a simple incrementing document ID.

        Args:
            filename: The PDF filename (unused, kept for compatibility)

        Returns:
            An integer document ID
        """
        # To avoid circular imports, we'll use the same algorithm
        # This should be replaced with a dependency injection pattern in the future
        import os
        import random

        # Path to the ID tracker file
        db_directory = os.getenv("CHROMA_DB_DIRECTORY", "./.chromadb")
        id_tracker_path = os.path.join(db_directory, "document_id_tracker.txt")

        try:
            if os.path.exists(id_tracker_path):
                with open(id_tracker_path, "r") as f:
                    current_id = int(f.read().strip())
                next_id = current_id + 1
            else:
                next_id = 1
                os.makedirs(db_directory, exist_ok=True)

            # Save the next ID for future use
            with open(id_tracker_path, "w") as f:
                f.write(str(next_id))

            return next_id
        except Exception as e:
            print(f"Error managing document ID: {str(e)}")
            # Fallback to a random ID if there's an error
            return random.randint(1000, 9999)

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load a PDF file and split it into chunks using PyMuPDF.

        The function:
        1. Extracts text and image from each page
        2. Stores full page data for later retrieval
        3. Chunks the entire document text across page boundaries
        4. Maps each chunk back to the pages it spans

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of document chunks with metadata + page documents
        """
        if not os.path.exists(pdf_path):
            console.print(f"[bold red]Error:[/] File {pdf_path} not found")
            return []

        try:
            filename = os.path.basename(pdf_path)
            console.print(f"[bold green]Loading[/] {filename}")

            # Generate document ID for this file
            doc_id = self.generate_document_id(filename)

            # Extract text content, images, and tables
            page_data = self._extract_pdf_data(pdf_path)

            if not page_data:
                console.print(f"[bold yellow]Warning:[/] No content extracted from {pdf_path}")
                return []

            chunks = []

            # First create documents for each full page (not for embedding)
            for page in page_data:
                # Extract tables data for metadata
                tables_info = []
                if page.get("tables"):
                    for i, table in enumerate(page["tables"]):
                        table_md = (
                            page["tables_markdown"][i]
                            if i < len(page.get("tables_markdown", []))
                            else ""
                        )
                        tables_info.append(
                            {
                                "bbox": list(table.bbox),
                                "markdown": table_md,
                            }
                        )

                # Convert tables to JSON string for ChromaDB which doesn't support lists in metadata
                tables_json = json.dumps(tables_info) if tables_info else ""

                chunks.append(
                    Document(
                        page_content=page["text"],
                        metadata={
                            "source": pdf_path,
                            "file": filename,
                            "document_id": doc_id,
                            "page": page["page_num"],
                            "is_full_page": True,
                            "image": page["image"],
                            "tables": tables_json,  # Store as JSON string
                            "searchable": False,  # Mark as not for embedding
                        },
                    )
                )

            # Then create chunks from the entire document text
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                chunk_task = progress.add_task("Splitting text into chunks", total=1)

                # Get the combined text from the page data
                combined_text = ""
                if page_data and "combined_text" in page_data[-1]:
                    combined_text = page_data[-1]["combined_text"]
                else:
                    # Fallback: combine all page texts
                    combined_text = "\n\n".join(page["text"] for page in page_data)

                # Split the combined text into chunks
                text_chunks = self.text_splitter.split_text(combined_text)
                progress.update(chunk_task, advance=1)

            # Create a progress bar for chunk processing
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                chunk_task = progress.add_task("Processing chunks", total=len(text_chunks))

                # For each text chunk, determine which pages it spans
                for chunk_idx, chunk_text in enumerate(text_chunks):
                    if not chunk_text.strip():
                        progress.update(chunk_task, advance=1)
                        continue

                    # Find start position of this chunk in the original text
                    chunk_start = combined_text.find(chunk_text)
                    if chunk_start == -1:  # Handle edge case where find fails
                        progress.update(chunk_task, advance=1)
                        continue
                    chunk_end = chunk_start + len(chunk_text)

                    # Find which pages this chunk spans
                    spanning_pages = []
                    for page in page_data:
                        # Check if chunk overlaps with this page
                        if chunk_start < page["end_char"] and chunk_end > page["start_char"]:
                            spanning_pages.append(page["page_num"])

                    # Create chunk document with page mapping
                    if spanning_pages:
                        # Store pages as a string representation instead of a list
                        spanning_pages_str = ",".join(str(p) for p in spanning_pages)
                        primary_page = spanning_pages[0] if spanning_pages else None

                        chunks.append(
                            Document(
                                page_content=chunk_text,
                                metadata={
                                    "source": pdf_path,
                                    "file": filename,
                                    "document_id": doc_id,
                                    "pages_list": spanning_pages_str,  # Store as string for ChromaDB
                                    "primary_page": primary_page,
                                    "is_full_page": False,
                                    "searchable": True,  # Mark as embeddable
                                },
                            )
                        )

                    progress.update(chunk_task, advance=1)

            console.print(
                f"Created [bold]{len(chunks)}[/] documents ({len(text_chunks)} searchable chunks)"
            )
            return chunks

        except Exception as e:
            console.print(f"[bold red]Error processing PDF:[/] {str(e)}")
            return []

    def extract_images(
        self, pdf_path: str, output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract images from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images (if None, images are not saved)

        Returns:
            List of dictionaries with image metadata
        """
        if not os.path.exists(pdf_path):
            console.print(f"[bold red]Error:[/] File {pdf_path} not found")
            return []

        try:
            filename = os.path.basename(pdf_path)
            console.print(f"[bold green]Extracting images from[/] {filename}")

            # Create output directory if specified
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Open the PDF with PyMuPDF
            pdf_document = fitz.open(pdf_path)

            # Extract document information
            total_pages = len(pdf_document)
            console.print(f"Document has [bold]{total_pages}[/] pages")

            # Process and extract images from each page
            image_list = []

            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                extract_task = progress.add_task("Extracting images from PDF", total=total_pages)

                for page_num, page in enumerate(pdf_document):
                    # Get images
                    image_dict = page.get_images(full=True)

                    for img_idx, img_info in enumerate(image_dict):
                        xref = img_info[0]  # Image reference number

                        # Extract the image
                        base_img = pdf_document.extract_image(xref)

                        if base_img:
                            image_bytes = base_img["image"]
                            image_ext = base_img["ext"]

                            # Image metadata
                            image_meta = {
                                "page_num": page_num + 1,
                                "img_idx": img_idx + 1,
                                "width": base_img["width"],
                                "height": base_img["height"],
                                "format": image_ext,
                                "colorspace": base_img["colorspace"],
                                "xref": xref,
                            }

                            # Save image if output directory is specified
                            if output_dir:
                                img_filename = f"page{page_num+1}_img{img_idx+1}.{image_ext}"
                                img_path = os.path.join(output_dir, img_filename)

                                with open(img_path, "wb") as img_file:
                                    img_file.write(image_bytes)

                                image_meta["saved_path"] = img_path

                            image_list.append(image_meta)

                    progress.update(extract_task, advance=1)

            total_images = len(image_list)
            console.print(f"Extracted [bold]{total_images}[/] images from PDF")

            if output_dir and total_images > 0:
                console.print(f"Images saved to [bold]{output_dir}[/]")

            return image_list

        except Exception as e:
            console.print(f"[bold red]Error extracting images:[/] {str(e)}")
            return []

    def _extract_pdf_data(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text, images, and tables from PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries with page data
        """
        try:
            # Open the PDF with PyMuPDF
            pdf_document = fitz.open(pdf_path)

            # Extract document information
            total_pages = len(pdf_document)
            console.print(f"Document has [bold]{total_pages}[/] pages")

            # Extract tables from the PDF
            console.print("Extracting tables from PDF")
            tables_by_page = extract_tables_from_pdf(pdf_path)

            # Convert tables to markdown format with progress
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as table_md_progress:
                table_md_task = table_md_progress.add_task("Converting tables to markdown", total=1)
                tables_markdown_by_page = tables_to_markdown(tables_by_page)
                table_md_progress.update(table_md_task, advance=1)

            # Process and extract text from each page
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                extract_task = progress.add_task("Extracting text from PDF", total=total_pages)

                # Store text and position data for each page
                page_data = []
                full_text = []  # Full document text
                char_offset = 0  # Track character offset for mapping chunks to pages

                # Extract text from each page and track character positions
                for page_num, page in enumerate(pdf_document):
                    # Get text with PyMuPDF
                    text = page.get_text()

                    if not text.strip():  # Skip empty pages
                        progress.update(extract_task, advance=1)
                        continue

                    # Store page image as base64
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    b64_img = base64.b64encode(img_data).decode("utf-8")

                    # Get tables for this page
                    page_tables = []
                    page_tables_markdown = []
                    if page_num in tables_by_page:
                        page_tables = tables_by_page[page_num]
                    if page_num in tables_markdown_by_page:
                        page_tables_markdown = tables_markdown_by_page[page_num]

                    # Add to full text
                    full_text.append(text)

                    # Store page data with character offsets
                    page_data.append(
                        {
                            "page_num": page_num + 1,
                            "text": text,
                            "image": b64_img,
                            "start_char": char_offset,
                            "end_char": char_offset + len(text),
                            "tables": page_tables,
                            "tables_markdown": page_tables_markdown,
                        }
                    )

                    # Update character offset for next page
                    char_offset += len(text)

                    progress.update(extract_task, advance=1)

                # Combine all text into one string
                combined_text = "\n\n".join(full_text)

                # Add combined text to the last element for processing
                if page_data:
                    page_data[-1]["combined_text"] = combined_text

                return page_data

        except Exception as e:
            console.print(f"[bold red]Error extracting PDF data:[/] {str(e)}")
            return []

    def get_page_data(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """Get both text and image data for a specific page.

        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (1-indexed)

        Returns:
            Dictionary with text and image data
        """
        if not os.path.exists(pdf_path):
            error_msg = f"[bold red]Error:[/] File {pdf_path} not found"
            console.print(error_msg)
            return {"error": "File not found"}

        try:
            # Get the base filename and generate document ID
            filename = os.path.basename(pdf_path)
            doc_id = self.generate_document_id(filename)

            # Open the PDF
            console.print(f"[bold blue]Opening PDF file:[/] {pdf_path}")
            pdf_document = fitz.open(pdf_path)

            # Validate page number
            if page_num < 1 or page_num > len(pdf_document):
                error_msg = f"[bold red]Error:[/] Invalid page number: {page_num}. PDF has {len(pdf_document)} pages."
                console.print(error_msg)
                return {
                    "error": f"Invalid page number: {page_num}. PDF has {len(pdf_document)} pages."
                }

            console.print(
                f"[bold green]Getting page {page_num} from PDF with {len(pdf_document)} pages[/]"
            )

            # Get the page (0-indexed in PyMuPDF)
            page = pdf_document[page_num - 1]

            # Get page text
            text = page.get_text()
            if not text.strip():
                console.print(f"[bold yellow]Warning: Page {page_num} has no text content[/]")

            # Get page image
            console.print(f"[bold blue]Generating image for page {page_num}[/]")
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            b64_img = base64.b64encode(img_data).decode("utf-8")

            # Extract tables from this page
            console.print(f"[bold blue]Extracting tables from page {page_num}[/]")
            tables_by_page = extract_tables_from_pdf(pdf_path, page_nums=[page_num - 1])
            tables_markdown_by_page = tables_to_markdown(tables_by_page)

            # Prepare tables information
            tables_info = []
            if page_num - 1 in tables_by_page:
                tables = tables_by_page[page_num - 1]
                tables_markdown = tables_markdown_by_page.get(page_num - 1, [])

                for i, table in enumerate(tables):
                    table_md = tables_markdown[i] if i < len(tables_markdown) else ""
                    tables_info.append(
                        {
                            "bbox": list(table.bbox),
                            "markdown": table_md,
                        }
                    )
                console.print(f"[bold green]Found {len(tables_info)} tables on page {page_num}[/]")

            console.print(f"[bold green]✓ Successfully retrieved page {page_num} data[/]")
            return {
                "page_num": page_num,
                "text": text,
                "image": b64_img,
                "total_pages": len(pdf_document),
                "tables": tables_info,
                "document_id": doc_id,
                "filename": filename,
            }

        except Exception as e:
            error_msg = f"[bold red]Error getting page data:[/] {str(e)}"
            console.print(error_msg)
            import traceback

            console.print(traceback.format_exc())
            return {"error": str(e)}
