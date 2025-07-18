"""Vector database module for managing document embeddings."""

import concurrent.futures
import os
import sys
from functools import partial
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.api.types import EmbeddingFunction
from langchain_openai import OpenAIEmbeddings
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console(file=sys.stderr)


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """Adapter to use OpenAI embeddings with ChromaDB."""

    def __init__(self, openai_embeddings: OpenAIEmbeddings):
        """Initialize with an OpenAI embeddings instance.

        Args:
            openai_embeddings: OpenAI embeddings instance
        """
        self.openai_embeddings = openai_embeddings

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for texts.

        Args:
            input: List of texts to embed

        Returns:
            List of embeddings
        """
        return self.openai_embeddings.embed_documents(input)


class VectorStore:
    """Vector database for document embeddings and retrieval."""

    def __init__(
        self,
        db_directory: str,
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: str = None,
        max_parallel_threads: int = 10,
    ):
        """Initialize the vector store.

        Args:
            db_directory: Directory to store the database
            embedding_model: OpenAI embedding model to use
            openai_api_key: OpenAI API key (required)
            max_parallel_threads: Maximum number of parallel threads for embeddings (default: 10)
        """
        self.db_directory = db_directory
        self.max_parallel_threads = max_parallel_threads
        os.makedirs(db_directory, exist_ok=True)

        # Initialize the id tracker file path
        self.id_tracker_path = os.path.join(db_directory, "document_id_tracker.txt")

        # Require OpenAI API key
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for embeddings")

        try:
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
            self.embedding_function = OpenAIEmbeddingFunction(self.embeddings)
        except Exception as e:
            console.print(f"[bold red]Error initializing OpenAI embeddings:[/] {str(e)}")
            raise

        # Initialize ChromaDB directly
        self.client = chromadb.PersistentClient(path=db_directory)

        # Try to get existing collection or create a new one
        try:
            # Collection for searchable content
            self.collection = self.client.get_or_create_collection(
                name="documents", embedding_function=self.embedding_function
            )

            # Collection for page data (without embeddings)
            self.pages_collection = self.client.get_or_create_collection(name="pages")
        except Exception as e:
            console.print(f"[bold red]Error initializing ChromaDB:[/] {str(e)}")
            raise

    def generate_document_id(self, filename: str) -> int:
        """Generate a document ID for a filename and store the mapping.

        Args:
            filename: The PDF filename

        Returns:
            An integer document ID
        """
        # Check if we already have a document ID for this filename
        try:
            existing_results = self.collection.get(
                where={"file": filename},
                include=["metadatas"],
            )

            # If the document already exists, use its existing ID
            if existing_results and existing_results["metadatas"]:
                for metadata in existing_results["metadatas"]:
                    if "document_id" in metadata:
                        console.print(f"[bold blue]Using existing document ID for:[/] {filename}")
                        return metadata["document_id"]

            # If not found in main collection, check pages collection
            existing_results = self.pages_collection.get(
                where={"file": filename},
                include=["metadatas"],
            )

            if existing_results and existing_results["metadatas"]:
                for metadata in existing_results["metadatas"]:
                    if "document_id" in metadata:
                        console.print(
                            f"[bold blue]Using existing document ID from pages collection for:[/] {filename}"
                        )
                        return metadata["document_id"]
        except Exception as e:
            console.print(f"[bold yellow]Warning checking for existing document ID:[/] {str(e)}")

        # If no existing ID is found, create a new one
        next_id = self._get_next_document_id()
        console.print(f"[bold blue]Created new document ID {next_id} for:[/] {filename}")
        return next_id

    def _get_next_document_id(self) -> int:
        """Get the next available document ID.

        Returns:
            An integer representing the next available ID
        """
        try:
            if os.path.exists(self.id_tracker_path):
                with open(self.id_tracker_path, "r") as f:
                    current_id = int(f.read().strip())
                next_id = current_id + 1
            else:
                next_id = 1

            # Save the next ID for future use
            with open(self.id_tracker_path, "w") as f:
                f.write(str(next_id))

            return next_id
        except Exception as e:
            console.print(f"[bold red]Error managing document ID:[/] {str(e)}")
            raise ValueError(
                "Unable to generate a consistent document ID. Please check file permissions."
            )

    def add_documents(
        self, documents: List[Dict[Any, Any]], collection_name: Optional[str] = None
    ) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of document chunks with text and metadata
            collection_name: Optional name for document collection
        """
        if not documents:
            console.print("[bold yellow]Warning:[/] No documents to add")
            return

        # Separate searchable chunks from page data
        searchable_docs = []
        page_docs = []
        document_metadata = {}

        for doc in documents:
            if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                # LangChain Document type
                metadata = doc.metadata
                if metadata.get("searchable", True):
                    searchable_docs.append(doc)
                elif metadata.get("is_full_page", False):
                    page_docs.append(doc)

                # Collect document metadata for validation
                filename = metadata.get("file", "unknown")
                if filename not in document_metadata:
                    document_metadata[filename] = {
                        "total_pages": metadata.get("total_pages", 0),
                        "pages": set(),
                    }
                if "page" in metadata:
                    document_metadata[filename]["pages"].add(metadata["page"])

            elif isinstance(doc, dict) and "content" in doc and "metadata" in doc:
                # Dictionary format
                metadata = doc["metadata"]
                if metadata.get("searchable", True):
                    searchable_docs.append(doc)
                elif metadata.get("is_full_page", False):
                    page_docs.append(doc)

                # Collect document metadata for validation
                filename = metadata.get("file", "unknown")
                if filename not in document_metadata:
                    document_metadata[filename] = {
                        "total_pages": metadata.get("total_pages", 0),
                        "pages": set(),
                    }
                if "page" in metadata:
                    document_metadata[filename]["pages"].add(metadata["page"])

        # First add searchable documents to the main collection
        if searchable_docs:
            self._add_docs_to_collection(
                searchable_docs, self.collection, "Creating embeddings for searchable chunks"
            )

        # Then add page documents to the pages collection (without embeddings)
        if page_docs:
            self._add_docs_to_collection(page_docs, self.pages_collection, "Storing page documents")

        # Validate ingestion for each document
        for filename, metadata in document_metadata.items():
            self._validate_document_ingestion(filename, metadata["total_pages"], metadata["pages"])

    def _validate_document_ingestion(
        self, filename: str, total_pages: int, ingested_pages: set
    ) -> None:
        """Validate that all pages of a document were properly ingested.

        Args:
            filename: Name of the PDF file
            total_pages: Total number of pages in the document
            ingested_pages: Set of page numbers that were ingested
        """
        # If total_pages wasn't provided in metadata, try to determine it from the collections
        if not total_pages:
            console.print(
                f"[bold yellow]Note:[/] total_pages not provided in metadata for {filename}, attempting to determine from database"
            )

            # Try to find it in the metadata of any document in either collection
            try:
                # First check pages collection as it's more likely to have complete metadata
                results = self.pages_collection.get(
                    where={"file": filename},
                    include=["metadatas"],
                )

                if results and results["metadatas"]:
                    for metadata in results["metadatas"]:
                        if "total_pages" in metadata and metadata["total_pages"]:
                            total_pages = metadata["total_pages"]
                            console.print(
                                f"[bold green]Found total_pages = {total_pages} in pages collection[/]"
                            )
                            break

                # If not found in pages collection, check main collection
                if not total_pages:
                    results = self.collection.get(
                        where={"file": filename},
                        include=["metadatas"],
                    )

                    if results and results["metadatas"]:
                        for metadata in results["metadatas"]:
                            if "total_pages" in metadata and metadata["total_pages"]:
                                total_pages = metadata["total_pages"]
                                console.print(
                                    f"[bold green]Found total_pages = {total_pages} in main collection[/]"
                                )
                                break

                # If still not found, estimate based on highest page number found
                if not total_pages:
                    # Try to estimate based on the highest page number found
                    highest_page = self._get_highest_page_for_document(filename)
                    if highest_page > 0:
                        total_pages = highest_page
                        console.print(
                            f"[bold yellow]Estimated total_pages = {total_pages} based on highest page number found[/]"
                        )
            except Exception as e:
                console.print(f"[bold yellow]Warning:[/] Error determining total_pages: {str(e)}")

        if not total_pages:
            console.print(
                f"[bold red]Validation Warning:[/] Cannot validate ingestion for {filename}: unable to determine total_pages"
            )
            return

        # Validate pages in main collection
        main_collection_pages = set()
        try:
            results = self.collection.get(
                where={"file": filename},
                include=["metadatas"],
            )

            if results and results["metadatas"]:
                for metadata in results["metadatas"]:
                    # Check different ways pages might be stored
                    if "page" in metadata:
                        main_collection_pages.add(metadata["page"])
                    elif "primary_page" in metadata:
                        main_collection_pages.add(metadata["primary_page"])
                    elif "pages_list" in metadata:
                        pages_list = [int(p) for p in metadata["pages_list"].split(",") if p]
                        main_collection_pages.update(pages_list)
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/] Error validating main collection: {str(e)}")

        # Validate pages in pages collection
        pages_collection_pages = set()
        try:
            results = self.pages_collection.get(
                where={"file": filename},
                include=["metadatas"],
            )

            if results and results["metadatas"]:
                for metadata in results["metadatas"]:
                    if "page" in metadata:
                        pages_collection_pages.add(metadata["page"])
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/] Error validating pages collection: {str(e)}")

        # Combine all found pages
        all_found_pages = main_collection_pages.union(pages_collection_pages)

        # Check if all pages were ingested
        expected_pages = set(range(1, total_pages + 1))
        missing_pages = expected_pages - all_found_pages

        if missing_pages:
            console.print(
                f"[bold red]Validation Warning:[/] Document {filename} is missing {len(missing_pages)} of {total_pages} pages in the database"
            )
            if len(missing_pages) <= 20:
                console.print(f"[bold yellow]Missing pages:[/] {sorted(list(missing_pages))}")
            else:
                console.print(
                    f"[bold yellow]First 20 missing pages:[/] {sorted(list(missing_pages))[:20]}..."
                )
        else:
            console.print(
                f"[bold green]Validation Success:[/] All {total_pages} pages of {filename} were successfully ingested"
            )

    def _get_highest_page_for_document(self, filename: str) -> int:
        """Determine the highest page number for a document in the database.

        Args:
            filename: Name of the PDF file

        Returns:
            Highest page number found
        """
        highest_page = 0

        # Check main collection
        try:
            results = self.collection.get(
                where={"file": filename},
                include=["metadatas"],
            )

            if results and results["metadatas"]:
                for metadata in results["metadatas"]:
                    if "page" in metadata and metadata["page"] > highest_page:
                        highest_page = metadata["page"]
                    elif "primary_page" in metadata and metadata["primary_page"] > highest_page:
                        highest_page = metadata["primary_page"]
                    elif "pages_list" in metadata:
                        pages_list = [int(p) for p in metadata["pages_list"].split(",") if p]
                        if pages_list and max(pages_list) > highest_page:
                            highest_page = max(pages_list)
        except Exception:
            pass

        # Check pages collection
        try:
            results = self.pages_collection.get(
                where={"file": filename},
                include=["metadatas"],
            )

            if results and results["metadatas"]:
                for metadata in results["metadatas"]:
                    if "page" in metadata and metadata["page"] > highest_page:
                        highest_page = metadata["page"]
        except Exception:
            pass

        return highest_page

    def _process_batch(self, batch_data, collection):
        """Process a single batch of documents in a worker thread.

        Args:
            batch_data: Tuple of (batch_texts, batch_metadatas, batch_ids)
            collection: ChromaDB collection to add documents to

        Returns:
            Number of documents processed
        """
        batch_texts, batch_metadatas, batch_ids = batch_data
        collection.add(documents=batch_texts, metadatas=batch_metadatas, ids=batch_ids)
        return len(batch_texts)

    def _add_docs_to_collection(self, docs: List, collection, progress_desc: str) -> None:
        """Helper method to add documents to a specific collection."""
        # Extract text content and metadata
        texts = []
        metadatas = []

        for doc in docs:
            if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                # LangChain Document type
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
            elif isinstance(doc, dict) and "content" in doc and "metadata" in doc:
                # Dictionary format
                texts.append(doc["content"])
                metadatas.append(doc["metadata"])

        # Generate unique IDs for documents that include filename to avoid collisions
        ids = []
        for i, metadata in enumerate(metadatas):
            # Extract filename from metadata to create a unique prefix
            filename = metadata.get("file", "unknown")

            # Generate a document ID and add it to metadata
            document_id = self.generate_document_id(filename)
            metadata["document_id"] = document_id  # Store as integer in metadata

            # Create unique ID for the chunk
            # Remove extension and replace any non-alphanumeric chars with underscore
            prefix = "".join(c if c.isalnum() else "_" for c in filename.split(".")[0])
            # Create unique ID with filename prefix
            ids.append(f"{prefix}_{i}")

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(progress_desc, total=len(texts))

            # Use smaller batch size for pages collection (which contains full page data)
            batch_size = 5 if "pages" in collection.name else 10

            # Process documents in batches
            batches = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]
                batch_ids = ids[i : i + batch_size]
                batches.append((batch_texts, batch_metadatas, batch_ids))

            # For pages collection, process sequentially to avoid temp file conflicts
            if "pages" in collection.name:
                for batch in batches:
                    batch_texts, batch_metadatas, batch_ids = batch
                    try:
                        collection.add(
                            documents=batch_texts, metadatas=batch_metadatas, ids=batch_ids
                        )
                        progress.update(task, advance=len(batch_texts))
                    except Exception as e:
                        console.print(f"[bold red]Error processing batch:[/] {str(e)}")
            else:
                # Use ThreadPoolExecutor for normal collections with searchable content
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_parallel_threads
                ) as executor:
                    process_func = partial(self._process_batch, collection=collection)
                    futures = [executor.submit(process_func, batch) for batch in batches]

                    # Update progress as futures complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            batch_size = future.result()
                            progress.update(task, advance=batch_size)
                        except Exception as exc:
                            console.print(f"[bold red]Error processing batch:[/] {str(exc)}")

        console.print(f"[bold green]Success:[/] Added {len(texts)} documents to collection")

    def similarity_search(self, query: str, k: int = 10) -> List[Dict[Any, Any]]:
        """Search for documents similar to the query.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of document chunks that match the query
        """
        try:
            console.print(
                f"[bold blue]======== VECTOR STORE SIMILARITY SEARCH ========[/]\nQuery: '{query}'\nResults: {k}"
            )

            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)

            # Search for similar documents
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            # Format results to match original structure
            formatted_results = []

            if not results or "documents" not in results or not results["documents"]:
                console.print("[bold yellow]No results found in vector store[/]")
                return []

            console.print(f"[bold green]Found {len(results['documents'][0])} matching documents[/]")

            for i, (doc, metadata, distance) in enumerate(
                zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
            ):
                formatted_results.append(
                    {"content": doc, "metadata": metadata, "distance": distance, "rank": i + 1}
                )

            return formatted_results

        except Exception as e:
            console.print(f"[bold red]Error in similarity search:[/] {str(e)}")
            return []

    def similarity_search_by_page(
        self,
        filename_or_id: Union[str, int],
        page_number: int,
        query: Optional[str] = None,
        k: int = 10,
    ) -> List[Dict[Any, Any]]:
        """Get content chunks for a specific page of a PDF.

        Args:
            filename_or_id: Name of the PDF file or document ID (integer)
            page_number: Page number to retrieve
            query: Optional search query to find relevant chunks within the page
            k: Maximum number of results to return

        Returns:
            List of document chunks from the specified page
        """
        try:
            # Check if input is a document ID (either integer or string that can be converted)
            is_document_id = isinstance(filename_or_id, int) or (
                isinstance(filename_or_id, str) and filename_or_id.isdigit()
            )

            # Convert to integer if it's a digit string
            if isinstance(filename_or_id, str) and filename_or_id.isdigit():
                filename_or_id = int(filename_or_id)

            if is_document_id:
                # Search by document_id field in metadata
                where_clause = {"document_id": filename_or_id}
                console.print(f"[bold blue]Searching by document ID:[/] {filename_or_id}")
            else:
                # Search by filename
                where_clause = {"file": filename_or_id}
                console.print(f"[bold blue]Searching by filename:[/] {filename_or_id}")

            if query:
                # If query provided, do similarity search first but filter by document and page
                results = self.similarity_search(query, k=k * 2)  # Get more results, then filter

                # Filter results by page and document
                page_results = []
                for result in results:
                    metadata = result.get("metadata", {})

                    document_match = False
                    # Check if document matches
                    if is_document_id:
                        if "document_id" in metadata and metadata["document_id"] == filename_or_id:
                            document_match = True
                    else:
                        if "file" in metadata and metadata["file"] == filename_or_id:
                            document_match = True

                    if document_match:
                        # Check if page matches - handle different page storage formats
                        page_match = False

                        # Check in pages list if it exists
                        if "pages" in result and page_number in result["pages"]:
                            page_match = True
                        # Check in pages_list string if it exists
                        elif "pages_list" in metadata:
                            pages_list = [int(p) for p in metadata["pages_list"].split(",") if p]
                            if page_number in pages_list:
                                page_match = True
                        # Check primary_page
                        elif "primary_page" in metadata and page_number == metadata["primary_page"]:
                            page_match = True
                        # Check page field directly
                        elif "page" in metadata and page_number == metadata["page"]:
                            page_match = True

                        if page_match:
                            page_results.append(result)

                # Return top k results after filtering
                return page_results[:k]
            else:
                # If no query, just get all chunks for the page
                results = self.collection.get(
                    where=where_clause, include=["documents", "metadatas"]
                )

                if not results or "documents" not in results or not results["documents"]:
                    console.print(
                        "[bold yellow]No results found for document in main collection[/]"
                    )
                    return []

                # Format results to match original structure and filter for page
                formatted_results = []

                for i, (doc, metadata) in enumerate(
                    zip(results["documents"], results["metadatas"])
                ):
                    # Check if chunk includes this page using various page storage formats
                    page_match = False

                    # Check in pages_list string if it exists
                    if "pages_list" in metadata:
                        # Convert comma-separated string to list of integers
                        pages_list = [int(p) for p in metadata["pages_list"].split(",") if p]
                        if page_number in pages_list:
                            page_match = True
                    # Check primary_page
                    elif "primary_page" in metadata and page_number == metadata["primary_page"]:
                        page_match = True
                    # Check page field directly
                    elif "page" in metadata and page_number == metadata["page"]:
                        page_match = True

                    if page_match:
                        formatted_results.append(
                            {"content": doc, "metadata": metadata, "rank": i + 1}
                        )

                console.print(
                    f"[bold green]Found {len(formatted_results)} chunks for page {page_number}[/]"
                )
                return formatted_results

        except Exception as e:
            console.print(f"[bold red]Error in page content search:[/] {str(e)}")
            import traceback

            console.print(traceback.format_exc())
            return []

    def get_full_page(
        self, filename_or_id: Union[str, int], page_number: int
    ) -> Optional[Dict[Any, Any]]:
        """Get the full page data including text and image.

        Args:
            filename_or_id: Name of the PDF file or document ID (integer)
            page_number: Page number to retrieve

        Returns:
            Full page data if found, None otherwise
        """
        try:
            console.print(
                f"[bold blue]Searching for page {page_number} in document:[/] {filename_or_id}"
            )

            # Check if input is a document ID (either integer or string that can be converted)
            is_document_id = isinstance(filename_or_id, int) or (
                isinstance(filename_or_id, str) and filename_or_id.isdigit()
            )

            # Convert to integer if it's a digit string
            if isinstance(filename_or_id, str) and filename_or_id.isdigit():
                filename_or_id = int(filename_or_id)

            # Determine if we're querying by document ID or filename
            if is_document_id:
                # Query by document_id field in metadata
                where_conditions = {
                    "$and": [
                        {"document_id": {"$eq": filename_or_id}},
                        {"page": {"$eq": page_number}},
                        {"is_full_page": {"$eq": True}},
                    ]
                }
                search_by = f"document ID: {filename_or_id}"
            else:
                # Query by filename
                where_conditions = {
                    "$and": [
                        {"file": {"$eq": filename_or_id}},
                        {"page": {"$eq": page_number}},
                        {"is_full_page": {"$eq": True}},
                    ]
                }
                search_by = f"filename: {filename_or_id}"

            # Query pages collection to find the full page entry
            console.print(f"Querying pages collection for {search_by}, page {page_number}")
            results = self.pages_collection.get(
                where=where_conditions,
                include=["documents", "metadatas"],
            )

            if not results or "documents" not in results or not results["documents"]:
                console.print(
                    f"[bold yellow]Page {page_number} not found for {search_by} in pages collection[/]"
                )
                return None

            console.print(f"[bold green]Found page {page_number} for {search_by}[/]")
            # Return the first full page entry
            return {
                "content": results["documents"][0],
                "metadata": results["metadatas"][0],
                "page": page_number,
                "file": results["metadatas"][0].get("file"),
                "document_id": results["metadatas"][0].get("document_id"),
            }

        except Exception as e:
            console.print(f"[bold red]Error retrieving full page:[/] {str(e)}")
            import traceback

            console.print(traceback.format_exc())
            return None
