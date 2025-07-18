# add processing for link to a PDF file

"""
Content Processing Utility Module

Features:
1. URL Processing
   - URL validation and parsing
   - YouTube URL detection and ID extraction
   - Web content scraping with Crawl4ai
   - Content length management
   - Image extraction and analysis
   - Image relevance detection with LLM

2. PDF Processing
   - Local and remote PDF handling
   - Text extraction and processing
   - Content length management
   - Async support for Slack integration

3. YouTube Integration
   - Video ID extraction from various URL formats
   - Transcript extraction and processing
   - Error handling for unavailable transcripts
   - Robust error handling for YouTube transcript API and XML parsing

4. Content Management
   - Content length validation
   - Text truncation
   - Format conversion
   - Character encoding handling
   - Image processing and filtering
   - Image relevance analysis

5. Debug Support
   - Detailed operation logging
   - Error tracking
   - Content length monitoring
   - Processing status updates
   - Image analysis logging

6. Content Embedding
   - Generate embeddings for article summaries
   - Store embeddings in memory manager
   - Support for similarity search
"""

import io
import re
import os
import base64
from PIL import Image
from typing import Optional, NamedTuple, List, Tuple
from urllib.parse import urlparse, parse_qs, unquote
from crawl4ai import AsyncWebCrawler
from youtube_transcript_api import YouTubeTranscriptApi
import PyPDF2
from utils import dprint
from openai import OpenAI
import aiohttp
import magic
from io import BytesIO
import yt_dlp
from memoripy import MemoryManager
import xml.etree.ElementTree as ET

# Debug flag
DEBUG: bool = False
MAX_CONTENT_LENGTH: int = 100000
MIN_IMAGE_SIZE: int = 300  # Minimum width/height in pixels
IMAGE_TEMP_DIR: str = ".images"

# Ensure temp directory exists
os.makedirs(IMAGE_TEMP_DIR, exist_ok=True)

# Create client instance
client = OpenAI(api_key=os.getenv("RODION_OPENAI_API_KEY"))


class ImageAnalysis(NamedTuple):
    """Container for image analysis results"""

    path: str
    dimensions: Tuple[int, int]
    is_relevant: bool
    description: Optional[str]


class ScrapingResult(NamedTuple):
    """Container for scraping results"""

    content: Optional[str]
    success: bool
    error_message: Optional[str]
    relevant_images: List[ImageAnalysis]

    def __str__(self) -> str:
        """Convert scraping result to string format"""
        if self.success and self.content:
            return self.content
        return (
            f"Error: {self.error_message}"
            if self.error_message
            else "Unknown error occurred"
        )


def analyze_image_relevance(
    image_path: str, page_content: str, client: OpenAI
) -> Tuple[bool, Optional[str]]:
    """Analyze if image is relevant to content using LLM"""
    try:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Is this image relevant to the following content? Respond with 'yes' or 'no' and a brief explanation.\n\nContent: {page_content}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )

        result = response.choices[0].message.content.lower()
        is_relevant = result.startswith("yes")
        dprint(f"Image relevance analysis: {result}")
        return is_relevant, result

    except Exception as e:
        dprint(f"Error analyzing image relevance: {str(e)}", error=True)
        return False, None


def sanitize_filename(url: str) -> str:
    """Create a safe filename from URL"""
    # Get the path part of the URL
    parsed = urlparse(unquote(url))
    path = parsed.path

    # Get the last part of the path (filename)
    filename = path.split("/")[-1]

    # If no filename found, use the hostname
    if not filename:
        filename = parsed.hostname or "image"

    # Remove query parameters if present
    filename = filename.split("?")[0]

    # Remove or replace invalid characters
    filename = re.sub(r"[^\w\-_.]", "_", filename)

    # Ensure we have an extension
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
        filename += ".jpg"

    # Limit length
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:95] + ext

    return filename


async def download_image(url: str, headers: dict) -> Optional[bytes]:
    """Download image from URL and return as bytes

    Args:
        url: URL of the image to download
        headers: Headers to use for the request

    Returns:
        Bytes of the image if successful, None otherwise
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    dprint(
                        f"Failed to download image {url}: HTTP {response.status}",
                        error=True,
                    )
                    return None
                return await response.read()
    except Exception as e:
        dprint(f"Error downloading image {url}: {str(e)}", error=True)
        return None


async def extract_images_from_url(
    url: str, page_content: str, client: OpenAI
) -> List[ImageAnalysis]:
    """Extract all images above minimum size from URL"""
    # Create images directory if it doesn't exist
    if not os.path.exists(IMAGE_TEMP_DIR):
        os.makedirs(IMAGE_TEMP_DIR)
        dprint(f"Created image directory: {IMAGE_TEMP_DIR}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": url,
    }

    images = []
    base_domain = urlparse(url).netloc

    # Use crawl4ai to get images
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)

            if result.success:
                image_info = result.media.get("images", [])
                dprint(f"Found {len(image_info)} images with crawl4ai")

                # Get URLs from all images
                for img in image_info:
                    img_url = img["src"]
                    img_domain = urlparse(img_url).netloc

                    # Double-check domain even with exclude_external_images
                    if img_domain and img_domain in base_domain:
                        images.append(
                            {
                                "url": img_url,
                                "alt": img.get("alt", ""),
                                "desc": img.get("desc", ""),
                            }
                        )

    except Exception as e:
        dprint(f"Crawl4ai image extraction failed: {str(e)}", error=True)

    dprint(f"Total valid images found: {len(images)}")

    # Process found images
    collected_images = []
    images_size_filtered = 0

    for img_info in images:
        try:
            img_url = img_info["url"]
            dprint(f"Processing image {img_url}")

            # Download image using async function
            image_data = await download_image(img_url, headers)
            if not image_data:
                continue

            # Create safe filename
            filename = sanitize_filename(img_url)
            img_path = os.path.join(IMAGE_TEMP_DIR, filename)

            # Save image to temp file
            with open(img_path, "wb") as f:
                f.write(image_data)

            # Check dimensions
            with Image.open(img_path) as img:
                width, height = img.size
                # dprint(f"Image dimensions: {width}x{height}")
                if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                    # dprint(
                    #     f"Image too small, skipping (minimum size: {MIN_IMAGE_SIZE}px)"
                    # )
                    os.remove(img_path)
                    images_size_filtered += 1
                    continue

                # dprint("Image meets size requirements, keeping")
                collected_images.append(
                    ImageAnalysis(img_path, (width, height), True, None)
                )

        except Exception as e:
            dprint(f"Error processing image {img_url}: {str(e)}", error=True)
            continue

    dprint("Image processing summary:")
    dprint(f"- Total images found: {len(images)}")
    dprint(f"- Images filtered by size: {images_size_filtered}")
    dprint(f"- Images kept: {len(collected_images)}")

    return collected_images


def is_url(text: str) -> bool:
    """Check if text is a valid URL

    Handles both regular URLs and Slack-formatted URLs (enclosed in < >)
    """
    # Remove Slack URL formatting if present
    text = text.strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
        # Slack URLs may have a pipe with display text - remove it
        if "|" in text:
            text = text.split("|")[0]

    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return bool(re.match(url_pattern, text))


def get_youtube_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL"""
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        return parsed.path[1:]
    if parsed.hostname in ("youtube.com", "www.youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
    return None


def get_youtube_info(video_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Get transcript and title from YouTube video

    Args:
        video_id: YouTube video ID

    Returns:
        Tuple of (transcript text, video title)
    """
    try:
        # Get transcript using the correct API usage
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            transcript = " ".join(item["text"] for item in transcript_list)
        except Exception as e:
            dprint(f"Error getting YouTube transcript: {str(e)}", error=True)
            return None, None

        # Get video title using yt-dlp
        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                title = info.get("title")
                if title:
                    dprint(f"Retrieved YouTube video title: {title}")
            except Exception as e:
                dprint(f"Error getting video title: {str(e)}", error=True)
                title = None

        return transcript, title

    except Exception as e:
        dprint(f"Error getting YouTube info: {str(e)}", error=True)
        return None, None


async def is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF file"""
    # Quick check of URL extension or arXiv PDF pattern
    if url.lower().endswith(".pdf") or "arxiv.org/pdf/" in url.lower():
        return True

    # Check content type header
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with aiohttp.ClientSession() as session:
            async with session.head(
                url, allow_redirects=True, headers=headers
            ) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                return "application/pdf" in content_type
    except Exception as e:
        dprint(f"Error checking PDF URL: {str(e)}", error=True)
        return False


async def download_pdf(url: str) -> Optional[bytes]:
    """Download PDF file from URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    dprint(
                        f"Failed to download PDF: HTTP {response.status}", error=True
                    )
                    return None

                content = await response.read()

                # Verify it's actually a PDF using python-magic
                mime = magic.from_buffer(content, mime=True)
                if "application/pdf" not in mime:
                    dprint(f"Downloaded content is not PDF: {mime}", error=True)
                    return None

                return content
    except Exception as e:
        dprint(f"Error downloading PDF: {str(e)}", error=True)
        return None


async def extract_pdf_content_from_url(url: str) -> Optional[str]:
    """Extract text from a PDF file at given URL

    Args:
        url: URL of the PDF file

    Returns:
        Extracted text if successful, None otherwise
    """
    dprint(f"Downloading PDF from URL: {url}")
    try:
        pdf_data = await download_pdf(url)
        if not pdf_data:
            return None

        pdf_file = BytesIO(pdf_data)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "\n".join(page.extract_text() for page in pdf_reader.pages)

        if len(text.strip()) == 0:
            dprint("Extracted empty text from PDF", error=True)
            return None

        dprint(f"Successfully extracted {len(text)} characters from PDF")
        return text

    except Exception as e:
        dprint(f"Error extracting PDF text: {str(e)}", error=True)
        return None


async def store_content_embedding(content: str, memory_manager: MemoryManager) -> None:
    """Store content embedding in memory manager for later similarity comparison

    Args:
        content: The text content to generate embedding for
        memory_manager: MemoryManager instance to store the embedding
    """
    try:
        # Clean the text before generating embedding
        cleaned_content = clean_text_for_embedding(content)

        # Generate embedding using memory manager
        embedding = memory_manager.get_embedding(cleaned_content)

        # Extract concepts for better context
        concepts = memory_manager.extract_concepts(cleaned_content)

        # Store in memory manager with embedding and concepts
        memory_manager.add_memory(
            cleaned_content, embedding=embedding, concepts=concepts
        )

        dprint(f"Successfully stored embedding for content ({len(content)} chars)")

    except Exception as e:
        dprint(f"Error storing content embedding: {str(e)}", error=True)


async def scrape_url_content(
    url: str,
    extract_images: bool = False,
    memory_manager: Optional[MemoryManager] = None,
) -> ScrapingResult:
    """Scrape content from a URL with fallback mechanisms"""
    # Clean URL from Slack formatting
    url = url.strip()
    if url.startswith("<") and url.endswith(">"):
        url = url[1:-1]
        # Handle Slack's display text format
        if "|" in url:
            url = url.split("|")[0]

    dprint(f"Processing URL: {url}")

    # Check if PDF URL first - before YouTube check
    if await is_pdf_url(url):
        dprint("Detected PDF URL")
        content = await extract_pdf_content_from_url(url)
        if content:
            return ScrapingResult(f"Content from PDF:\n\n{content}", True, None, [])
        return ScrapingResult(None, False, "Failed to extract PDF content", [])

    # Check if YouTube URL next
    video_id = get_youtube_id(url)
    if video_id:
        dprint(f"Processing YouTube video: {video_id}")
        transcript, title = get_youtube_info(video_id)
        if transcript:
            content = (
                f"Title: {title}\n\nTranscript:\n{transcript}" if title else transcript
            )
            return ScrapingResult(
                f"Content from YouTube video:\n\n{content}", True, None, []
            )
        return ScrapingResult(None, False, "Failed to get YouTube transcript", [])

    dprint(f"Scraping URL: {url}")
    content = None
    relevant_images = []

    dprint("Attempting crawl4ai extraction...")
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result.markdown:
                content = result.markdown
                dprint(f"Successfully extracted {len(content)} chars with crawl4ai")
    except Exception as e:
        dprint(f"Crawl4ai extraction failed: {str(e)}", error=True)

    if content:
        # Only analyze images if requested
        if extract_images:
            dprint("Analyzing images...")
            relevant_images = await extract_images_from_url(url, content, client)
        else:
            dprint("Skipping image extraction")

        # Store embedding if memory manager provided
        if memory_manager:
            await store_content_embedding(content, memory_manager)

        # Truncate if too long
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "..."

        dprint(
            f"Successfully scraped {len(content)} characters and {len(relevant_images)} relevant images"
        )
        return ScrapingResult(content, True, None, relevant_images)

    error_msg = "Failed to extract any content"
    dprint(error_msg, error=True)
    return ScrapingResult(None, False, error_msg, [])


def clean_text_for_embedding(text: str) -> str:
    """Clean and prepare text for embedding model

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text suitable for embedding generation
    """
    if not isinstance(text, str):
        return str(text)

    # Remove URLs
    text = re.sub(r"http[s]?://\S+", "", text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s.,!?-]", "", text)

    # Normalize whitespace
    text = " ".join(text.split())

    # Truncate if too long (embedding models often have token limits)
    max_chars = 8000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    return text
