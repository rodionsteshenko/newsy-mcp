#!/usr/bin/env python3
"""
News Article Bot

Features:
1. Article Pipeline
   - Periodic batch processing every 5 minutes:
     - RSS feed downloading
     - Automatic article processing with LLM
     - Article distribution to users
   - All operations in UTC for consistency
   - Reaction tracking and filtering
   - Detailed reaction debugging and validation
   - Comprehensive debug logging of all operations
   - Proper async/await handling for coroutines
   - Time-based relevance scoring
   - Tracking of sent articles to prevent duplicates

2. Article Distribution
   - Sends articles in batches after processing
   - Filters for articles from last 24 hours
   - Includes relevance scores and summaries
   - Interactive reaction buttons
   - Strict filtering of previously rated articles
   - Detailed reaction tracking and validation
   - Comprehensive reaction database debugging
   - Debug logging of article selection and sending
   - Time-aware article relevance evaluation
   - Prevents resending of previously sent articles

3. User Feedback
   - Thumbs up/down/neutral reactions
   - Stores user relevance scores
   - Updates user preferences via LLM
   - Gradual preference adaptation
   - Prevents duplicate reactions
   - Detailed reaction validation logging
   - Debug logging of reaction processing
   - Generates audio summary for liked articles

4. Database Integration
   - Unified SQLite database access
   - Tracks article reactions and scores
   - Maintains user preferences
   - Caches processed articles
   - Reaction validation and verification
   - Comprehensive reaction debugging
   - Debug logging of database operations
   - Time-based article relevance tracking
   - Records sent articles to prevent duplicates

5. Debug Support
   - Detailed logging of pipeline stages
   - Error tracking and reporting
   - Performance monitoring
   - Reaction tracking validation
   - Database query verification
   - Reaction database state logging
   - Article filtering debug output
   - RSS feed download logging
   - Article processing logging
   - Database operation logging
"""

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
import sqlite3
from datetime import datetime, timedelta, timezone
import json
from typing import List, Dict, Optional
from rich.console import Console
import asyncio
from process_articles import load_user_preferences, process_articles
import random
import os
from tts import TTS
from gpt_util import GptUtil
from utils import dprint
from rss_downloader import RssDownloader
from content_util import clean_text_for_embedding
from memoripy import MemoryManager, JSONStorage
from memoripy.implemented_models import OpenAIChatModel, OllamaEmbeddingModel
from aibo_types import OPENAI_API_KEY
import faiss
import numpy as np
from pathlib import Path
import traceback

# Configuration
SLACK_BOT_TOKEN = os.getenv("SLACK_NEWSY_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_NEWSY_APP_TOKEN")
ALLOWED_CHANNEL = "C0123456789"
DEBUG = True

# Pipeline intervals (in seconds)
BATCH_INTERVAL = 60 * 60

# Maximum age of articles to serve
ARTICLE_WINDOW_HOURS = 2

# File paths
CHANNEL_ID_FILE = "channel_id.txt"
AUDIO_DIR = "audio_summaries"
USER_PREFERENCES_FILE = "user_preferences.txt"

# Update constants
TECH_ARTICLES = 10
ENTERTAINMENT_ARTICLES = 10
LIKELY_POSITIVE_COUNT = 0  # Number of likely positive articles
NEUTRAL_COUNT = 1  # Number of neutral articles
LIKELY_NEGATIVE_COUNT = 1  # Number of likely negative articles
MAX_PREF_CHANGE = 0.25  # Maximum 25% change in preferences

# Add constants
FAISS_INDEX_DIR = "faiss_indexes"
FAISS_INDEX_PATH = Path(FAISS_INDEX_DIR) / "article_summaries.index"
EMBEDDING_DIMENSION = 1024  # ollama mxbai-embed-large dimension size

# Initialize console
console = Console()

# Initialize Slack app
app = AsyncApp(token=SLACK_BOT_TOKEN)

# Track active channels
likes_channel_id: Optional[str] = None
meh_channel_id: Optional[str] = None
dislikes_channel_id: Optional[str] = None

# Create audio directory if it doesn't exist
os.makedirs(AUDIO_DIR, exist_ok=True)

# Create directory
Path(FAISS_INDEX_DIR).mkdir(exist_ok=True)


def calculate_time_relevance(article_timestamp: str) -> float:
    """Calculate relevance factor based on article age"""
    article_time = datetime.fromisoformat(article_timestamp)
    current_time = datetime.now(timezone.utc)
    age_hours = (current_time - article_time).total_seconds() / 3600

    # Linear decay from 1.0 to 0.1 over the article window
    relevance = 1.0 - (0.9 * (age_hours / ARTICLE_WINDOW_HOURS))
    return max(0.1, relevance)


# Load saved channel IDs
def load_channel_ids():
    """Load channel IDs from file"""
    global likes_channel_id, meh_channel_id, dislikes_channel_id
    try:
        if os.path.exists(CHANNEL_ID_FILE):
            with open(CHANNEL_ID_FILE, "r") as f:
                channels = json.load(f)
                likes_channel_id = channels.get("likes")
                meh_channel_id = channels.get("meh")
                dislikes_channel_id = channels.get("dislikes")
                dprint(
                    f"Loaded channel IDs from file: likes={likes_channel_id}, meh={meh_channel_id}, dislikes={dislikes_channel_id}"
                )
    except Exception as e:
        dprint(f"Error loading channel IDs: {e}", error=True)


# Load channel IDs at startup
load_channel_ids()


async def generate_audio_summary(article_id: int) -> Optional[dict]:
    """Generate detailed audio summary for liked article using TTS"""
    try:
        conn = sqlite3.connect("newsy.db")
        cursor = conn.execute(
            """
            SELECT a.title, an.summary, a.content_html2text 
            FROM articles a
            JOIN article_analysis an ON an.article_id = a.id
            WHERE a.id = ?
            """,
            (article_id,),
        )
        result = cursor.fetchone()
        conn.close()

        if not result:
            return None

        title, summary, content = result

        # Generate detailed summary using GPT
        gpt = GptUtil()
        detailed_summary = await gpt.send_prompt(
            f"""Please provide a detailed summary of this article in a natural, conversational tone suitable for text-to-speech.
            - Introduce the article title in the beginning
            - Use complete sentences and natural transitions
            - Focus on key points and interesting details
            - Make it engaging and easy to follow when listened to
            - Aim for no more than 400 words, but the length should vary depending on the article
            
            Article content:
            {content}"""
        )

        # Generate audio from detailed summary
        tts = TTS()
        done = asyncio.Event()
        output_file = None
        error = None

        async def audio_callback(filename: Optional[str], err: Optional[Exception]):
            nonlocal output_file, error
            output_file = filename
            error = err
            done.set()

        # Generate audio with default settings
        tts.generate_audio(detailed_summary, audio_callback)
        await done.wait()

        if error:
            dprint(f"Error generating audio summary: {error}", error=True)
            return None

        return {"audio_file": output_file, "text_summary": detailed_summary}

    except Exception as e:
        dprint(f"Error generating audio summary: {e}", error=True)
        return None


async def send_news_digest(channel: str, digest: str) -> None:
    """Send news digest as a single message"""
    try:
        dprint("\nSending news digest...")

        # Use an emoji to break up the Slack messages
        digest = ":star:" * 15 + "\n" + digest

        # Send the entire digest as one message
        await app.client.chat_postMessage(
            channel=channel, text=digest, unfurl_links=False
        )

        dprint("Sent news digest")

    except Exception as e:
        dprint(f"Error sending news digest: {e}", error=True)


def fix_digest_formatting(digest: str) -> str:
    """Fix digest formatting issues by properly formatting URLs and other elements

    Args:
        digest: Raw digest text from GPT

    Returns:
        Properly formatted digest text
    """
    import re

    # Fix URLs that are missing < > characters
    # Matches: https://example.com/(Read more on Source)
    # But excludes: <https://example.com/|(Read more on Source)>
    url_pattern = r"(?<!<)(https?://[^\s]+)\(Read more on ([^)]+)\)(?!>)"
    fixed_digest = re.sub(url_pattern, r"<\1|(Read more on \2)>", digest)

    # Fix URLs that have < > but are missing the | separator
    # Matches: <https://example.com/(Read more on Source)>
    # Also handles cases with extra spaces before closing >
    malformed_url_pattern = r"<(https?://[^\s]+)\(Read more on ([^)]+)\)\s*>"
    fixed_digest = re.sub(
        malformed_url_pattern, r"<\1|(Read more on \2)>", fixed_digest
    )

    # Fix URLs that have double || separators
    # Matches: <url||(...>
    double_pipe_pattern = r"<(https?://[^\s]+)\|\|(\(Read more on [^>]+\)>)"
    fixed_digest = re.sub(double_pipe_pattern, r"<\1|\2", fixed_digest)

    return fixed_digest


async def generate_news_digest(
    tags: List[str], max_articles: int = 10, hours: int = ARTICLE_WINDOW_HOURS
) -> Optional[str]:
    """Generate a digest of likely interesting articles from the current window

    Args:
        tags: List of tags to filter articles by
        max_articles: Maximum number of articles to include
        hours: How many hours back to look for articles (default: ARTICLE_WINDOW_HOURS)
    """
    try:
        dprint(f"\nGenerating news digest for tags: {tags}")
        conn = sqlite3.connect("newsy.db")
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Build tag conditions for SQL query
        tag_conditions = []
        params = [cutoff.isoformat()]
        for tag in tags:
            tag_conditions.append("an.tags LIKE ?")
            params.append(f"%{tag}%")

        # Modified query to use timestamp instead of created_at
        query = """
            SELECT 
                a.title,
                a.url,
                an.summary,
                an.relevance,
                an.tags
            FROM articles a
            JOIN article_analysis an ON an.article_id = a.id
            WHERE a.timestamp > ?
            AND an.relevance > 0.3
            AND (
                {}
            )
            ORDER BY an.relevance DESC
        """.format(" OR ".join(tag_conditions))

        cursor = conn.execute(query, params)
        articles = cursor.fetchall()

        conn.close()

        dprint(f"\nFound {len(articles)} matching articles for tags {tags}")

        # Randomly sample if we have more articles than the limit
        if len(articles) > max_articles:
            dprint(f"Sampling {max_articles} articles from {len(articles)} total")
            articles = random.sample(articles, max_articles)

        # Debug output
        for title, url, _, relevance, article_tags in articles:
            dprint(f"  Title: {title}")
            dprint(f"  URL: {url}")
            dprint(f"  Relevance: {relevance}")
            dprint(f"  Tags: {article_tags}")
            dprint("  ---")

        if not articles:
            dprint("No articles found for digest")
            return None

        # Format articles for the prompt
        articles_text = "\n\n".join(
            [
                f"Title: {title}\nURL: {url}\nSummary: {summary}"
                for title, url, summary, *_ in articles
            ]
        )

        dprint("\nGenerating digest with GPT...")
        # Generate digest using GPT
        gpt = GptUtil()
        digest = await gpt.send_prompt(
            f"""- Create a comprehensive news digest for {", ".join(tags)} news using this Slack markup template.
- Don't include the triple ' or ` or ".
- Be sure to bold text with a single * on each side and seen in the template.
- Combine into no more than 4 sections.
- Replace EMOJI, SUBJECT, URL, SECTION NAME, SOURCE NAME.

'''  
:newspaper: *{", ".join(tags).title()} News* | `{datetime.now().strftime("%Y-%m-%d")}`

EMOJI *SECTION NAME*
• EMOJI *SUBJECT* 1-2 sentence summary. <URL|(Read more on SOURCE NAME)>
• EMOJI *SUBJECT* 1-2 sentence summary. <URL|(Read more on SOURCE NAME)>
• ...

EMOJI *SECTION NAME*
• EMOJI *SUBJECT* 1-2 sentence summary. <URL|(Read more on SOURCE NAME)>
• EMOJI *SUBJECT* 1-2 sentence summary. <URL|(Read more on SOURCE NAME)>
• ...
'''

Using these articles:
{articles_text}"""
        )

        # Fix any formatting issues in the digest
        digest = fix_digest_formatting(digest)

        dprint("\nGenerated digest content:")
        dprint(digest)

        return digest

    except Exception as e:
        dprint(f"Error generating news digest: {e}", error=True)
        return None


def init_faiss_index() -> faiss.Index:
    """Initialize or load FAISS index for article summary embeddings"""
    try:
        # If index exists but has wrong dimension, delete it
        if FAISS_INDEX_PATH.exists():
            try:
                index = faiss.read_index(str(FAISS_INDEX_PATH))
                if index.d != EMBEDDING_DIMENSION:
                    dprint(
                        f"Existing index has wrong dimension ({index.d}), recreating..."
                    )
                    FAISS_INDEX_PATH.unlink()  # Delete the file
                else:
                    dprint("Loading existing FAISS index")
                    return index
            except Exception as e:
                dprint(f"Error reading existing index: {str(e)}, recreating...")
                FAISS_INDEX_PATH.unlink()  # Delete corrupted file

        # Create new index
        dprint(f"Creating new FAISS index with dimension {EMBEDDING_DIMENSION}")
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)

        # Verify dimension
        if index.d != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Created index has wrong dimension: {index.d} != {EMBEDDING_DIMENSION}"
            )

        faiss.write_index(index, str(FAISS_INDEX_PATH))
        dprint("Successfully created and saved new FAISS index")
        return index

    except Exception as e:
        dprint(f"Error initializing FAISS index: {str(e)}", error=True)
        dprint(f"Traceback: {traceback.format_exc()}")
        raise


async def process_articles_with_embeddings() -> None:
    """Process new articles from last 24 hours and store summary embeddings"""
    try:
        conn = sqlite3.connect("newsy.db")
        memory_manager = init_memory_manager()
        faiss_index = init_faiss_index()

        # Get recent articles with summaries but no embeddings
        cursor = conn.execute(
            """
            SELECT a.id, an.summary, a.title, a.timestamp 
            FROM articles a
            JOIN article_analysis an ON an.article_id = a.id
            WHERE a.timestamp > datetime('now', '-24 hours')
            AND an.summary IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM article_embeddings 
                WHERE article_id = a.id
            )
        """
        )

        for article_id, summary, title, timestamp in cursor:
            try:
                dprint(f"\nProcessing article {article_id}: {title}")

                # Generate embedding
                cleaned_summary = clean_text_for_embedding(summary)
                if not cleaned_summary.strip():
                    continue

                embedding = memory_manager.get_embedding(cleaned_summary)

                # Ensure embedding is 2D array with shape (1, dimension)
                embedding_array = np.array(embedding).astype("float32")
                if len(embedding_array.shape) == 1:
                    embedding_array = embedding_array.reshape(1, -1)

                # Store raw bytes in SQLite
                embedding_bytes = embedding_array.tobytes()

                # Store in SQLite
                conn.execute(
                    """
                    INSERT INTO article_embeddings 
                    (article_id, embedding, processed_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    """,
                    (article_id, embedding_bytes),
                )
                conn.commit()

                # Add to FAISS index
                faiss_index.add(
                    embedding_array
                )  # Should now have correct shape (1, dimension)

                dprint(f"Generated and stored embedding for article {article_id}")

            except Exception as e:
                dprint(f"Error processing article {article_id}: {str(e)}", error=True)
                dprint(
                    f"Embedding shape: {embedding_array.shape if 'embedding_array' in locals() else 'unknown'}"
                )
                continue

        # Save updated FAISS index
        faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
        dprint("Saved updated FAISS index")

    except Exception as e:
        dprint(f"Error in process_articles_with_embeddings: {str(e)}", error=True)
    finally:
        conn.close()


async def run_batch_pipeline() -> None:
    """Run complete batch pipeline"""
    try:
        # Download RSS feeds
        dprint("Starting RSS feed download...")
        downloader = RssDownloader()
        await downloader.run()
        dprint("RSS feed download completed")

        # Process articles
        dprint("Starting article processing...")
        await process_articles()  # Original article processing
        await process_articles_with_embeddings()  # Add embeddings
        dprint("Article processing completed")

        dprint(f"Batch pipeline complete. Next run in {BATCH_INTERVAL} seconds")

    except Exception as e:
        dprint(f"Critical error in batch pipeline: {e}", error=True)
        raise


def migrate_database() -> None:
    """Run any necessary database migrations"""
    conn = sqlite3.connect("newsy.db")
    try:
        # Check if embedding column exists
        cursor = conn.execute("PRAGMA table_info(article_embeddings)")
        columns = [row[1] for row in cursor.fetchall()]

        if "embedding" not in columns:
            dprint("Adding embedding column to article_embeddings table...")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS article_embeddings_new (
                    id INTEGER PRIMARY KEY,
                    article_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(article_id) REFERENCES articles(id),
                    UNIQUE(article_id)
                )
            """)

            # Copy any existing data
            conn.execute("""
                INSERT INTO article_embeddings_new (article_id, processed_at)
                SELECT article_id, processed_at FROM article_embeddings
            """)

            # Drop old table and rename new one
            conn.execute("DROP TABLE article_embeddings")
            conn.execute(
                "ALTER TABLE article_embeddings_new RENAME TO article_embeddings"
            )

            conn.commit()
            dprint("Database migration completed")

        # MIGRATION: Add relevance_details column to article_analysis if missing
        cursor = conn.execute("PRAGMA table_info(article_analysis)")
        analysis_columns = [row[1] for row in cursor.fetchall()]
        if "relevance_details" not in analysis_columns:
            dprint("Adding relevance_details column to article_analysis table...")
            # Add the column
            conn.execute("ALTER TABLE article_analysis ADD COLUMN relevance_details TEXT DEFAULT '{\"preference_matches\":[],\"preference_mismatches\":[],\"reaction_patterns\":[]}'")
            conn.commit()
            dprint("Added relevance_details column to article_analysis table")
    except Exception as e:
        dprint(f"Error during migration: {str(e)}", error=True)
        conn.rollback()
    finally:
        conn.close()


def init_database() -> None:
    """Initialize single database with all tables"""
    dprint("Initializing database...")
    conn = sqlite3.connect("newsy.db")
    conn.execute("PRAGMA foreign_keys = ON")

    # Create articles table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            feed_domain TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            content TEXT,
            content_html2text TEXT,
            content_markdown TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(url, feed_domain)
        )
    """)

    # Create article analysis table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS article_analysis (
            id INTEGER PRIMARY KEY,
            article_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            tags TEXT,
            summary TEXT,
            relevance FLOAT,
            relevance_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(article_id) REFERENCES articles(id),
            UNIQUE(article_id)
        )
    """)

    # Create reactions table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS article_reactions (
            id INTEGER PRIMARY KEY,
            article_id INTEGER NOT NULL,
            user_score INTEGER,
            reaction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(article_id) REFERENCES articles(id),
            UNIQUE(article_id)
        )
    """)

    # Create sent articles table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sent_articles (
            id INTEGER PRIMARY KEY,
            article_id INTEGER NOT NULL,
            sent_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(article_id) REFERENCES articles(id),
            UNIQUE(article_id)
        )
    """)

    # Create article embeddings table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS article_embeddings (
            id INTEGER PRIMARY KEY,
            article_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(article_id) REFERENCES articles(id),
            UNIQUE(article_id)
        )
    """)

    conn.commit()
    conn.close()

    # Run migrations after initial setup
    migrate_database()
    dprint("Database initialization complete")


def save_channel_id(channel_type: str, channel_id: str) -> None:
    """Save channel ID based on type"""
    global likes_channel_id, meh_channel_id, dislikes_channel_id

    if channel_type == "likes":
        likes_channel_id = channel_id
    elif channel_type == "meh":
        meh_channel_id = channel_id
    elif channel_type == "dislikes":
        dislikes_channel_id = channel_id

    # Save to file
    channels = {
        "likes": likes_channel_id,
        "meh": meh_channel_id,
        "dislikes": dislikes_channel_id,
    }
    try:
        with open(CHANNEL_ID_FILE, "w") as f:
            json.dump(channels, f)
        dprint(f"Saved {channel_type} channel ID: {channel_id}")
    except Exception as e:
        dprint(f"Error saving channel IDs: {e}", error=True)


def normalize_url(url: str) -> str:
    """Normalize URL to ensure consistent formatting for comparisons"""
    # Remove common variations
    url = url.replace("_", "")
    url = url.replace("-", "")
    url = url.replace(" ", "")
    url = url.lower()
    return url


# Add memory manager initialization
def init_memory_manager() -> MemoryManager:
    """Initialize memory manager for article embeddings"""
    try:
        memory_manager = MemoryManager(
            OpenAIChatModel(OPENAI_API_KEY, "gpt-4o-mini"),
            OllamaEmbeddingModel("mxbai-embed-large"),
            storage=JSONStorage("article_memory.json"),
        )
        dprint("Successfully initialized memory manager")
        return memory_manager
    except Exception as e:
        dprint(f"Error initializing memory manager: {str(e)}", error=True)
        raise


# Update get_articles_by_relevance to use embeddings for similarity
async def get_articles_by_relevance(
    count: int, min_relevance: float, max_relevance: float
) -> List[Dict]:
    """Get articles using summary embedding similarity and relevance filtering"""
    dprint(
        f"Getting {count} articles with relevance between {min_relevance} and {max_relevance}"
    )

    try:
        memory_manager = init_memory_manager()
        faiss_index = init_faiss_index()

        # Get user preferences and generate query embedding
        user_prefs = load_user_preferences()
        query_embedding = memory_manager.get_embedding(user_prefs)

        # Convert to correct format for FAISS
        query_vector = np.array([query_embedding]).astype("float32")

        # Search for similar articles (get more than needed for filtering)
        k = count * 4  # Get extra to allow for filtering
        distances, indices = faiss_index.search(query_vector, k)

        # Get article details from database
        conn = sqlite3.connect("newsy.db")
        filtered_articles = []

        for idx in indices[0]:  # indices[0] because search returns 2D array
            cursor = conn.execute(
                """
                SELECT a.id, a.title, a.url, an.tags, an.summary, an.relevance, 
                       an.relevance_reason, a.timestamp
                FROM articles a
                JOIN article_analysis an ON an.article_id = a.id
                JOIN article_embeddings ae ON ae.article_id = a.id
                WHERE ae.id = ?
                AND NOT EXISTS (
                    SELECT 1 FROM article_reactions ar WHERE ar.article_id = a.id
                )
                AND NOT EXISTS (
                    SELECT 1 FROM sent_articles sa WHERE sa.article_id = a.id
                )
            """,
                (int(idx),),
            )

            result = cursor.fetchone()
            if result and min_relevance <= result[5] <= max_relevance:
                filtered_articles.append(
                    {
                        "id": result[0],
                        "title": result[1],
                        "url": result[2],
                        "tags": json.loads(result[3]),
                        "summary": result[4],
                        "relevance": result[5],
                        "reason": result[6],
                        "timestamp": result[7],
                    }
                )

                if len(filtered_articles) >= count:
                    break

        conn.close()
        return filtered_articles

    except Exception as e:
        dprint(f"Error getting articles by relevance: {str(e)}", error=True)
        return []


async def get_balanced_article_batch() -> List[Dict]:
    """Get a balanced batch of articles with different predicted reactions"""
    articles = []

    # Get likely positive articles (relevance > 0.3)
    positives = await get_articles_by_relevance(LIKELY_POSITIVE_COUNT, 0.3, 1.0)
    articles.extend(positives)
    dprint(f"Found {len(positives)} likely positive articles")

    # Get neutral articles (-0.3 <= relevance <= 0.3)
    neutrals = await get_articles_by_relevance(NEUTRAL_COUNT, -0.3, 0.3)
    articles.extend(neutrals)
    dprint(f"Found {len(neutrals)} neutral articles")

    # Get likely negative articles (relevance < -0.3)
    negatives = await get_articles_by_relevance(LIKELY_NEGATIVE_COUNT, -1.0, -0.3)
    articles.extend(negatives)
    dprint(f"Found {len(negatives)} likely negative articles")

    # Only return if we have at least one article
    if not articles:
        dprint("No unreacted articles available")
        return []

    # Shuffle the articles
    random.shuffle(articles)

    return articles


async def send_article_batch() -> None:
    """Send a batch of articles to appropriate channels based on relevance"""
    articles = await get_balanced_article_batch()
    if articles:
        dprint(f"Sending batch of {len(articles)} articles")

        conn = sqlite3.connect("newsy.db")
        conn.execute("PRAGMA foreign_keys = ON")

        for article in articles:
            # Calculate time since article publication
            article_time = datetime.fromisoformat(article["timestamp"])
            current_time = datetime.now(timezone.utc)
            age_hours = (current_time - article_time).total_seconds() / 3600

            # Simplified time context - just show how many hours old
            time_context = (
                "Just in: " if age_hours <= 1 else f"From {int(age_hours)} hours ago: "
            )

            # Determine target channel based on relevance
            if article["relevance"] > 0.3 and likes_channel_id:
                channel_id = likes_channel_id
            elif article["relevance"] < -0.3 and dislikes_channel_id:
                channel_id = dislikes_channel_id
            elif meh_channel_id:
                channel_id = meh_channel_id
            else:
                dprint("No appropriate channel configured for article")
                continue

            prediction = predict_reaction(article["relevance"])

            # Format relevance details
            relevance_details = article.get("relevance_details", {})
            matches = "\n".join(
                [f"✓ {m}" for m in relevance_details.get("preference_matches", [])]
            )
            mismatches = "\n".join(
                [f"✗ {m}" for m in relevance_details.get("preference_mismatches", [])]
            )
            patterns = "\n".join(
                [f"• {p}" for p in relevance_details.get("reaction_patterns", [])]
            )

            details_text = "*Relevance Analysis:*\n"
            if matches:
                details_text += "\n*Matches:*\n" + matches
            if mismatches:
                details_text += "\n*Mismatches:*\n" + mismatches
            if patterns:
                details_text += "\n*Reaction Patterns:*\n" + patterns

            blocks = [
                {
                    "type": "section",
                    "text": {
                        "text": f"{time_context}📰 *{article['title']}*\n\n"
                        f"Tags: _{', '.join(article['tags'])}_\n"
                        f"Relevance Score: {prediction}\n"
                        f"URL: {article['url']}\n\n"
                        f"{article['summary']}\n\n"
                        f"{details_text}\n"
                        f"*Reasoning:* {article['reason']}",
                    },
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "👎"},
                            "action_id": "dislike_button",
                            "value": f"dislike:{article['id']}",
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "😐"},
                            "action_id": "neutral_button",
                            "value": f"neutral:{article['id']}",
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "👍"},
                            "action_id": "like_button",
                            "value": f"like:{article['id']}",
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "🎧👍"},
                            "action_id": "like_audio_button",
                            "value": f"like_audio:{article['id']}",
                        },
                    ],
                },
            ]

            try:
                # Send message
                await app.client.chat_postMessage(
                    channel=channel_id,
                    blocks=blocks,
                    text=article["title"],  # Fallback text
                )

                # Record that this article was sent
                conn.execute(
                    """
                    INSERT INTO sent_articles (article_id, sent_time)
                    VALUES (?, CURRENT_TIMESTAMP)
                    """,
                    (article["id"],),
                )
                conn.commit()

                dprint(
                    f"Sent article: {article['title']} (relevance: {article['relevance']})"
                )
                # Add small delay between messages to avoid flooding
                await asyncio.sleep(1)
            except Exception as e:
                dprint(f"Error sending article: {e}", error=True)

        conn.close()

    # Generate and send news digests after articles
    if likes_channel_id:
        # Tech digest
        tech_digest = await generate_news_digest(
            ["tech", "ai", "hardware", "software", "science"],
            max_articles=TECH_ARTICLES,
        )
        if tech_digest:
            await send_news_digest(likes_channel_id, tech_digest)
            await asyncio.sleep(2)  # Delay between digests

        # Entertainment digest
        entertainment_digest = await generate_news_digest(
            ["entertainment", "gaming", "mobile", "shopping"],
            max_articles=ENTERTAINMENT_ARTICLES,
        )
        if entertainment_digest:
            await send_news_digest(likes_channel_id, entertainment_digest)


def predict_reaction(relevance: float) -> str:
    """Predict likely user reaction based on relevance score"""
    if relevance > 0:
        return "👍"
    elif relevance < 0:
        return "👎"
    else:
        return "😐"


@app.action("dislike_button")
@app.action("neutral_button")
@app.action("like_button")
@app.action("like_audio_button")
async def handle_reaction(ack, body, client):
    """Handle user reactions to articles"""
    await ack()

    try:
        # Parse reaction and article_id from action value
        reaction_type, article_id = body["actions"][0]["value"].split(":")
        article_id = int(article_id)

        # Map reaction to score
        score_map = {"dislike": -1, "neutral": 0, "like": 1, "like_audio": 1}
        user_score = score_map[reaction_type]

        conn = sqlite3.connect("newsy.db")
        conn.execute("PRAGMA foreign_keys = ON")

        try:
            # Verify article exists first
            cursor = conn.execute("SELECT 1 FROM articles WHERE id = ?", (article_id,))
            if not cursor.fetchone():
                raise ValueError(f"Article with id {article_id} not found")

            # Save reaction
            conn.execute(
                """
                INSERT OR REPLACE INTO article_reactions 
                (article_id, user_score, reaction_time)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (article_id, user_score),
            )
            conn.commit()

            # Update message UI by removing buttons
            await client.chat_update(
                channel=body["channel"]["id"],
                ts=body["message"]["ts"],
                blocks=[body["message"]["blocks"][0]],  # Keep only the content block
            )

            reaction_emoji = (
                "👍" if user_score == 1 else "😐" if user_score == 0 else "👎"
            )

            # For like_audio reactions, generate and send audio summary
            if reaction_type == "like_audio":
                result = await generate_audio_summary(article_id)

                if result:
                    # Upload audio file using files_upload_v2
                    await client.files_upload_v2(
                        channel=body["channel"]["id"],
                        thread_ts=body["message"]["ts"],
                        file=result["audio_file"],
                        title="Audio Summary",
                    )
                    dprint(f"Sent audio summary: {result['audio_file']}")

                    # Send text summary
                    await client.chat_postMessage(
                        channel=body["channel"]["id"],
                        thread_ts=body["message"]["ts"],
                        text=result["text_summary"],
                    )

            await client.chat_postMessage(
                channel=body["channel"]["id"],
                thread_ts=body["message"]["ts"],
                text=f"Thanks for your feedback! {reaction_emoji}",
            )
        finally:
            conn.close()
    except Exception as e:
        dprint(f"Failed to handle reaction: {e}", error=True)
        raise


@app.command("/likes")
async def handle_likes_command(ack, body):
    """Handle /likes command to set channel for likely interesting articles"""
    await ack()
    save_channel_id("likes", body["channel_id"])
    await app.client.chat_postMessage(
        channel=body["channel_id"],
        text="Set this channel for likely interesting articles",
    )


@app.command("/meh")
async def handle_meh_command(ack, body):
    """Handle /meh command to set channel for neutral articles"""
    await ack()
    save_channel_id("meh", body["channel_id"])
    await app.client.chat_postMessage(
        channel=body["channel_id"], text="Set this channel for neutral articles"
    )


@app.command("/dislikes")
async def handle_dislikes_command(ack, body):
    """Handle /dislikes command to set channel for likely uninteresting articles"""
    await ack()
    save_channel_id("dislikes", body["channel_id"])
    await app.client.chat_postMessage(
        channel=body["channel_id"],
        text="Set this channel for likely uninteresting articles",
    )


@app.command("/interest")
async def handle_interest_command(ack, body):
    """Handle /interest command to add new interest"""
    await ack()
    try:
        interest = body["text"].strip()
        current_prefs = load_user_preferences()
        gpt = GptUtil()
        new_prefs = await gpt.send_prompt(
            f"""Update these user preferences to include the new interest '{interest}' while preserving existing preferences:
            {current_prefs}"""
        )
        with open(USER_PREFERENCES_FILE, "w") as f:
            f.write(new_prefs)
        await app.client.chat_postMessage(
            channel=body["channel_id"],
            text=f"Added interest: {interest}\nUpdated preferences:\n```\n{new_prefs}\n```",
        )
    except Exception as e:
        dprint(f"Error handling interest command: {e}", error=True)
        await app.client.chat_postMessage(
            channel=body["channel_id"],
            text="Sorry, there was an error processing your command",
        )


@app.command("/disinterest")
async def handle_disinterest_command(ack, body):
    """Handle /disinterest command to add new disinterest"""
    await ack()
    try:
        disinterest = body["text"].strip()
        current_prefs = load_user_preferences()
        gpt = GptUtil()
        new_prefs = await gpt.send_prompt(
            f"""Update these user preferences to include the new disinterest '{disinterest}' while preserving existing preferences:
            {current_prefs}"""
        )
        with open(USER_PREFERENCES_FILE, "w") as f:
            f.write(new_prefs)
        await app.client.chat_postMessage(
            channel=body["channel_id"],
            text=f"Added disinterest: {disinterest}\nUpdated preferences:\n```\n{new_prefs}\n```",
        )
    except Exception as e:
        dprint(f"Error handling disinterest command: {e}", error=True)
        await app.client.chat_postMessage(
            channel=body["channel_id"],
            text="Sorry, there was an error processing your command",
        )


@app.command("/listinterests")
async def handle_listinterests_command(ack, body):
    """Handle /listinterests command to show current preferences"""
    await ack()
    try:
        current_prefs = load_user_preferences()
        await app.client.chat_postMessage(
            channel=body["channel_id"],
            text=f"Current preferences:\n```\n{current_prefs}\n```",
        )
    except Exception as e:
        dprint(f"Error handling listinterests command: {e}", error=True)
        await app.client.chat_postMessage(
            channel=body["channel_id"],
            text="Sorry, there was an error processing your command",
        )


@app.event("message")
async def handle_message(body, say):
    """Handle incoming messages and commands"""
    try:
        text = body["event"]["text"].strip().lower()
        channel_id = body["event"]["channel"]

        if text.startswith(("/interest", "/i")):
            # Add interest
            interest = text.split(" ", 1)[1]
            current_prefs = load_user_preferences()
            gpt = GptUtil()
            new_prefs = await gpt.send_prompt(
                f"""Update these user preferences to include the new interest '{interest}' while preserving existing preferences. Restructure the preferences to be more readable and easier to understand if:
                {current_prefs}"""
            )
            with open(USER_PREFERENCES_FILE, "w") as f:
                f.write(new_prefs)
            await say(f"Added interest: {interest}")
            await say(f"Updated preferences:\n```\n{new_prefs}\n```")

        elif text.startswith(("/disinterest", "/d")):
            # Add disinterest
            disinterest = text.split(" ", 1)[1]
            current_prefs = load_user_preferences()
            gpt = GptUtil()
            new_prefs = await gpt.send_prompt(
                f"""Update these user preferences to include the new disinterest '{disinterest}' while preserving existing preferences. Restructure the preferences to be more readable and easier to understand if:
                {current_prefs}"""
            )
            with open(USER_PREFERENCES_FILE, "w") as f:
                f.write(new_prefs)
            await say(f"Added disinterest: {disinterest}")
            await say(f"Updated preferences:\n```\n{new_prefs}\n```")

        elif text.startswith(("/listinterests", "/l")):
            # List interests
            current_prefs = load_user_preferences()
            await say(f"Current preferences:\n```\n{current_prefs}\n```")

        elif text.startswith("/likes"):
            # Set likes channel
            save_channel_id("likes", channel_id)
            await say("Set this channel for likely interesting articles")

        elif text.startswith("/meh"):
            # Set meh channel
            save_channel_id("meh", channel_id)
            await say("Set this channel for neutral articles")

        elif text.startswith("/dislikes"):
            # Set dislikes channel
            save_channel_id("dislikes", channel_id)
            await say("Set this channel for likely uninteresting articles")

    except Exception as e:
        dprint(f"Error handling message: {e}", error=True)
        await say("Sorry, there was an error processing your command")


async def main():
    # Initialize databases
    dprint("Starting application...")
    init_database()
    migrate_database()  # Run migrations explicitly at startup

    # Schedule batch pipeline
    async def run_scheduled_pipeline():
        while True:
            await run_batch_pipeline()
            await asyncio.sleep(BATCH_INTERVAL)

    # Start the socket mode handler
    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)

    # Run both the handler and pipeline
    await asyncio.gather(handler.start_async(), run_scheduled_pipeline())


if __name__ == "__main__":
    asyncio.run(main())
