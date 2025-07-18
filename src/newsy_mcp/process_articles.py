#!/usr/bin/env python3
"""
News Article Processor

Features:
1. Article Retrieval
   - Fetches ALL unprocessed articles from SQLite database
   - Configurable time window (default 24 hours)
   - Parallel processing of articles
   - Real-time progress tracking

2. Content Analysis
   - LLM-based article classification and tagging
   - Relevance scoring based on user preferences
   - Summary generation with word limit
   - Caches analysis results in database
   - Skips already processed articles

3. User Preferences
   - Reads preferences from user_preferences.txt
   - Matches article content against preferences
   - Scores relevance on -1 to 1 scale

4. Structured Output
   - Uses function calling for consistent formatting
   - Tags from predefined valid set
   - Word-limited summaries
   - Relevance scores with rationale
   - Real-time display of results
   - Persistent storage in database

5. Debug Support
   - Detailed logging of processing steps
   - Error tracking and reporting
   - Performance monitoring
"""

import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
import concurrent.futures
import asyncio
from utils import dprint
import faiss
import numpy as np
from memoripy import MemoryManager, JSONStorage
from memoripy.implemented_models import OpenAIChatModel, OllamaEmbeddingModel
from aibo_types import OPENAI_API_KEY
from content_util import clean_text_for_embedding
from db_init import init_database, get_db_connection

from rss_config import VALID_TAGS

# Constants
DEFAULT_HOURS = 24
MODEL = "gpt-4o-mini"
MAX_WORKERS = 10
SUMMARY_WORD_LIMIT = 150
INDEX_PATH = "faiss_index.bin"
EMBEDDING_DIMENSION = 1024  # Dimension for mxbai-embed-large

# Configure rich console
console = Console()


@dataclass
class ArticleAnalysis:
    """Container for article analysis results"""

    article_id: int
    title: str
    url: str
    tags: List[str]
    summary: str
    relevance: int
    relevance_reason: str
    relevance_details: Dict[str, List[str]]


def migrate_analysis_db() -> None:
    """Migrate the analysis database to add new columns"""
    conn = sqlite3.connect("newsy.db")
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        # Create temporary table with new schema
        conn.execute("""
            CREATE TABLE IF NOT EXISTS article_analysis_new (
                id INTEGER PRIMARY KEY,
                article_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                tags TEXT,
                summary TEXT,
                relevance FLOAT,
                relevance_reason TEXT,
                relevance_details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(article_id) REFERENCES articles(id),
                UNIQUE(article_id)
            )
        """)

        # Copy data from old table to new table
        conn.execute("""
            INSERT OR REPLACE INTO article_analysis_new 
            SELECT 
                id,
                article_id,
                title,
                url,
                tags,
                summary,
                relevance,
                relevance_reason,
                '{"preference_matches":[],"preference_mismatches":[],"reaction_patterns":[]}',
                created_at
            FROM article_analysis
        """)

        # Drop old table and rename new table
        conn.execute("DROP TABLE IF EXISTS article_analysis")
        conn.execute("ALTER TABLE article_analysis_new RENAME TO article_analysis")

        conn.commit()
        dprint("Successfully migrated article_analysis table")
    except Exception as e:
        dprint(f"Error during migration: {e}", error=True)
        conn.rollback()
    finally:
        conn.close()


def get_cached_analysis(article_id: int) -> Optional[ArticleAnalysis]:
    """Check if article has already been analyzed"""
    conn = sqlite3.connect("newsy.db")
    cursor = conn.execute(
        """
        SELECT 
            a.id,
            a.title,
            a.url,
            an.tags,
            an.summary,
            an.relevance,
            an.relevance_reason,
            an.relevance_details
        FROM articles a
        JOIN article_analysis an ON an.article_id = a.id
        WHERE a.id = ?
    """,
        (article_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return ArticleAnalysis(
            article_id=row[0],
            title=row[1],
            url=row[2],
            tags=json.loads(row[3]),
            summary=row[4],
            relevance=row[5],
            relevance_reason=row[6],
            relevance_details=json.loads(row[7]),
        )
    return None


def save_analysis(db_path: str, analysis: ArticleAnalysis) -> None:
    """Save analysis results to database"""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO article_analysis 
            (article_id, title, url, tags, summary, relevance, relevance_reason, relevance_details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                analysis.article_id,
                analysis.title,
                analysis.url,
                json.dumps(analysis.tags),
                analysis.summary,
                analysis.relevance,
                analysis.relevance_reason,
                json.dumps(analysis.relevance_details),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_user_preferences() -> str:
    """Load user preferences from file"""
    try:
        with open("user_preferences.txt", "r") as f:
            prefs = f.read()
            dprint(f"Loaded preferences: {prefs}")
            return prefs
    except Exception as e:
        console.print(f"[red]Failed to load user preferences: {e}[/red]")
        return ""


def get_recent_reactions(
    limit_likes: int = 15, limit_neutral: int = 10, limit_dislikes: int = 10
) -> Dict[int, List[Dict]]:
    """Get recent article reactions grouped by score"""
    conn = sqlite3.connect("newsy.db")
    conn.execute("PRAGMA foreign_keys = ON")
    reactions = {1: [], 0: [], -1: []}

    for score, limit in [(1, limit_likes), (0, limit_neutral), (-1, limit_dislikes)]:
        cursor = conn.execute(
            """
            SELECT 
                a.title,
                a.url,
                an.summary
            FROM article_reactions ar
            JOIN articles a ON ar.article_id = a.id
            JOIN article_analysis an ON an.article_id = a.id
            WHERE ar.user_score = ?
            ORDER BY ar.reaction_time DESC
            LIMIT ?
        """,
            (score, limit),
        )

        reactions[score] = [
            {"title": row[0], "url": row[1], "summary": row[2]}
            for row in cursor.fetchall()
        ]

    conn.close()
    return reactions


def format_reaction_history(reactions: Dict[int, List[Dict]]) -> str:
    """Format reaction history for LLM context"""
    history = []

    if reactions[1]:  # Liked articles
        history.append("\nLiked Articles:")
        for article in reactions[1]:
            history.append(f"- {article['title']}\n  Summary: {article['summary']}")

    if reactions[0]:  # Neutral articles
        history.append("\nNeutral Articles:")
        for article in reactions[0]:
            history.append(f"- {article['title']}\n  Summary: {article['summary']}")

    if reactions[-1]:  # Disliked articles
        history.append("\nDisliked Articles:")
        for article in reactions[-1]:
            history.append(f"- {article['title']}\n  Summary: {article['summary']}")

    return "\n".join(history)


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def analyze_article(
    client: OpenAI, content: str, preferences: str, reaction_history: str
) -> Optional[ArticleAnalysis]:
    """Analyze article content using LLM with function calling"""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "process_article",
                "description": "Process article content and generate structured analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tags": {
                            "type": "array",
                            "items": {"type": "string", "enum": VALID_TAGS},
                            "description": "List of relevant tags from the valid set",
                        },
                        "summary": {
                            "type": "string",
                            "description": f"Brief summary under {SUMMARY_WORD_LIMIT} words",
                        },
                        "relevance": {
                            "type": "integer",
                            "enum": [-1, 0, 1],
                            "description": "Relevance score based on user preferences",
                        },
                        "relevance_reason": {
                            "type": "string",
                            "description": "Detailed explanation of relevance score, including specific matches or mismatches with preferences and reaction history",
                        },
                        "relevance_details": {
                            "type": "object",
                            "description": "Detailed analysis of how the score was determined",
                            "properties": {
                                "preference_matches": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Specific matches with user preferences",
                                },
                                "preference_mismatches": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Specific mismatches with user preferences",
                                },
                                "reaction_patterns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Relevant patterns from reaction history",
                                },
                            },
                        },
                    },
                    "required": [
                        "tags",
                        "summary",
                        "relevance",
                        "relevance_reason",
                        "relevance_details",
                    ],
                },
            },
        }
    ]

    try:
        dprint(f"Sending request to OpenAI API: {content[:50]}")
        system_prompt = f"""Analyze this article based on:

1. User Preferences:
{preferences}

2. Recent Reaction History:
{reaction_history}

Provide detailed justification for the relevance score by:
- Identifying specific matches and mismatches with user preferences
- Noting any patterns from reaction history
- Explaining how these factors influenced the final score
- Being explicit about which aspects were most important

Use both the explicit preferences and reaction patterns to determine relevance."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "process_article"}},
        )

        if response.choices[0].message.tool_calls:
            result = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )
            dprint("Successfully received analysis from API")
            return result

        dprint("No analysis received from API", error=True)
        return None

    except Exception as e:
        console.print(f"[red]Failed to analyze article: {e}[/red]")
        return None


def get_recent_articles(hours: int = DEFAULT_HOURS) -> List[Dict]:
    """Get ALL unprocessed articles from last N hours from database"""
    conn = sqlite3.connect("newsy.db")
    cutoff = datetime.now() - timedelta(hours=hours)

    query = """
    SELECT 
        a.id,
        a.title, 
        a.url, 
        a.content_html2text 
    FROM articles a
    LEFT JOIN article_analysis aa ON a.id = aa.article_id
    WHERE a.created_at > ? 
    AND aa.id IS NULL  -- Only get articles without analysis
    ORDER BY a.created_at DESC
    """

    try:
        cursor = conn.execute(query, (cutoff.isoformat(),))
        articles = [
            {"id": row[0], "title": row[1], "url": row[2], "content": row[3]}
            for row in cursor.fetchall()
        ]

        dprint(f"Found {len(articles)} unprocessed articles from last {hours} hours")

        return articles
    finally:
        conn.close()


async def process_articles(hours: int = DEFAULT_HOURS) -> List[ArticleAnalysis]:
    """Main function to process recent articles"""
    console.print("\n[bold cyan]Starting article analysis...[/bold cyan]")

    # Initialize both databases
    dprint("Initializing databases...")
    init_analysis_db()

    client = OpenAI()
    preferences = load_user_preferences()

    # Get reaction history
    dprint("Fetching reaction history...")
    reactions = get_recent_reactions()
    reaction_history = format_reaction_history(reactions)

    # Get ALL unprocessed articles
    articles = get_recent_articles(hours)
    total_articles = len(articles)

    if total_articles == 0:
        console.print("[green]No unprocessed articles found![/green]")
        return []

    console.print(
        f"[cyan]Found {total_articles} unprocessed articles to analyze[/cyan]"
    )

    # Initialize memory manager for embeddings
    memory_manager = init_memory_manager()

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        main_task = progress.add_task(
            "[cyan]Processing articles...", total=total_articles
        )

        # Process articles in parallel with limited workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_article = {}

            for article in articles:
                cached = get_cached_analysis(article["id"])
                if cached:
                    results.append(cached)
                    progress.advance(main_task)
                    continue

                future = executor.submit(
                    analyze_article,
                    client,
                    article["content"],
                    preferences,
                    reaction_history,
                )
                future_to_article[future] = article

            for future in concurrent.futures.as_completed(future_to_article):
                article = future_to_article[future]
                analysis = future.result()

                if analysis:
                    result = ArticleAnalysis(
                        article_id=article["id"],
                        title=article["title"],
                        url=article["url"],
                        tags=analysis["tags"],
                        summary=analysis["summary"],
                        relevance=analysis["relevance"],
                        relevance_reason=analysis["relevance_reason"],
                        relevance_details=analysis["relevance_details"],
                    )
                    results.append(result)

                    # Save analysis
                    save_analysis("newsy.db", result)

                    # Generate and save embedding
                    try:
                        # Clean and validate summary
                        cleaned_summary = clean_text_for_embedding(analysis["summary"])
                        if not cleaned_summary or len(cleaned_summary.split()) < 10:
                            dprint(
                                f"Summary too short for article {article['id']}",
                                error=True,
                            )
                            continue

                        # Generate embedding
                        embedding = memory_manager.get_embedding(cleaned_summary)
                        if embedding is None:
                            dprint(
                                f"Failed to generate embedding for article {article['id']}",
                                error=True,
                            )
                            continue

                        embedding_array = np.array(embedding).astype("float32")
                        if embedding_array.shape[-1] != EMBEDDING_DIMENSION:
                            dprint(
                                f"Wrong embedding dimension for article {article['id']}: {embedding_array.shape[-1]} != {EMBEDDING_DIMENSION}",
                                error=True,
                            )
                            continue

                        # Store embedding in database
                        conn = get_db_connection("newsy.db")
                        try:
                            conn.execute(
                                """
                                INSERT INTO article_embeddings 
                                (article_id, embedding)
                                VALUES (?, ?)
                                """,
                                (article["id"], embedding_array.tobytes()),
                            )
                            conn.commit()

                            # Update FAISS index
                            if len(embedding_array.shape) == 1:
                                embedding_array = embedding_array.reshape(1, -1)
                            update_faiss_index({article["id"]: embedding_array})

                            dprint(f"Saved embedding for article {article['id']}")
                        finally:
                            conn.close()
                    except Exception as e:
                        dprint(
                            f"Error saving embedding for article {article['id']}: {e}",
                            error=True,
                        )

                    # Display result as it's processed
                    # display_result(result)

                progress.advance(main_task)

    console.print(
        f"\n[green]Successfully processed {len(results)}/{total_articles} articles![/green]"
    )
    return results


def display_result(result: ArticleAnalysis) -> None:
    """Display a single result as it's processed"""
    console.print(
        Panel(
            f"[cyan]Title:[/cyan] {result.title}\n"
            f"[cyan]URL:[/cyan] {result.url}\n"
            f"[cyan]Tags:[/cyan] {', '.join(result.tags)}\n"
            f"[cyan]Summary:[/cyan] {result.summary}\n"
            f"[cyan]Relevance:[/cyan] {result.relevance}\n"
            f"[cyan]Reason:[/cyan] {result.relevance_reason}",
            title="New Article Analysis",
            border_style="green",
        )
    )


def display_results(results: List[ArticleAnalysis]) -> None:
    """Display final results summary in a rich formatted table"""
    table = Table(title="Article Analysis Results")
    table.add_column("Title", style="cyan", no_wrap=True)
    table.add_column("Tags", style="magenta")
    table.add_column("Summary", style="green")
    table.add_column("Relevance", justify="center", style="blue")

    for result in results:
        relevance_style = (
            "green"
            if result.relevance > 0
            else "red"
            if result.relevance < 0
            else "yellow"
        )
        table.add_row(
            result.title[:50] + "..." if len(result.title) > 50 else result.title,
            ", ".join(result.tags),
            result.summary[:100] + "..."
            if len(result.summary) > 100
            else result.summary,
            f"[{relevance_style}]{result.relevance}[/{relevance_style}]",
        )

    console.print(table)


def init_faiss_index() -> None:
    """Initialize or load the FAISS index"""
    conn = sqlite3.connect("newsy.db")
    try:
        # Create embeddings table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS article_embeddings (
                id INTEGER PRIMARY KEY,
                article_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(article_id) REFERENCES articles(id),
                UNIQUE(article_id)
            )
        """)

        # Get all embeddings
        cursor = conn.execute("SELECT id, embedding FROM article_embeddings")
        rows = cursor.fetchall()

        # Create empty index
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)

        if rows:
            # Process embeddings one at a time to validate dimensions
            valid_embeddings = []
            for row in rows:
                try:
                    embedding = np.frombuffer(row[1], dtype=np.float32)
                    if len(embedding) == EMBEDDING_DIMENSION:
                        embedding = embedding.reshape(1, -1)
                        valid_embeddings.append(embedding)
                    else:
                        dprint(
                            f"Skipping embedding for id {row[0]} - wrong dimension {len(embedding)}",
                            error=True,
                        )
                except Exception as e:
                    dprint(
                        f"Error processing embedding for id {row[0]}: {e}", error=True
                    )
                    continue

            if valid_embeddings:
                embeddings_array = np.vstack(valid_embeddings)
                index.add(embeddings_array)
                dprint(f"FAISS index created with {len(valid_embeddings)} vectors")
            else:
                dprint("No valid embeddings found - created empty index")

        # Write index to disk
        faiss.write_index(index, INDEX_PATH)
        return index

    except Exception as e:
        dprint(f"Error initializing FAISS index: {e}", error=True)
        # Create and save empty index as fallback
        empty_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        faiss.write_index(empty_index, INDEX_PATH)
        dprint("Created empty fallback index")
        return empty_index
    finally:
        conn.close()


def update_faiss_index(new_embeddings: Dict[int, np.ndarray]) -> None:
    """Update FAISS index with new embeddings"""
    try:
        # Load existing index
        index = faiss.read_index(INDEX_PATH)

        # Validate and prepare vectors
        valid_vectors = []
        for article_id, embedding in new_embeddings.items():
            if embedding.shape[-1] == EMBEDDING_DIMENSION:
                # Ensure 2D shape
                vector = embedding.reshape(1, -1)
                valid_vectors.append(vector)
            else:
                dprint(
                    f"Skipping invalid embedding for article {article_id}", error=True
                )

        if valid_vectors:
            vectors = np.vstack(valid_vectors)
            index.add(vectors)
            faiss.write_index(index, INDEX_PATH)
            dprint(f"Added {len(valid_vectors)} vectors to FAISS index")
        else:
            dprint("No valid vectors to add to index")

    except Exception as e:
        dprint(f"Error updating FAISS index: {e}", error=True)


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


def init_analysis_db() -> None:
    """Initialize the analysis database"""
    init_database("newsy.db")
    init_faiss_index()


if __name__ == "__main__":
    console.print("[bold cyan]News Article Processor[/bold cyan]")
    asyncio.run(process_articles())
