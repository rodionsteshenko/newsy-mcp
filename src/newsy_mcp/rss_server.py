"""
RSS MCP Server

This module provides an MCP API for the RSS server functionality:
1. User preferences management
2. RSS feed management
3. Article retrieval and search
4. Database query capabilities
"""

import json
import os
import sqlite3
import sys
import base64
import tempfile
import requests
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP, Image
from rich.console import Console

# Use relative imports when running with --directory flag
from db_init import get_db_connection
from rss_config import RSS_FEEDS, ALL_RSS_FEEDS
from vector_store import VectorStore

# Initialize console for rich output, redirected to stderr
console = Console(file=sys.stderr)

# Initialize MCP server
mcp = FastMCP("newsy-rss")

# Database path
DB_PATH = "newsy.db"

# User preferences path
USER_PREFS_PATH = "user_preferences.json"

# Initialize database connection
def get_db():
    """Get a database connection."""
    return get_db_connection(DB_PATH)

# Initialize vector store for similarity search
def get_vector_store():
    """Get a vector store instance."""
    # This would typically require an OpenAI API key from config
    # For now, we'll just return None - implement based on your needs
    return None

#
# User Preferences API
#

def _load_user_preferences() -> Dict[str, Any]:
    """Load user preferences from database."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get topics liked
            cursor.execute(
                "SELECT key, value FROM user_preferences WHERE category = 'topic_liked'"
            )
            topics_liked = [row[1] for row in cursor.fetchall()]
            
            # Get topics to avoid
            cursor.execute(
                "SELECT key, value FROM user_preferences WHERE category = 'topic_avoided'"
            )
            topics_to_avoid = [row[1] for row in cursor.fetchall()]
            
            # Get content preferences
            cursor.execute(
                "SELECT key, value FROM user_preferences WHERE category = 'content_preference'"
            )
            content_preferences = [row[1] for row in cursor.fetchall()]
            
            return {
                "topics_liked": topics_liked,
                "topics_to_avoid": topics_to_avoid,
                "content_preferences": content_preferences
            }
    except Exception as e:
        console.print(f"Error loading user preferences from database: {str(e)}")
        
        # Fall back to loading from file if available
        try:
            if os.path.exists(USER_PREFS_PATH):
                with open(USER_PREFS_PATH, "r") as f:
                    return json.load(f)
        except Exception:
            pass
            
        return {"topics_liked": [], "topics_to_avoid": [], "content_preferences": []}

def _save_user_preferences(preferences: Dict[str, Any]) -> bool:
    """Save user preferences to database."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Start a transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Save topics liked
            if "topics_liked" in preferences:
                # Delete existing topics liked
                cursor.execute("DELETE FROM user_preferences WHERE category = 'topic_liked'")
                
                # Insert new topics liked
                for i, topic in enumerate(preferences["topics_liked"]):
                    cursor.execute(
                        """
                        INSERT INTO user_preferences (key, value, category)
                        VALUES (?, ?, ?)
                        """,
                        (f"like_{i}", topic, "topic_liked")
                    )
            
            # Save topics to avoid
            if "topics_to_avoid" in preferences:
                # Delete existing topics to avoid
                cursor.execute("DELETE FROM user_preferences WHERE category = 'topic_avoided'")
                
                # Insert new topics to avoid
                for i, topic in enumerate(preferences["topics_to_avoid"]):
                    cursor.execute(
                        """
                        INSERT INTO user_preferences (key, value, category)
                        VALUES (?, ?, ?)
                        """,
                        (f"avoid_{i}", topic, "topic_avoided")
                    )
            
            # Save content preferences
            if "content_preferences" in preferences:
                # Delete existing content preferences
                cursor.execute("DELETE FROM user_preferences WHERE category = 'content_preference'")
                
                # Insert new content preferences
                for i, pref in enumerate(preferences["content_preferences"]):
                    cursor.execute(
                        """
                        INSERT INTO user_preferences (key, value, category)
                        VALUES (?, ?, ?)
                        """,
                        (f"content_pref_{i}", pref, "content_preference")
                    )
            
            # Commit the transaction
            conn.commit()
            return True
            
    except Exception as e:
        console.print(f"Error saving user preferences to database: {str(e)}")
        
        # Fall back to saving to file
        try:
            with open(USER_PREFS_PATH, "w") as f:
                json.dump(preferences, f, indent=2)
            return True
        except Exception:
            return False

@mcp.tool()
def get_user_preferences() -> Dict[str, Any]:
    """
    Get the current user preferences.

    Returns:
        Dict[str, Any]: Current user preference settings including liked topics,
        topics to avoid, and content preferences.
    """
    try:
        return _load_user_preferences()
    except Exception as e:
        return {"error": f"Error getting user preferences: {str(e)}"}

@mcp.tool()
def set_user_preferences(preferences: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the user preferences.
    
    Args:
        preferences (Dict[str, Any]): User preferences to update. Can include 'topics_liked',
        'topics_to_avoid', and/or 'content_preferences' lists.
        
    Returns:
        Dict[str, Any]: Result of the update operation with updated preferences.
    """
    try:
        # Load current preferences
        current_prefs = _load_user_preferences()
        
        # Update with new preferences
        for key, value in preferences.items():
            if key in current_prefs:
                current_prefs[key] = value
        
        # Save updated preferences
        if _save_user_preferences(current_prefs):
            return {
                "success": True, 
                "message": "User preferences updated successfully", 
                "preferences": current_prefs
            }
        else:
            return {"error": "Failed to save user preferences"}
    except Exception as e:
        return {"error": f"Error updating user preferences: {str(e)}"}

#
# RSS Feed Management API
#

def _load_active_feeds() -> List[str]:
    """Load active RSS feeds from database."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT url FROM rss_feeds 
                WHERE is_active = 1
                ORDER BY created_at
                """
            )
            return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        console.print(f"Error loading active feeds from database: {str(e)}")
        # Fall back to config file
        return RSS_FEEDS.copy()

def _load_all_feeds() -> List[str]:
    """Load all available RSS feeds from database."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT url FROM rss_feeds 
                ORDER BY created_at
                """
            )
            feeds = [row[0] for row in cursor.fetchall()]
            
            if feeds:
                return feeds
            else:
                # If database is empty, return from config
                return ALL_RSS_FEEDS.copy()
    except Exception as e:
        console.print(f"Error loading all feeds from database: {str(e)}")
        # Fall back to config file
        return ALL_RSS_FEEDS.copy()

def _save_active_feeds(feeds: List[str]) -> bool:
    """Save active RSS feeds to database."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Start transaction
            conn.execute("BEGIN TRANSACTION")
            
            # First, deactivate all feeds
            cursor.execute("UPDATE rss_feeds SET is_active = 0")
            
            # Then activate the specified feeds
            for feed_url in feeds:
                # Check if feed exists in the database
                cursor.execute(
                    """
                    SELECT id FROM rss_feeds WHERE url = ?
                    """, 
                    (feed_url,)
                )
                
                if cursor.fetchone():
                    # Activate existing feed
                    cursor.execute(
                        """
                        UPDATE rss_feeds 
                        SET is_active = 1 
                        WHERE url = ?
                        """,
                        (feed_url,)
                    )
                else:
                    # Insert new feed with active state
                    domain = urlparse(feed_url).netloc
                    if domain.startswith('www.'):
                        domain = domain[4:]
                        
                    cursor.execute(
                        """
                        INSERT INTO rss_feeds (url, name, is_active, category)
                        VALUES (?, ?, 1, 'new')
                        """,
                        (feed_url, domain)
                    )
            
            # Commit changes
            conn.commit()
            return True
    except Exception as e:
        console.print(f"Error saving active feeds to database: {str(e)}")
        return False

@mcp.tool()
def get_active_feeds() -> Dict[str, Any]:
    """
    Get the list of currently active RSS feeds.

    Returns:
        Dict[str, Any]: List of active RSS feed URLs.
    """
    try:
        return {"feeds": _load_active_feeds()}
    except Exception as e:
        return {"error": f"Error getting active feeds: {str(e)}"}

@mcp.tool()
def get_all_feeds() -> Dict[str, Any]:
    """
    Get all available RSS feeds (active and inactive).

    Returns:
        Dict[str, Any]: List of all available RSS feed URLs with metadata.
    """
    try:
        # Try to get detailed feed information from database
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, url, name, category, is_active, last_fetch, created_at
                    FROM rss_feeds 
                    ORDER BY name, url
                    """
                )
                
                all_feeds_info = []
                active_feeds_urls = []
                inactive_feeds_urls = []
                
                for row in cursor.fetchall():
                    feed_info = {
                        "id": row[0],
                        "url": row[1],
                        "name": row[2] or urlparse(row[1]).netloc,
                        "category": row[3],
                        "is_active": bool(row[4]),
                        "last_fetch": row[5],
                        "created_at": row[6]
                    }
                    
                    all_feeds_info.append(feed_info)
                    if feed_info["is_active"]:
                        active_feeds_urls.append(feed_info["url"])
                    else:
                        inactive_feeds_urls.append(feed_info["url"])
                
                # If we found feeds in the database, return the detailed information
                if all_feeds_info:
                    return {
                        "feeds": all_feeds_info,
                        "active_feeds": active_feeds_urls,
                        "inactive_feeds": inactive_feeds_urls,
                        "total_count": len(all_feeds_info),
                        "active_count": len(active_feeds_urls)
                    }
        
        except Exception as e:
            console.print(f"Error getting detailed feed info from database: {str(e)}")
        
        # Fall back to simple lists from config if database query fails
        active_feeds = _load_active_feeds()
        all_feeds = _load_all_feeds()
        
        return {
            "active_feeds": active_feeds,
            "all_feeds": all_feeds,
            "inactive_feeds": [feed for feed in all_feeds if feed not in active_feeds],
            "total_count": len(all_feeds),
            "active_count": len(active_feeds)
        }
        
    except Exception as e:
        return {"error": f"Error getting all feeds: {str(e)}"}

@mcp.tool()
def get_feed_by_url(feed_url: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific RSS feed.
    
    Args:
        feed_url (str): The URL of the feed to retrieve.
        
    Returns:
        Dict[str, Any]: Feed information including metadata and status.
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, url, name, category, is_active, last_fetch, created_at
                FROM rss_feeds 
                WHERE url = ?
                """,
                (feed_url,)
            )
            
            row = cursor.fetchone()
            if not row:
                return {"error": f"Feed not found: {feed_url}"}
            
            # Get the latest articles from this feed
            cursor.execute(
                """
                SELECT a.id, a.title, a.timestamp
                FROM articles a
                WHERE a.feed_domain = ?
                ORDER BY a.timestamp DESC
                LIMIT 10
                """,
                (urlparse(feed_url).netloc,)
            )
            
            recent_articles = [
                {"id": article[0], "title": article[1], "timestamp": article[2]} 
                for article in cursor.fetchall()
            ]
            
            return {
                "id": row[0],
                "url": row[1],
                "name": row[2] or urlparse(row[1]).netloc,
                "category": row[3],
                "is_active": bool(row[4]),
                "last_fetch": row[5],
                "created_at": row[6],
                "recent_articles": recent_articles,
                "article_count": len(recent_articles)
            }
    except Exception as e:
        return {"error": f"Error retrieving feed information: {str(e)}"}

@mcp.tool()
def set_active_feeds(feeds: List[str]) -> Dict[str, Any]:
    """
    Update the list of active RSS feeds.
    
    Args:
        feeds (List[str]): List of RSS feed URLs to activate.
        
    Returns:
        Dict[str, Any]: Result of the update operation.
    """
    try:
        if _save_active_feeds(feeds):
            return {
                "success": True, 
                "message": "Active feeds updated successfully", 
                "feeds": feeds
            }
        else:
            return {"error": "Failed to update active feeds"}
    except Exception as e:
        return {"error": f"Error updating active feeds: {str(e)}"}

#
# Article Retrieval API
#

@mcp.tool()
def get_recent_articles(limit: int = 20, offset: int = 0, exclude_displayed: bool = False) -> Dict[str, Any]:
    """
    Get a list of recently downloaded articles.
    
    Args:
        limit (int): Maximum number of articles to return (default: 20)
        offset (int): Number of articles to skip (default: 0)
        
    Returns:
        Dict[str, Any]: List of articles with their metadata.
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            if exclude_displayed:
                cursor.execute(
                    """
                    SELECT a.id, a.title, a.url, a.feed_domain, a.timestamp, 
                           a.content_html2text, an.tags, an.summary, an.relevance
                    FROM articles a
                    LEFT JOIN article_analysis an ON a.id = an.article_id
                    LEFT JOIN displayed_articles da ON a.id = da.article_id
                    WHERE da.article_id IS NULL
                    ORDER BY a.timestamp DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset)
                )
            else:
                cursor.execute(
                    """
                    SELECT a.id, a.title, a.url, a.feed_domain, a.timestamp, 
                           a.content_html2text, an.tags, an.summary, an.relevance
                    FROM articles a
                    LEFT JOIN article_analysis an ON a.id = an.article_id
                    ORDER BY a.timestamp DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset)
                )
            
            articles = []
            for row in cursor.fetchall():
                article = {
                    "id": row[0],
                    "title": row[1],
                    "url": row[2],
                    "feed_domain": row[3],
                    "timestamp": row[4],
                    "content": row[5][:500] + "..." if row[5] and len(row[5]) > 500 else row[5],
                }
                
                # Add analysis data if available
                if row[6]:  # tags
                    article["tags"] = row[6].split(",") if row[6] else []
                if row[7]:  # summary
                    article["summary"] = row[7]
                if row[8]:  # relevance
                    article["relevance"] = row[8]
                
                articles.append(article)
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM articles")
            total_count = cursor.fetchone()[0]
            
            return {
                "articles": articles,
                "total_count": total_count,
                "offset": offset,
                "limit": limit
            }
    except Exception as e:
        return {"error": f"Error retrieving articles: {str(e)}"}

@mcp.tool()
def get_article_by_id(article_id: int) -> Dict[str, Any]:
    """
    Get a specific article by ID.
    
    Args:
        article_id (int): ID of the article to retrieve.
        
    Returns:
        Dict[str, Any]: Article data with full content.
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT a.id, a.title, a.url, a.feed_domain, a.timestamp, 
                       a.content_markdown, an.tags, an.summary, an.relevance
                FROM articles a
                LEFT JOIN article_analysis an ON a.id = an.article_id
                WHERE a.id = ?
                """,
                (article_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                return {"error": f"Article not found with ID: {article_id}"}
            
            article = {
                "id": row[0],
                "title": row[1],
                "url": row[2],
                "feed_domain": row[3],
                "timestamp": row[4],
                "content": row[5]
            }
            
            # Add analysis data if available
            if row[6]:  # tags
                article["tags"] = row[6].split(",") if row[6] else []
            if row[7]:  # summary
                article["summary"] = row[7]
            if row[8]:  # relevance
                article["relevance"] = row[8]
            
            # Get images for the article
            cursor.execute(
                """
                SELECT url, alt_text, caption
                FROM article_images
                WHERE article_id = ?
                """,
                (article_id,)
            )
            
            images = []
            for img_row in cursor.fetchall():
                images.append({
                    "url": img_row[0],
                    "alt_text": img_row[1],
                    "caption": img_row[2]
                })
            
            article["images"] = images
            
            return article
    except Exception as e:
        return {"error": f"Error retrieving article: {str(e)}"}

@mcp.tool()
def search_articles(query: str, limit: int = 10, exclude_displayed: bool = False) -> Dict[str, Any]:
    """
    Search articles using keyword matching.
    
    Args:
        query (str): Search query
        limit (int): Maximum number of results to return (default: 10)
        
    Returns:
        Dict[str, Any]: List of articles matching the query.
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            search_term = f"%{query}%"
            
            if exclude_displayed:
                cursor.execute(
                    """
                    SELECT a.id, a.title, a.url, a.feed_domain, a.timestamp, 
                           a.content_html2text, an.tags, an.summary, an.relevance
                    FROM articles a
                    LEFT JOIN article_analysis an ON a.id = an.article_id
                    LEFT JOIN displayed_articles da ON a.id = da.article_id
                    WHERE (a.title LIKE ? OR a.content_html2text LIKE ?)
                    AND da.article_id IS NULL
                    ORDER BY a.timestamp DESC
                    LIMIT ?
                    """,
                    (search_term, search_term, limit)
                )
            else:
                cursor.execute(
                    """
                    SELECT a.id, a.title, a.url, a.feed_domain, a.timestamp, 
                           a.content_html2text, an.tags, an.summary, an.relevance
                    FROM articles a
                    LEFT JOIN article_analysis an ON a.id = an.article_id
                    WHERE a.title LIKE ? OR a.content_html2text LIKE ?
                    ORDER BY a.timestamp DESC
                    LIMIT ?
                    """,
                    (search_term, search_term, limit)
                )
            
            articles = []
            for row in cursor.fetchall():
                article = {
                    "id": row[0],
                    "title": row[1],
                    "url": row[2],
                    "feed_domain": row[3],
                    "timestamp": row[4],
                    "content_snippet": row[5][:200] + "..." if row[5] and len(row[5]) > 200 else row[5],
                }
                
                # Add analysis data if available
                if row[6]:  # tags
                    article["tags"] = row[6].split(",") if row[6] else []
                if row[7]:  # summary
                    article["summary"] = row[7]
                if row[8]:  # relevance
                    article["relevance"] = row[8]
                
                articles.append(article)
            
            return {
                "query": query,
                "articles": articles,
                "count": len(articles)
            }
    except Exception as e:
        return {"error": f"Error searching articles: {str(e)}"}

@mcp.tool()
def search_articles_by_vector(query: str, limit: int = 10, exclude_displayed: bool = False) -> Dict[str, Any]:
    """
    Search articles using vector similarity (semantic search).
    
    Args:
        query (str): Search query
        limit (int): Maximum number of results to return (default: 10)
        
    Returns:
        Dict[str, Any]: List of articles semantically similar to the query.
    """
    try:
        vector_store = get_vector_store()
        if not vector_store:
            return {"error": "Vector store is not initialized"}
            
        # Perform similarity search with higher similarity threshold
        results = vector_store.similarity_search(query, k=limit, similarity_threshold=0.9)
        
        articles = []
        displayed_articles = set()
        
        # Get list of displayed article IDs if needed
        if exclude_displayed:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT article_id FROM displayed_articles"
                )
                displayed_articles = {row[0] for row in cursor.fetchall()}
        
        for result in results:
            article_id = result.get("metadata", {}).get("article_id")
            if article_id and (not exclude_displayed or article_id not in displayed_articles):
                # Fetch full article data
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT a.id, a.title, a.url, a.feed_domain, a.timestamp, 
                               a.content_html2text, an.tags, an.summary
                        FROM articles a
                        LEFT JOIN article_analysis an ON a.id = an.article_id
                        WHERE a.id = ?
                        """,
                        (article_id,)
                    )
                    
                    row = cursor.fetchone()
                    if row:
                        article = {
                            "id": row[0],
                            "title": row[1],
                            "url": row[2],
                            "feed_domain": row[3],
                            "timestamp": row[4],
                            "content_snippet": row[5][:200] + "..." if row[5] and len(row[5]) > 200 else row[5],
                            "similarity_score": result.get("distance"),
                            "rank": result.get("rank")
                        }
                        
                        # Add analysis data if available
                        if row[6]:  # tags
                            article["tags"] = row[6].split(",") if row[6] else []
                        if row[7]:  # summary
                            article["summary"] = row[7]
                        
                        articles.append(article)
        
        return {
            "query": query,
            "articles": articles,
            "count": len(articles)
        }
    except Exception as e:
        return {
            "error": f"Error performing vector search: {str(e)}",
            "fallback": search_articles(query, limit)
        }

#
# Database Statistics API
#

@mcp.tool()
def get_database_stats() -> Dict[str, Any]:
    """
    Get statistics about the RSS database.
    
    Returns:
        Dict[str, Any]: Database statistics including article counts, feed counts, etc.
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Total articles
            cursor.execute("SELECT COUNT(*) FROM articles")
            total_articles = cursor.fetchone()[0]
            
            # Articles by feed domain
            cursor.execute("""
                SELECT feed_domain, COUNT(*) as count 
                FROM articles 
                GROUP BY feed_domain 
                ORDER BY count DESC
            """)
            feeds = [{"domain": row[0], "count": row[1]} for row in cursor.fetchall()]
            
            # Articles with images
            cursor.execute("""
                SELECT COUNT(DISTINCT article_id) 
                FROM article_images
            """)
            articles_with_images = cursor.fetchone()[0]
            
            # Recent articles
            cursor.execute("""
                SELECT COUNT(*) 
                FROM articles 
                WHERE timestamp >= datetime('now', '-1 day')
            """)
            recent_articles = cursor.fetchone()[0]
            
            return {
                "total_articles": total_articles,
                "articles_with_images": articles_with_images,
                "articles_last_24h": recent_articles,
                "active_feeds": len(_load_active_feeds()),
                "available_feeds": len(_load_all_feeds()),
                "feed_statistics": feeds
            }
    except Exception as e:
        return {"error": f"Error retrieving database stats: {str(e)}"}

@mcp.tool()
def download_image_from_url(url: str) -> Tuple[Dict[str, Any], Image]:
    """
    Download an image from a URL and return it as an Image object.
    
    Args:
        url (str): URL of the image to download
        
    Returns:
        Tuple[Dict[str, Any], Image]: Metadata about the image and the image itself
    """
    try:
        console.print(f"[bold blue]Downloading image from URL: {url}[/]")
        
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return ({"error": "Invalid URL format"}, Image(path=""))
        
        # Download the image with a timeout
        response = requests.get(url, timeout=10, stream=True)
        if response.status_code != 200:
            return ({"error": f"Failed to download image: HTTP {response.status_code}"}, Image(path=""))
            
        # Get content type to verify it's an image
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            return ({"error": f"URL does not contain an image (Content-Type: {content_type})"}, Image(path=""))
        
        # Determine file extension based on content type
        extension = "png"  # Default extension
        if "jpeg" in content_type or "jpg" in content_type:
            extension = "jpg"
        elif "gif" in content_type:
            extension = "gif"
        elif "png" in content_type:
            extension = "png"
        elif "webp" in content_type:
            extension = "webp"
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}")
        temp_path = temp_file.name
        
        # Write image data to temporary file
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        console.print(f"[bold green]✓ Image downloaded successfully to: {temp_path}[/]")
        
        # Get image metadata
        size_bytes = os.path.getsize(temp_path)
        
        return ({
            "url": url,
            "filename": os.path.basename(parsed_url.path) or f"image.{extension}",
            "content_type": content_type,
            "size_bytes": size_bytes,
            "success": True
        }, Image(path=temp_path))
        
    except Exception as e:
        error_msg = f"[bold red]ERROR downloading image: {str(e)}[/]"
        console.print(error_msg)
        return ({"error": f"Error downloading image: {str(e)}"}, Image(path=""))

@mcp.tool()
def mark_article_displayed(article_id: int, session_id: str = None) -> Dict[str, Any]:
    """
    Mark an article as displayed by the agent.
    
    Args:
        article_id (int): ID of the article to mark as displayed
        session_id (str, optional): Identifier for the agent session
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Verify article exists
            cursor.execute(
                "SELECT id FROM articles WHERE id = ?",
                (article_id,)
            )
            if not cursor.fetchone():
                return {"error": f"Article not found with ID: {article_id}"}
                
            # Insert or update displayed status
            cursor.execute(
                """
                INSERT INTO displayed_articles (article_id, status, agent_session_id)
                VALUES (?, 'displayed', ?)
                ON CONFLICT(article_id) DO UPDATE SET 
                    status = 'displayed',
                    display_time = CURRENT_TIMESTAMP,
                    agent_session_id = ?
                """,
                (article_id, session_id, session_id)
            )
            
            conn.commit()
            
            return {
                "success": True,
                "message": f"Article {article_id} marked as displayed",
                "article_id": article_id,
                "status": "displayed"
            }
    except Exception as e:
        return {"error": f"Error marking article as displayed: {str(e)}"}

@mcp.tool()
def mark_article_ignored(article_id: int, session_id: str = None) -> Dict[str, Any]:
    """
    Mark an article as ignored by the agent.
    
    Args:
        article_id (int): ID of the article to mark as ignored
        session_id (str, optional): Identifier for the agent session
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Verify article exists
            cursor.execute(
                "SELECT id FROM articles WHERE id = ?",
                (article_id,)
            )
            if not cursor.fetchone():
                return {"error": f"Article not found with ID: {article_id}"}
                
            # Insert or update ignored status
            cursor.execute(
                """
                INSERT INTO displayed_articles (article_id, status, agent_session_id)
                VALUES (?, 'ignored', ?)
                ON CONFLICT(article_id) DO UPDATE SET 
                    status = 'ignored',
                    display_time = CURRENT_TIMESTAMP,
                    agent_session_id = ?
                """,
                (article_id, session_id, session_id)
            )
            
            conn.commit()
            
            return {
                "success": True,
                "message": f"Article {article_id} marked as ignored",
                "article_id": article_id,
                "status": "ignored"
            }
    except Exception as e:
        return {"error": f"Error marking article as ignored: {str(e)}"}

@mcp.tool()
def get_displayed_article_ids() -> Dict[str, Any]:
    """
    Get a list of article IDs that have been displayed or ignored.
    
    Returns:
        Dict[str, Any]: Lists of displayed and ignored article IDs
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get displayed articles
            cursor.execute(
                "SELECT article_id FROM displayed_articles WHERE status = 'displayed'"
            )
            displayed_ids = [row[0] for row in cursor.fetchall()]
            
            # Get ignored articles
            cursor.execute(
                "SELECT article_id FROM displayed_articles WHERE status = 'ignored'"
            )
            ignored_ids = [row[0] for row in cursor.fetchall()]
            
            return {
                "displayed_ids": displayed_ids,
                "ignored_ids": ignored_ids,
                "total_displayed": len(displayed_ids),
                "total_ignored": len(ignored_ids)
            }
    except Exception as e:
        return {"error": f"Error getting displayed article IDs: {str(e)}"}

@mcp.tool()
def reset_displayed_status() -> Dict[str, Any]:
    """
    Reset all displayed and ignored article statuses.
    This allows articles to be shown again that were previously displayed or ignored.
    
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM displayed_articles")
            conn.commit()
            
            return {
                "success": True,
                "message": "All displayed and ignored article statuses have been reset"
            }
    except Exception as e:
        return {"error": f"Error resetting displayed article statuses: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport="stdio")