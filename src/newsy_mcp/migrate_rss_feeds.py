#!/usr/bin/env python3
"""
Migrate RSS feeds from rss_config.py to the database.

This script moves RSS feed configuration from the static Python file
to the new database structure.
"""

import sqlite3
from urllib.parse import urlparse
import sys

from rich.console import Console
from rss_config import RSS_FEEDS, ALL_RSS_FEEDS
from db_init import get_db_connection, init_database

console = Console()


def migrate_rss_feeds(db_path: str = "newsy.db") -> None:
    """
    Migrate RSS feeds from config file to the database.
    
    Args:
        db_path: Path to the SQLite database
    """
    console.print("[bold blue]Starting RSS feed migration to database[/]")
    
    try:
        # Initialize database first to ensure tables exist
        init_database(db_path)
        console.print("[bold green]Database tables initialized[/]")
        
        conn = get_db_connection(db_path)
        
        # Check if feeds are already migrated
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM rss_feeds")
        count = cursor.fetchone()[0]
        
        if count > 0:
            console.print(f"[yellow]Found {count} feeds already in database, clearing them[/]")
            # Always continue without prompting for non-interactive environments
        
        # Migrate all feeds
        feeds_added = 0
        
        # First, ensure the table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rss_feeds (
                id INTEGER PRIMARY KEY,
                url TEXT NOT NULL,
                name TEXT,
                category TEXT,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                last_fetch TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(url)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rss_feeds_is_active ON rss_feeds(is_active)")
        conn.commit()
        
        # Clear existing feeds to avoid duplicates
        cursor.execute("DELETE FROM rss_feeds")
        conn.commit()
        
        # Add all feeds from config
        for feed_url in ALL_RSS_FEEDS:
            # Check if feed is active
            is_active = feed_url in RSS_FEEDS
            
            # Extract domain for name
            domain = urlparse(feed_url).netloc
            if domain.startswith('www.'):
                domain = domain[4:]
                
            # Determine category (very basic)
            category = "tech"  # Default
            if "news" in domain:
                category = "news"
            elif any(x in domain for x in ["reddit", "hn", "hackernews", "slashdot"]):
                category = "social"
            
            try:
                cursor.execute(
                    """
                    INSERT INTO rss_feeds (url, name, category, is_active)
                    VALUES (?, ?, ?, ?)
                    """,
                    (feed_url, domain, category, is_active)
                )
                feeds_added += 1
                conn.commit()  # Commit each feed to ensure it's saved
            except sqlite3.Error as e:
                console.print(f"[bold red]Error adding feed {feed_url}: {str(e)}[/]")
        
        conn.commit()
        console.print(f"[bold green]Successfully migrated {feeds_added} RSS feeds to database[/]")
        
    except Exception as e:
        console.print(f"[bold red]Migration error: {str(e)}[/]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()


def migrate_user_preferences(db_path: str = "newsy.db") -> None:
    """
    Migrate user preferences from text file to database.
    
    Args:
        db_path: Path to the SQLite database
    """
    console.print("[bold blue]Starting user preferences migration to database[/]")
    
    try:
        # Initialize database first to ensure tables exist
        init_database(db_path)
        
        # Read preferences from user_preferences.txt
        likes = []
        avoids = []
        content_prefs = []
        
        current_section = None
        
        try:
            with open("user_preferences.txt", "r") as f:
                for line in f:
                    line = line.strip()
                    
                    if "## Topics I Like" in line:
                        current_section = "likes"
                    elif "## Topics to Avoid" in line:
                        current_section = "avoids"
                    elif "## Content Preferences" in line:
                        current_section = "content"
                    elif line.startswith("- ") and current_section:
                        item = line[2:].strip()
                        if current_section == "likes":
                            likes.append(item)
                        elif current_section == "avoids":
                            avoids.append(item)
                        elif current_section == "content":
                            content_prefs.append(item)
        except FileNotFoundError:
            console.print("[yellow]User preferences file not found, skipping migration[/]")
            return
            
        # Store in database
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        
        # Check if preferences are already migrated
        cursor.execute("SELECT COUNT(*) FROM user_preferences")
        count = cursor.fetchone()[0]
        
        if count > 0:
            console.print(f"[yellow]Found {count} preferences already in database, clearing them[/]")
            # Clear existing preferences
            cursor.execute("DELETE FROM user_preferences")
            conn.commit()
        
        # Migrate preferences
        prefs_added = 0
        
        # Add likes
        for idx, item in enumerate(likes):
            try:
                cursor.execute(
                    """
                    INSERT INTO user_preferences (key, value, category)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key, category) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (f"like_{idx}", item, "topic_liked")
                )
                prefs_added += 1
            except sqlite3.Error as e:
                console.print(f"[bold red]Error adding preference {item}: {str(e)}[/]")
        
        # Add avoids
        for idx, item in enumerate(avoids):
            try:
                cursor.execute(
                    """
                    INSERT INTO user_preferences (key, value, category)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key, category) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (f"avoid_{idx}", item, "topic_avoided")
                )
                prefs_added += 1
            except sqlite3.Error as e:
                console.print(f"[bold red]Error adding preference {item}: {str(e)}[/]")
                
        # Add content preferences
        for idx, item in enumerate(content_prefs):
            try:
                cursor.execute(
                    """
                    INSERT INTO user_preferences (key, value, category)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key, category) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (f"content_pref_{idx}", item, "content_preference")
                )
                prefs_added += 1
            except sqlite3.Error as e:
                console.print(f"[bold red]Error adding preference {item}: {str(e)}[/]")
        
        conn.commit()
        console.print(f"[bold green]Successfully migrated {prefs_added} user preferences to database[/]")
        
    except Exception as e:
        console.print(f"[bold red]Preferences migration error: {str(e)}[/]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()


if __name__ == "__main__":
    migrate_rss_feeds()
    migrate_user_preferences()
    console.print("[bold green]Migration completed![/]")