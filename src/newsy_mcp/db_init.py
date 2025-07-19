"""
Database Initialization Module

Features:
1. Centralized Schema Management
   - Single source of truth for database schema
   - Consistent table creation across modules
   - Foreign key relationship management
   - Index creation and optimization

2. Migration Support
   - Version tracking
   - Schema updates
   - Data preservation during upgrades

3. Error Handling
   - Proper connection management
   - Transaction support
   - Detailed error reporting
"""

import sqlite3
from utils import dprint


def init_database(db_path: str = "newsy.db") -> None:
    """Initialize SQLite database with complete schema"""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        # Create articles table first (parent table)
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

        # Create article_analysis table
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
                relevance_details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(article_id) REFERENCES articles(id),
                UNIQUE(article_id)
            )
        """)

        # Create sent_articles table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sent_articles (
                id INTEGER PRIMARY KEY,
                article_id INTEGER NOT NULL,
                sent_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(article_id) REFERENCES articles(id),
                UNIQUE(article_id)
            )
        """)

        # Create article_reactions table
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

        # Create article_embeddings table
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

        # Create failed_urls table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS failed_urls (
                url TEXT PRIMARY KEY,
                error_message TEXT NOT NULL,
                attempt_count INTEGER DEFAULT 1,
                last_attempt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                first_attempt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create user_preferences table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(key, category)
            )
        """)

        # Create rss_feeds table
        conn.execute("""
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
        
        # Create displayed_articles table to track viewed/ignored articles
        conn.execute("""
            CREATE TABLE IF NOT EXISTS displayed_articles (
                id INTEGER PRIMARY KEY,
                article_id INTEGER NOT NULL,
                status TEXT NOT NULL,  -- 'displayed' or 'ignored'
                display_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                agent_session_id TEXT,
                FOREIGN KEY(article_id) REFERENCES articles(id),
                UNIQUE(article_id)
            )
        """)

        # Create performance indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_article_id ON article_analysis(article_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reactions_article_id ON article_reactions(article_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_article_id ON article_embeddings(article_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sent_articles_article_id ON sent_articles(article_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_preferences_key ON user_preferences(key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_preferences_category ON user_preferences(category)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rss_feeds_is_active ON rss_feeds(is_active)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_displayed_articles_article_id ON displayed_articles(article_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_displayed_articles_status ON displayed_articles(status)"
        )

        conn.commit()
        dprint("Database initialization completed successfully")

    except Exception as e:
        dprint(f"Error initializing database: {str(e)}", error=True)
        raise
    finally:
        conn.close()


def get_db_connection(db_path: str = "newsy.db") -> sqlite3.Connection:
    """Get database connection with proper settings"""
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
