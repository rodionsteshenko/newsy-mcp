# Newsy MCP

Newsy MCP is an RSS feed manager with an integrated MCP API that allows you to manage RSS feeds, user preferences, and search articles. This document provides guidance for working with this codebase.

## Environment Setup

Before running any commands, agents should activate the virtual environment:

```bash
source .venv/bin/activate
```

After activating the environment, all Python commands should be run with `uv run` instead of `python`:

```bash
# Instead of: python rss_downloader.py
uv run rss_downloader.py

# Instead of: python db_init.py  
uv run db_init.py

# Instead of: python migrate_preferences.py
uv run migrate_preferences.py
```

The MCP server command remains the same as it already uses `uv run`.

## Repository Structure

- `/src/newsy_mcp/` - Main code directory containing all Python files
  - `rss_downloader.py` - Core module for downloading and processing RSS feeds
  - `rss_server.py` - MCP API server for RSS functionality
  - `db_init.py` - Database initialization and schema definition
  - `rss_config.py` - RSS feed configuration and constants
  - `migrate_preferences.py` - Tool to migrate user preferences to the database

## Key Components

### Database

The system uses SQLite with the following main tables:
- `articles` - Stores article content and metadata
- `article_analysis` - Stores analysis results for articles
- `user_preferences` - Stores user preferences by category
- `rss_feeds` - Stores RSS feed configuration
- `failed_urls` - Tracks failed download attempts

Database initialization is handled in `db_init.py`, which defines the complete schema and provides connection management utilities.

### RSS Downloader

The RSS downloader (`rss_downloader.py`) supports:
- Asynchronous feed processing with configurable concurrency
- URL deduplication and failure tracking
- Content processing (HTML to markdown/text conversion)
- Loop mode with configurable intervals via `-i` flag
- Special handling for feeds like Nature that may not include standard publication dates

Run the downloader with:
```bash
uv run rss_downloader.py
```

For loop mode with a 30-minute interval:
```bash
uv run rss_downloader.py -i 30
```

### MCP API Server

The MCP API server (`rss_server.py`) provides endpoints for:
- User preferences management
- RSS feed configuration
- Article retrieval and search
- Database statistics

Run the server with:
```
uv --directory /Users/rodion/newsy_mcp/src/newsy_mcp run mcp run rss_server.py
```

### User Preferences

User preferences are stored in the `user_preferences` table with three categories:
- `topic_liked` - Topics the user is interested in
- `topic_avoided` - Topics the user wants to avoid
- `content_preference` - General content preferences

The `migrate_preferences.py` script can parse preferences from a markdown-formatted text file and store them in the database.

## Common Operations

### Initialize Database

```bash
uv run db_init.py
```

### Migrate RSS Feeds

```bash
uv run migrate_rss_feeds.py
```

### Migrate User Preferences

```bash
uv run migrate_preferences.py
```

### Run RSS Downloader

Single run:
```bash
uv run rss_downloader.py
```

Loop mode (runs continuously with specified interval in minutes):
```bash
uv run rss_downloader.py -i <minutes>
```

### Run MCP Server

```bash
uv run mcp run rss_server.py
```

## Dependencies

- `feedparser` - RSS feed parsing
- `aiohttp` - Async HTTP requests
- `markdownify` - HTML to markdown conversion
- `rich` - Console output formatting
- `crawl4ai` - Optional dependency for enhanced content extraction

## Development Notes

- The downloader has fallback methods when `crawl4ai` is not available
- Image downloading has been removed to simplify the process
- Nature RSS feeds require special handling for publication dates
- Loop mode uses an interval parameter to control download frequency