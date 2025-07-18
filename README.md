# Newsy MCP: RSS Feed Manager with MCP API

Newsy MCP is an RSS feed manager with an integrated MCP API that allows you to manage your RSS feeds, user preferences, and search articles.

## Features

- RSS feed management
- User preference storage and retrieval
- Article search and retrieval
- Vector-based similarity search
- Database statistics and monitoring

## Running the MCP Server

To run the original PDF MCP server:
```
uv --directory /Users/rodion/newsy_mcp/src/newsy_mcp run mcp run server.py
```

To run the new RSS MCP server:
```
uv --directory /Users/rodion/newsy_mcp/src/newsy_mcp run mcp run rss_server.py
```

## Database Migration

Before using the RSS MCP server, you should run the migration script to initialize the database with your RSS feeds and user preferences:

```
uv --directory /Users/rodion/newsy_mcp/src/newsy_mcp run migrate_rss_feeds.py
```

## API Documentation

### User Preferences API

#### Get User Preferences
```python
get_user_preferences() -> Dict[str, Any]
```
Returns the current user preferences including liked topics, topics to avoid, and content preferences.

#### Set User Preferences
```python
set_user_preferences(preferences: Dict[str, Any]) -> Dict[str, Any]
```
Updates user preferences. The preferences dictionary can include:
- `topics_liked`: List of topics the user likes
- `topics_to_avoid`: List of topics the user wants to avoid
- `content_preferences`: List of general content preferences

### RSS Feed Management API

#### Get Active Feeds
```python
get_active_feeds() -> Dict[str, Any]
```
Returns a list of currently active RSS feed URLs.

#### Get All Feeds
```python
get_all_feeds() -> Dict[str, Any]
```
Returns detailed information about all available RSS feeds, both active and inactive.

#### Get Feed by URL
```python
get_feed_by_url(feed_url: str) -> Dict[str, Any]
```
Returns detailed information about a specific RSS feed, including recent articles.

#### Set Active Feeds
```python
set_active_feeds(feeds: List[str]) -> Dict[str, Any]
```
Updates the list of active RSS feeds.

### Article Retrieval API

#### Get Recent Articles
```python
get_recent_articles(limit: int = 20, offset: int = 0) -> Dict[str, Any]
```
Retrieves a paginated list of recently downloaded articles.

#### Get Article by ID
```python
get_article_by_id(article_id: int) -> Dict[str, Any]
```
Retrieves a specific article by its ID, including full content and images.

#### Search Articles
```python
search_articles(query: str, limit: int = 10) -> Dict[str, Any]
```
Searches articles using keyword matching.

#### Search Articles by Vector
```python
search_articles_by_vector(query: str, limit: int = 10) -> Dict[str, Any]
```
Searches articles using vector similarity (semantic search).

### Database Statistics API

#### Get Database Stats
```python
get_database_stats() -> Dict[str, Any]
```
Returns statistics about the RSS database, including article counts, feed counts, etc.

## Database Schema

The system uses SQLite with the following main tables:

- `articles`: Stores article content and metadata
- `article_analysis`: Stores analysis results for articles
- `article_images`: Stores images associated with articles
- `user_preferences`: Stores user preferences by category
- `rss_feeds`: Stores RSS feed information and status