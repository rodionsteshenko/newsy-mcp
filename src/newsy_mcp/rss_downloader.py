#!/usr/bin/env python3
"""
RSS Feed Downloader

Features:
1. Parallel RSS Feed Processing
   - Asynchronous feed downloads
   - Configurable concurrency limits
   - Error handling and retries
   - URL deduplication check before downloading

2. Content Processing
   - HTML to text conversion
   - Markdown generation
   - Content deduplication

3. Database Storage
   - SQLite with proper schema
   - Failed URL tracking
   - Article metadata storage
   - Unique URL constraints

4. Error Handling
   - Feed validation
   - URL failure tracking
   - Graceful degradation

5. Monitoring
   - Rich console output
   - Download statistics
   - Error reporting
"""

import sqlite3
import feedparser
import requests
import argparse
import time
from datetime import datetime, timezone, timedelta
from markdownify import markdownify
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import Optional, Dict, List
from urllib.parse import urlparse
from pathlib import Path
from dataclasses import dataclass, field
from time import mktime
import asyncio
from aiohttp import ClientSession, ClientTimeout
import urllib3
from urllib3.exceptions import InsecureRequestWarning
# Note: content_util import requires crawl4ai which is optional
# The downloader will work without it but won't get full article content
try:
    from content_util import scrape_url_content
    HAS_CONTENT_UTIL = True
except ImportError:
    HAS_CONTENT_UTIL = False
    def scrape_url_content(*args, **kwargs):
        return None
from rss_config import RSS_FEEDS
from utils import dprint
from db_init import init_database, get_db_connection
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from asyncio import Semaphore

# Configure sleep interval between downloads (15 minutes)
SLEEP_INTERVAL = 15 * 60

# Debug flag
DEBUG = True

# Timeouts and retry settings
REQUEST_TIMEOUT = 10  # Reduced from 30 to prevent long hangs
MAX_RETRIES = 1
RETRY_DELAY = 2  # Reduced from 5 for quicker operation

# Increase window to 48 hours to ensure we don't miss articles
ARTICLE_AGE_WINDOW = 48 * 60 * 60  # 48 hours in seconds

# Configure console with no_color option to avoid colorama recursion error
console = Console(no_color=True, highlight=False)

# Suppress insecure request warnings
urllib3.disable_warnings(InsecureRequestWarning)


@dataclass
class ArticleContent:
    """Container for article content types"""

    raw: str
    text: str
    markdown: str


@dataclass
class FeedStats:
    """Container for feed processing statistics"""

    total_articles: int = 0
    new_articles: int = 0
    failed_articles: int = 0
    feed_url: str = ""
    error_message: Optional[str] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ArticleToDownload:
    """Container for article information before download"""

    title: str
    url: str
    feed_url: str
    published: datetime


class RssDownloader:
    def __init__(self, db_path: str = "newsy.db"):
        self.db_path = Path(db_path)
        self.setup_database()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 RSS Feed Downloader"})
        self.session.verify = False
        self.session.timeout = REQUEST_TIMEOUT
        self.feed_stats: Dict[str, FeedStats] = {}
        self.timeout = ClientTimeout(total=REQUEST_TIMEOUT)

    def setup_database(self) -> None:
        """Initialize SQLite database"""
        init_database(self.db_path)

    def is_failed_url(self, url: str) -> bool:
        """Check if URL is in failed_urls table with too many attempts"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT attempt_count FROM failed_urls WHERE url = ?", (url,))
            result = c.fetchone()
            return bool(result and result[0] >= MAX_RETRIES)

    def record_failed_url(self, url: str, error_message: str) -> None:
        """Record or update failed URL attempt"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO failed_urls (url, error_message) 
                VALUES (?, ?)
                ON CONFLICT(url) DO UPDATE SET 
                    attempt_count = attempt_count + 1,
                    last_attempt = CURRENT_TIMESTAMP,
                    error_message = ?
            """,
                (url, error_message, error_message),
            )

    async def get_article_content(self, url: str) -> Optional[ArticleContent]:
        """Download and extract article content with improved error handling"""
        if self.is_failed_url(url):
            dprint(f"Skipping previously failed URL: {url}")
            return None

        try:
            dprint(f"Attempting to download content from: {url}")
            
            # If content_util is not available, use a simple download method
            if not HAS_CONTENT_UTIL:
                # Simple fallback method to get page content
                async with ClientSession() as session:
                    async with session.get(url, timeout=self.timeout) as response:
                        if response.status != 200:
                            raise ValueError(f"HTTP {response.status}")
                        content = await response.text()
                        
                        # Create a simple ArticleContent object
                        return ArticleContent(
                            raw=content,
                            text=content,
                            markdown=content
                        )
            
            # If content_util is available, use it
            result = await scrape_url_content(url, extract_images=False)
            if not result.success:
                error_msg = result.error_message or "Unknown error"
                dprint(f"Failed to scrape URL {url}: {error_msg}", error=True)
                self.record_failed_url(url, error_msg)
                return None

            # Create ArticleContent object without images
            content = ArticleContent(
                raw=result.content,
                text=result.content,
                markdown=markdownify(result.content),
            )

            if not content.raw:
                dprint(f"No content extracted from {url}", error=True)
                self.record_failed_url(url, "No content extracted")
                return None

            dprint(
                f"Successfully downloaded {len(content.raw)} chars from {url}"
            )
            return content

        except Exception as e:
            dprint(f"Error downloading {url}: {str(e)}", error=True)
            self.record_failed_url(url, str(e))
            return None

    def article_exists(self, url: str) -> bool:
        """Check if article already exists using connection context manager"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM articles WHERE url = ?", (url,))
            count = c.fetchone()[0]
            if count > 0:
                dprint(f"Article already exists in database: {url}")
                return True
            return False

    def store_article(
        self,
        title: str,
        url: str,
        feed_url: str,
        timestamp: str,
        content: ArticleContent,
    ) -> None:
        """Store article with proper connection handling"""
        feed_domain = urlparse(feed_url).netloc

        with sqlite3.connect(self.db_path, timeout=10) as conn:
            c = conn.cursor()
            # First check if URL exists to prevent duplicates
            if self.article_exists(url):
                dprint(f"Skipping duplicate article URL: {url}")
                return

            # Insert article
            c.execute(
                """
                INSERT INTO articles 
                (title, url, feed_domain, timestamp, content, content_html2text, content_markdown)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    title,
                    url,
                    feed_domain,
                    timestamp,
                    content.raw,
                    content.text,
                    content.markdown,
                ),
            )

    async def process_feed(self, feed_url: str) -> FeedStats:
        """Process a single RSS feed with retries and validation"""
        stats = FeedStats(feed_url=feed_url)
        dprint(f"\nProcessing feed {feed_url}")
        dprint(f"Current UTC time: {stats.start_time.isoformat()}")
        
        # Check if this is a known feed that might need special handling
        if "nature.com" in feed_url:
            dprint(f"Nature feed detected: {urlparse(feed_url).netloc} (will handle missing publication dates)")


        # Calculate cutoff time (48 hours ago)
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=ARTICLE_AGE_WINDOW)
        dprint(
            f"Cutoff time ({ARTICLE_AGE_WINDOW / 3600:.1f}h ago): {cutoff_time.isoformat()}"
        )

        for attempt in range(MAX_RETRIES):
            try:
                async with ClientSession(timeout=self.timeout) as session:
                    async with session.get(feed_url) as response:
                        if response.status != 200:
                            raise ValueError(f"HTTP {response.status}")
                        content = await response.text()

                # Parse feed content
                feed = feedparser.parse(content)

                if not feed.entries:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    stats.error_message = (
                        f"No entries found in feed after {MAX_RETRIES} attempts"
                    )
                    return stats

                stats.total_articles = len(feed.entries)
                dprint(f"Found {stats.total_articles} articles in feed")

                # Create progress bar for this feed
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                ) as progress:
                    task = progress.add_task(
                        f"[cyan]Processing {urlparse(feed_url).netloc}[/cyan]",
                        total=len(feed.entries),
                    )

                    for entry in feed.entries:
                        if not hasattr(entry, "link"):
                            dprint(f"Entry missing link in {feed_url}", error=True)
                            progress.advance(task)
                            continue

                        # Skip if article URL already exists
                        if self.article_exists(entry.link):
                            dprint(f"Skipping existing article: {entry.link}")
                            progress.advance(task)
                            continue

                        # Get article timestamp
                        try:
                            # Try to get published date from entry
                            if hasattr(entry, "published_parsed") and entry.published_parsed:
                                published = datetime.fromtimestamp(
                                    mktime(entry.published_parsed), timezone.utc
                                )
                                dprint(
                                    f"Article timestamp from feed: {published.isoformat()}"
                                )
                            elif "nature.com" in feed_url and hasattr(entry, "link"):
                                # Special handling for Nature URLs which have dates encoded in them
                                match = re.search(r'/d\d{5}-(\d{3})-(\d{5})-', entry.link)
                                if match:
                                    try:
                                        year_code = match.group(1)
                                        date_code = match.group(2)
                                        
                                        # Convert year_code to full year (e.g., 025 -> 2025)
                                        year = 2000 + int(year_code)
                                        
                                        # Try to extract month and day from date_code
                                        month = min(max(int(date_code[:2]), 1), 12)  # First 2 digits as month
                                        day = min(max(int(date_code[2:4]), 1), 28)   # Next 2 digits as day
                                        
                                        published = datetime(year, month, day, tzinfo=timezone.utc)
                                        dprint(f"Date extracted from Nature URL: {published.isoformat()}")
                                    except Exception as e:
                                        dprint(f"Error extracting date from Nature URL: {e}")
                                        published = datetime.now(timezone.utc)  # Fallback
                                else:
                                    published = datetime.now(timezone.utc)  # Fallback for Nature URLs without date pattern
                            else:
                                # If no timestamp, assume it's new
                                published = datetime.now(timezone.utc)
                                dprint(
                                    f"No timestamp in feed, using current UTC: {published.isoformat()}"
                                )
                        except Exception as e:
                            published = datetime.now(timezone.utc)
                            dprint(
                                f"Invalid date in entry from {feed_url}: {e}",
                                error=True,
                            )
                            dprint(
                                f"Using current UTC time instead: {published.isoformat()}"
                            )

                        # Skip articles older than cutoff
                        if published < cutoff_time:
                            dprint(
                                f"Skipping old article: {entry.title} "
                                f"(age: {(datetime.now(timezone.utc) - published).total_seconds() / 3600:.1f}h)"
                            )
                            progress.advance(task)
                            continue

                        # Update progress description while downloading
                        progress.update(
                            task,
                            description=f"[cyan]Downloading: {entry.title[:40]}...",
                        )

                        content = await self.get_article_content(entry.link)
                        if not content:
                            stats.failed_articles += 1
                            progress.advance(task)
                            continue

                        self.store_article(
                            entry.title,
                            entry.link,
                            feed_url,
                            published.isoformat(),
                            content,
                        )
                        stats.new_articles += 1
                        dprint(f"Successfully stored: {entry.title}")
                        progress.advance(task)

                    return stats

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                stats.error_message = str(e)
                return stats

        return stats

    def _load_active_feeds(self) -> List[str]:
        """Load active RSS feeds from database, falling back to config file if necessary."""
        try:
            with get_db_connection(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT url FROM rss_feeds 
                    WHERE is_active = 1
                    ORDER BY created_at
                    """
                )
                feeds = [row[0] for row in cursor.fetchall()]
                if feeds:
                    dprint(f"Loaded {len(feeds)} active feeds from database")
                    return feeds
        except Exception as e:
            dprint(f"Error loading feeds from database: {str(e)}", error=True)
        
        # Fall back to config file
        dprint(f"Using {len(RSS_FEEDS)} feeds from config file")
        return RSS_FEEDS.copy()

    async def run(self) -> List[FeedStats]:
        """Run a single download cycle and return stats"""
        # Get active feeds from database or config
        active_feeds = self._load_active_feeds()
        
        if not active_feeds:
            raise ValueError("No active RSS feeds configured")

        # Set a global timeout for the entire run to prevent hanging
        try:
            # 5 minute maximum runtime
            async with asyncio.timeout(300):
                return await self._run_with_feeds(active_feeds)
        except asyncio.TimeoutError:
            dprint("[bold red]Global timeout reached after 5 minutes! Stopping.[/]")
            return [FeedStats(feed_url="TIMEOUT", error_message="Global timeout reached after 5 minutes")]
        except KeyboardInterrupt:
            dprint("[yellow]Operation cancelled by user[/yellow]")
            return []
            
    async def _run_with_feeds(self, active_feeds: List[str]) -> List[FeedStats]:
        """Run the download process with the provided feeds"""
        feed_stats: List[FeedStats] = []
        articles_to_download: List[ArticleToDownload] = []

        # First gather all articles from feeds
        dprint("\nGathering articles from feeds...")

        # Calculate cutoff time once
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=ARTICLE_AGE_WINDOW)
        dprint(
            f"Cutoff time ({ARTICLE_AGE_WINDOW / 3600:.1f}h ago): {cutoff_time.isoformat()}"
        )

        async with ClientSession(timeout=self.timeout) as session:
            for feed_url in active_feeds:
                stats = FeedStats(feed_url=feed_url)
                dprint(f"Processing feed: {feed_url}")
                try:
                    async with session.get(feed_url) as response:
                        if response.status != 200:
                            raise ValueError(f"HTTP {response.status}")
                        content = await response.text()

                    # Parse feed content
                    feed = feedparser.parse(content)
                    
                    stats.total_articles = len(feed.entries)
                    dprint(f"Found {stats.total_articles} articles in {feed_url}")

                    for entry in feed.entries:
                        if not hasattr(entry, "link") or self.article_exists(
                            entry.link
                        ):
                            continue

                        try:
                            # Try to get published date from entry
                            if hasattr(entry, "published_parsed") and entry.published_parsed:
                                published = datetime.fromtimestamp(
                                    mktime(entry.published_parsed), timezone.utc
                                )
                                dprint(
                                    f"Article timestamp from feed: {published.isoformat()}"
                                )
                            elif "nature.com" in feed_url and hasattr(entry, "link"):
                                # Special handling for Nature URLs which have dates encoded in them
                                match = re.search(r'/d\d{5}-(\d{3})-(\d{5})-', entry.link)
                                if match:
                                    try:
                                        year_code = match.group(1)
                                        date_code = match.group(2)
                                        
                                        # Convert year_code to full year (e.g., 025 -> 2025)
                                        year = 2000 + int(year_code)
                                        
                                        # Try to extract month and day from date_code
                                        month = min(max(int(date_code[:2]), 1), 12)  # First 2 digits as month
                                        day = min(max(int(date_code[2:4]), 1), 28)   # Next 2 digits as day
                                        
                                        published = datetime(year, month, day, tzinfo=timezone.utc)
                                        dprint(f"Date extracted from Nature URL: {published.isoformat()}")
                                    except Exception as e:
                                        dprint(f"Error extracting date from Nature URL: {e}")
                                        published = datetime.now(timezone.utc)  # Fallback
                                else:
                                    published = datetime.now(timezone.utc)  # Fallback for Nature URLs without date pattern
                            else:
                                # If no timestamp, assume it's new
                                published = datetime.now(timezone.utc)
                                dprint(
                                    f"No timestamp in feed, using current UTC: {published.isoformat()}"
                                )
                        except Exception:
                            published = datetime.now(timezone.utc)

                        if published >= cutoff_time:
                            articles_to_download.append(
                                ArticleToDownload(
                                    title=entry.title,
                                    url=entry.link,
                                    feed_url=feed_url,
                                    published=published,
                                )
                            )

                except Exception as e:
                    stats.error_message = str(e)

                feed_stats.append(stats)

        total_to_download = len(articles_to_download)
        dprint(f"\nFound {total_to_download} new articles to download")

        if total_to_download == 0:
            return feed_stats

        # Now download all articles with a single progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Downloading {total_to_download} articles[/cyan]",
                total=total_to_download,
            )

            # Use semaphore to limit concurrent downloads
            semaphore = Semaphore(5)  # Reduced from 10 to 5 for better stability
            tasks = [
                self.download_article(article, progress, task, semaphore)
                for article in articles_to_download
            ]
            # Don't use gather with return_exceptions as it can hide issues
            # Instead process tasks in batches with individual timeouts
            results = []
            batch_size = 10  # Process 10 tasks at a time to avoid overwhelming memory
            
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                batch_results = await asyncio.gather(*batch)
                results.extend(batch_results)

            # Calculate stats based on success/failure in the database
            # This avoids issues with Exception tracking from gather()
            for article in articles_to_download:
                for stats in feed_stats:
                    if stats.feed_url == article.feed_url:
                        if self.article_exists(article.url):
                            stats.new_articles += 1
                        else:
                            stats.failed_articles += 1
                        break

        return feed_stats

    async def download_article(
        self,
        article: ArticleToDownload,
        progress: Progress,
        task_id: int,
        semaphore: Semaphore,
    ) -> None:
        """Download and store a single article"""
        async with semaphore:
            try:
                # Update progress description
                progress.update(
                    task_id,
                    description=f"[cyan]Downloading: {article.title[:40]}...",
                )

                # Skip if URL previously failed
                if self.is_failed_url(article.url):
                    dprint(f"Skipping previously failed URL: {article.url}")
                    progress.update(
                        task_id,
                        description=f"[yellow]Skipped failed URL: {article.title[:40]}...",
                    )
                    return
                    
                # Set a timeout for this specific article download
                try:
                    async with asyncio.timeout(REQUEST_TIMEOUT):
                        content = await self.get_article_content(article.url)
                except asyncio.TimeoutError:
                    self.record_failed_url(article.url, "Download timed out")
                    progress.update(
                        task_id,
                        description=f"[red]Timeout: {article.title[:40]}...",
                    )
                    return
                    
                if not content:
                    progress.update(
                        task_id,
                        description=f"[red]Failed: {article.title[:40]}...",
                    )
                    return

                self.store_article(
                    article.title,
                    article.url,
                    article.feed_url,
                    article.published.isoformat(),
                    content,
                )

                progress.update(
                    task_id,
                    description=f"[green]Success: {article.title[:40]}...",
                )

                if DEBUG:
                    dprint(f"Successfully stored: {article.title}")

            except Exception as e:
                dprint(f"Failed to download {article.url}: {str(e)}", error=True)
                self.record_failed_url(article.url, str(e))
                # Don't re-raise the exception - just record it and continue
            finally:
                progress.advance(task_id)


def display_summary(feed_stats: List[FeedStats], db_path: str) -> None:
    """Display a rich summary table of feed processing results"""
    current_utc = datetime.now(timezone.utc)
    dprint(f"\nFeed processing summary at UTC: {current_utc.isoformat()}")
    dprint(f"Your local time (Eastern): {current_utc.astimezone().isoformat()}")

    # Get recent article counts from the database
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        # Get recent article counts
        c.execute("""
            SELECT COUNT(*) as article_count
            FROM articles
            WHERE timestamp >= datetime('now', '-1 day')
        """)
        recent_stats = c.fetchone()
        recent_articles = recent_stats[0]

        # Get failed URLs
        c.execute(
            """
            SELECT url, error_message, attempt_count, last_attempt 
            FROM failed_urls 
            WHERE attempt_count >= ?
            ORDER BY last_attempt DESC
            LIMIT 5
        """,
            (MAX_RETRIES,),
        )
        failed_urls = c.fetchall()

    if failed_urls:
        failed_table = Table(title="Recent Failed URLs")
        failed_table.add_column("URL", style="cyan")
        failed_table.add_column("Error Message", style="red")
        failed_table.add_column("Attempts", justify="right")
        failed_table.add_column("Last Attempt", style="yellow")

        for url, error, attempts, last_attempt in failed_urls:
            failed_table.add_row(
                url[:50] + "..." if len(url) > 50 else url,
                error[:50] + "..." if len(error) > 50 else error,
                str(attempts),
                last_attempt,
            )

        console.print(failed_table)
        console.print("")

    table = Table(title="RSS Feed Processing Summary")
    table.add_column("Feed URL", style="cyan")
    table.add_column("Total Articles", justify="right", style="blue")
    table.add_column("New Articles", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Error", style="red")
    table.add_column("Start Time (UTC)", style="yellow")

    total_new = 0
    total_failed = 0

    for stats in feed_stats:
        table.add_row(
            stats.feed_url,
            str(stats.total_articles),
            str(stats.new_articles),
            str(stats.failed_articles),
            stats.error_message or "",
            stats.start_time.isoformat(),
        )
        total_new += stats.new_articles
        total_failed += stats.failed_articles

    console.print(table)

    # Create recent articles statistics string
    recent_stats_str = (
        f"[blue]Last 24 hours:[/blue] "
        f"[green]{recent_articles}[/green] articles"
    )

    console.print(
        Panel(
            "\n".join(
                [
                    f"[green]Total new articles: {total_new}[/green]",
                    f"[red]Total failed: {total_failed}[/red]",
                    "",
                    recent_stats_str,
                    "",
                    f"[yellow]Process completed at UTC: {current_utc.isoformat()}[/yellow]",
                    f"[yellow]Your local time (Eastern): {current_utc.astimezone().isoformat()}[/yellow]",
                ]
            ),
            title="Summary",
        )
    )


async def run_once() -> None:
    """Run a single download cycle"""
    downloader = RssDownloader()
    stats = await downloader.run()
    display_summary(stats, downloader.db_path)
    return stats

async def run_loop(interval_minutes: int) -> None:
    """Run the downloader in a loop with the specified interval"""
    console.print(f"[bold blue]Starting RSS downloader in loop mode. Interval: {interval_minutes} minutes[/bold blue]")
    
    try:
        while True:
            start_time = time.time()
            console.print(f"\n[bold green]Starting download cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold green]")
            
            try:
                stats = await run_once()
                total_new = sum(stat.new_articles for stat in stats)
                total_failed = sum(stat.failed_articles for stat in stats)
                console.print(f"[bold]Cycle summary: {total_new} new articles, {total_failed} failed[/bold]")
            except Exception as e:
                console.print(f"[bold red]Error in download cycle: {str(e)}[/bold red]")
            
            # Calculate sleep time (accounting for how long the download took)
            elapsed = time.time() - start_time
            sleep_time = max(0, interval_minutes * 60 - elapsed)
            
            if sleep_time > 0:
                console.print(f"[yellow]Waiting {sleep_time:.1f} seconds until next cycle...[/yellow]")
                await asyncio.sleep(sleep_time)
    
    except asyncio.CancelledError:
        console.print("[bold yellow]Loop cancelled by user[/bold yellow]")
    except KeyboardInterrupt:
        console.print("[bold yellow]Loop interrupted by user[/bold yellow]")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Download RSS feeds and store articles in database")
    
    # Loop mode options
    parser.add_argument("-i", "--interval", type=int, help="Run in loop mode with specified interval in minutes")
    
    return parser.parse_args()

async def main() -> None:
    """Main entry point with argument handling"""
    args = parse_args()
    
    if args.interval:
        # Run in loop mode with specified interval
        await run_loop(args.interval)
    else:
        # Run once
        await run_once()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Process interrupted by user[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error: {str(e)}[/bold red]")
