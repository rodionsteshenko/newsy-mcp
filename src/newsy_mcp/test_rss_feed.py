#!/usr/bin/env python3
"""
RSS Feed Tester

A debugging tool to test the download and parsing of a specific RSS feed.
This script helps identify issues with specific RSS feeds by showing detailed
debugging information about the feed structure.
"""

import sys
import asyncio
import feedparser
from datetime import datetime, timezone
from aiohttp import ClientSession, ClientTimeout
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint
from rich.syntax import Syntax
from rich.table import Table
from urllib.parse import urlparse
from time import mktime

# Configure console
console = Console()

# Default timeout (seconds)
REQUEST_TIMEOUT = 30

async def test_rss_feed(feed_url: str, show_entries: bool = True, verbose: bool = False) -> None:
    """Test an RSS feed by downloading and parsing it with detailed output"""
    console.print(f"[bold blue]Testing RSS feed:[/bold blue] {feed_url}")
    console.print(f"[cyan]Started at:[/cyan] {datetime.now().isoformat()}")
    
    # Create a timeout for the request
    timeout = ClientTimeout(total=REQUEST_TIMEOUT)
    
    try:
        # Step 1: Download the raw feed content
        console.print(Panel("[bold]Step 1: Downloading feed content[/bold]"))
        
        async with ClientSession(timeout=timeout) as session:
            start_time = datetime.now()
            async with session.get(feed_url) as response:
                elapsed = (datetime.now() - start_time).total_seconds()
                status = response.status
                content_type = response.headers.get('Content-Type', 'unknown')
                content = await response.text()
                content_length = len(content)
        
        console.print(f"[green]✓[/green] Download successful ({elapsed:.2f}s)")
        console.print(f"   Status code: [bold]{status}[/bold]")
        console.print(f"   Content type: [bold]{content_type}[/bold]")
        console.print(f"   Content length: [bold]{content_length}[/bold] bytes")
        
        if verbose:
            console.print(Panel("[bold]Raw Feed Content (first 1000 chars)[/bold]"))
            console.print(Syntax(content[:1000], "xml", theme="monokai", line_numbers=True))
        
        # Step 2: Parse with feedparser
        console.print(Panel("[bold]Step 2: Parsing with feedparser[/bold]"))
        
        feed = feedparser.parse(content)
        
        # Check if parsing was successful
        if hasattr(feed, 'bozo') and feed.bozo:
            console.print(f"[red]✗[/red] Parser reported errors:")
            console.print(f"   [bold red]Error:[/bold red] {feed.bozo_exception}")
        else:
            console.print(f"[green]✓[/green] Feed parsed successfully")
        
        # Display feed metadata
        console.print(Panel("[bold]Feed Metadata[/bold]"))
        metadata_table = Table(title="Feed Information")
        metadata_table.add_column("Property", style="cyan")
        metadata_table.add_column("Value")
        
        # Feed title
        feed_title = feed.feed.get('title', 'Unknown')
        metadata_table.add_row("Title", feed_title)
        
        # Feed link
        feed_link = feed.feed.get('link', 'Unknown')
        metadata_table.add_row("Link", feed_link)
        
        # Feed description
        feed_description = feed.feed.get('description', 'Unknown')
        metadata_table.add_row("Description", feed_description[:100] + "..." if len(feed_description) > 100 else feed_description)
        
        # Feed language
        feed_language = feed.feed.get('language', 'Unknown')
        metadata_table.add_row("Language", feed_language)
        
        # Feed updated
        feed_updated = feed.feed.get('updated', 'Unknown')
        metadata_table.add_row("Updated", feed_updated)
        
        # Feed generator
        feed_generator = feed.feed.get('generator', 'Unknown')
        metadata_table.add_row("Generator", feed_generator)
        
        # Entry count
        entry_count = len(feed.entries)
        metadata_table.add_row("Entry Count", str(entry_count))
        
        console.print(metadata_table)
        
        # Step 3: Display feed entries
        if show_entries and feed.entries:
            console.print(Panel(f"[bold]Feed Entries ({len(feed.entries)})[/bold]"))
            
            for i, entry in enumerate(feed.entries):
                if i >= 5 and not verbose:  # Show only first 5 entries unless verbose
                    console.print(f"[yellow]... {len(feed.entries) - 5} more entries (use --verbose to see all)[/yellow]")
                    break
                
                # Create a table for this entry
                entry_table = Table(title=f"Entry {i+1}")
                entry_table.add_column("Property", style="cyan")
                entry_table.add_column("Value")
                
                # Title
                entry_title = getattr(entry, 'title', 'Unknown')
                entry_table.add_row("Title", entry_title)
                
                # Link
                entry_link = getattr(entry, 'link', 'Unknown')
                entry_table.add_row("Link", entry_link)
                
                # Published date
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        published = datetime.fromtimestamp(mktime(entry.published_parsed), timezone.utc)
                        entry_table.add_row("Published", published.isoformat())
                    except Exception as e:
                        entry_table.add_row("Published", f"Error parsing date: {e}")
                else:
                    entry_table.add_row("Published", "No date available")
                
                # Check content fields
                content_fields = []
                if hasattr(entry, 'content'):
                    content_fields.append('entry.content')
                if hasattr(entry, 'description'):
                    content_fields.append('entry.description')
                if hasattr(entry, 'summary'):
                    content_fields.append('entry.summary')
                
                entry_table.add_row("Content Fields", ", ".join(content_fields) if content_fields else "None")
                
                # Check for any other interesting fields
                other_fields = [field for field in dir(entry) if not field.startswith('_') and field not in 
                               ['title', 'link', 'published', 'published_parsed', 'content', 'description', 'summary']]
                entry_table.add_row("Other Fields", ", ".join(other_fields[:5]) + 
                                   (f"... +{len(other_fields)-5} more" if len(other_fields) > 5 else ""))
                
                console.print(entry_table)
                
                # If verbose, show more details about content
                if verbose and content_fields:
                    for field in content_fields:
                        if field == 'entry.content':
                            for i, content_item in enumerate(entry.content):
                                console.print(f"[bold]Content {i+1}:[/bold]")
                                console.print(f"Type: {content_item.get('type', 'Unknown')}")
                                console.print(f"Language: {content_item.get('language', 'Unknown')}")
                                console.print(f"Value (first 200 chars): {content_item.value[:200]}...")
                        elif field == 'entry.description':
                            console.print("[bold]Description (first 200 chars):[/bold]")
                            console.print(entry.description[:200] + "...")
                        elif field == 'entry.summary':
                            console.print("[bold]Summary (first 200 chars):[/bold]")
                            console.print(entry.summary[:200] + "...")
        
        # Step 4: Summary
        console.print(Panel("[bold]Feed Test Summary[/bold]"))
        
        if not hasattr(feed, 'bozo') or not feed.bozo:
            console.print(f"[green]✓[/green] Feed format appears valid")
        else:
            console.print(f"[red]✗[/red] Feed has format issues")
        
        if feed.entries:
            console.print(f"[green]✓[/green] Feed contains {len(feed.entries)} entries")
        else:
            console.print(f"[red]✗[/red] Feed contains no entries")
        
        # Check for common issues
        issues = []
        
        # Check if entries have links
        entries_without_links = sum(1 for entry in feed.entries if not hasattr(entry, 'link'))
        if entries_without_links:
            issues.append(f"{entries_without_links} entries missing links")
        
        # Check if entries have published dates
        entries_without_dates = sum(1 for entry in feed.entries if not hasattr(entry, 'published_parsed') or not entry.published_parsed)
        if entries_without_dates:
            issues.append(f"{entries_without_dates} entries missing published dates")
        
        # Check if entries have content
        entries_without_content = sum(1 for entry in feed.entries 
                                    if (not hasattr(entry, 'content') or not entry.content) and 
                                       (not hasattr(entry, 'description') or not entry.description) and
                                       (not hasattr(entry, 'summary') or not entry.summary))
        if entries_without_content:
            issues.append(f"{entries_without_content} entries missing content")
        
        if issues:
            console.print("[yellow]⚠️ Issues found:[/yellow]")
            for issue in issues:
                console.print(f"   - {issue}")
        else:
            console.print("[green]✓[/green] No common issues detected")
        
    except Exception as e:
        console.print(f"[bold red]Error testing feed:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())

def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description="Test an RSS feed")
    parser.add_argument("url", help="URL of the RSS feed to test")
    parser.add_argument("--no-entries", action="store_true", help="Don't show feed entries")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    return parser.parse_args()

async def main():
    args = parse_args()
    await test_rss_feed(args.url, show_entries=not args.no_entries, verbose=args.verbose)

if __name__ == "__main__":
    asyncio.run(main())