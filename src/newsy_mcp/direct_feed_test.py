#!/usr/bin/env python3
"""
Direct RSS Feed Test

A minimal script to test downloading an RSS feed directly without any dependencies
on the rest of the codebase.
"""

import asyncio
import aiohttp
import feedparser
import sys
import json
from datetime import datetime
from rich.console import Console
from urllib.parse import urlparse

# Configure console
console = Console()

async def test_feed_direct(feed_url: str) -> None:
    """Test downloading and parsing a specific RSS feed directly"""
    console.print(f"[bold blue]Testing direct feed download:[/bold blue] {feed_url}")
    
    try:
        # Step 1: Download the raw feed content
        async with aiohttp.ClientSession() as session:
            console.print("[yellow]Downloading feed content...[/yellow]")
            async with session.get(feed_url) as response:
                if response.status != 200:
                    console.print(f"[bold red]HTTP Error: {response.status}[/bold red]")
                    return
                
                content = await response.text()
                console.print(f"[green]✓[/green] Downloaded {len(content)} bytes")
        
        # Step 2: Parse with feedparser
        console.print("[yellow]Parsing feed content...[/yellow]")
        feed = feedparser.parse(content)
        
        if hasattr(feed, 'bozo') and feed.bozo:
            console.print(f"[bold red]Parser reported errors:[/bold red]")
            console.print(f"   {feed.bozo_exception}")
        
        # Step 3: Check feed entries
        entry_count = len(feed.entries)
        console.print(f"[green]✓[/green] Found {entry_count} entries in feed")
        
        if entry_count > 0:
            # Process first 3 entries as example
            for i, entry in enumerate(feed.entries[:3]):
                console.print(f"[bold]Entry {i+1}:[/bold] {entry.title}")
                console.print(f"  Link: {entry.link}")
                
                # Check for date information
                if hasattr(entry, 'published'):
                    console.print(f"  Published (raw): {entry.published}")
                
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    console.print(f"  Published (parsed): {entry.published_parsed}")
                else:
                    console.print("  [yellow]No parsed publication date available[/yellow]")
                
                # Check content fields
                content_fields = []
                if hasattr(entry, 'content'):
                    content_fields.append('content')
                if hasattr(entry, 'description'):
                    content_fields.append('description')
                if hasattr(entry, 'summary'):
                    content_fields.append('summary')
                
                console.print(f"  Content fields: {', '.join(content_fields) if content_fields else 'None'}")
                console.print("")
        
        # Step 4: Save sample entries to a JSON file for inspection
        sample_entries = []
        for entry in feed.entries[:5]:
            entry_dict = {k: v for k, v in entry.items() if not k.startswith('_')}
            sample_entries.append(entry_dict)
        
        with open('sample_entries.json', 'w') as f:
            json.dump(sample_entries, f, indent=2, default=str)
        
        console.print(f"[green]✓[/green] Saved 5 sample entries to sample_entries.json")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())

async def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        console.print("[bold red]Error: Please provide an RSS feed URL[/bold red]")
        console.print("Usage: python direct_feed_test.py <feed_url>")
        return
    
    feed_url = sys.argv[1]
    await test_feed_direct(feed_url)

if __name__ == "__main__":
    asyncio.run(main())