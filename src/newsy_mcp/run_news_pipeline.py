#!/usr/bin/env python3
"""
News Pipeline Runner

This script runs the RSS downloader and article summarizer in a continuous loop with
configurable intervals. This keeps the news database fresh with the latest articles
and their summaries.

Usage:
  python run_news_pipeline.py [--download-interval MINUTES] [--summarize-interval MINUTES]

Options:
  --download-interval   Minutes between RSS feed downloads (default: 30)
  --summarize-interval  Minutes between summarization runs (default: 15)
  --batch-size          Number of articles to summarize per batch (default: 10)
"""

import sys
import time
import asyncio
import argparse
from datetime import datetime
import logging
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler

# Import our modules
from rss_downloader import RssDownloader
import summarize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("news_pipeline")

# Rich console for output
console = Console()


async def run_downloader() -> None:
    """Run the RSS downloader"""
    try:
        log.info("Starting RSS feed download process...")
        downloader = RssDownloader()
        stats = await downloader.run()
        
        # Count total new articles
        total_new = sum(stat.new_articles for stat in stats)
        log.info(f"RSS download completed. Downloaded {total_new} new articles.")
        
        return total_new
    except Exception as e:
        log.error(f"Error during RSS download: {str(e)}")
        return 0


def run_summarizer(batch_size: int) -> int:
    """Run the article summarizer"""
    try:
        log.info("Starting article summarization process...")
        
        # The main function in summarize.py processes articles but doesn't return counts,
        # so we need to get unsummarized articles first to know the count
        articles = summarize.get_unsummarized_articles(batch_size)
        article_count = len(articles)
        
        if article_count == 0:
            log.info("No unsummarized articles found.")
            return 0
        
        # Run the main summarization function
        summarize.main()
        
        log.info(f"Summarization completed. Processed {article_count} articles.")
        return article_count
    except Exception as e:
        log.error(f"Error during summarization: {str(e)}")
        return 0


async def run_pipeline_iteration(download_interval: int, summarize_interval: int, batch_size: int) -> None:
    """Run one iteration of the pipeline"""
    # Track the last run times
    last_download = datetime.now()
    last_summarize = datetime.now()
    
    while True:
        now = datetime.now()
        download_elapsed = (now - last_download).total_seconds() / 60  # minutes
        summarize_elapsed = (now - last_summarize).total_seconds() / 60  # minutes
        
        # Run RSS downloader if it's time
        if download_elapsed >= download_interval:
            console.rule(f"[bold blue]RSS Download - {now.strftime('%Y-%m-%d %H:%M:%S')}[/bold blue]")
            new_articles = await run_downloader()
            last_download = datetime.now()
            
            # If new articles were downloaded, immediately run summarization
            if new_articles > 0:
                console.rule(f"[bold green]Immediate Summarization - {now.strftime('%Y-%m-%d %H:%M:%S')}[/bold green]")
                run_summarizer(batch_size)
                last_summarize = datetime.now()
        
        # Run summarizer if it's time
        elif summarize_elapsed >= summarize_interval:
            console.rule(f"[bold green]Scheduled Summarization - {now.strftime('%Y-%m-%d %H:%M:%S')}[/bold green]")
            run_summarizer(batch_size)
            last_summarize = datetime.now()
        
        # Sleep for a minute before checking again
        await asyncio.sleep(60)
        
        # Show status
        next_download = download_interval - (datetime.now() - last_download).total_seconds() / 60
        next_summarize = summarize_interval - (datetime.now() - last_summarize).total_seconds() / 60
        log.info(f"Next download in {next_download:.1f} minutes, next summarization in {next_summarize:.1f} minutes")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the news pipeline")
    parser.add_argument("--download-interval", type=int, default=30, 
                        help="Minutes between RSS feed downloads")
    parser.add_argument("--summarize-interval", type=int, default=15, 
                        help="Minutes between summarization runs")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of articles to summarize per batch")
    return parser.parse_args()


async def main() -> None:
    """Main entry point"""
    args = parse_args()
    
    console.print(f"""[bold]News Pipeline Starting[/bold]
Download interval: {args.download_interval} minutes
Summarize interval: {args.summarize_interval} minutes
Batch size: {args.batch_size} articles
Press Ctrl+C to stop
""")
    
    try:
        # Run initial download and summarization
        console.rule("[bold blue]Initial RSS Download[/bold blue]")
        await run_downloader()
        
        console.rule("[bold green]Initial Summarization[/bold green]")
        run_summarizer(args.batch_size)
        
        # Then enter the loop
        await run_pipeline_iteration(args.download_interval, args.summarize_interval, args.batch_size)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Pipeline stopped by user[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Pipeline error: {str(e)}[/bold red]")
        raise


if __name__ == "__main__":
    # Set default event loop policy to avoid errors on Windows
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())