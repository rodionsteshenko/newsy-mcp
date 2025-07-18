#!/usr/bin/env python3
"""
Article Dump CLI Viewer

Features:
1. Interactive Article Browsing
   - Filter by tags and time range
   - Sort by timestamp
   - Paginated article listing
   - Detailed article view

2. Rich Console Interface
   - Colorful formatting
   - Interactive menus
   - Progress indicators
   - Error handling

3. Database Integration
   - SQLite queries
   - Cached results
   - Efficient filtering

4. Time Range Support
   - Flexible time window selection
   - Default ranges (24h, 7d, 30d)
   - Custom range input
"""

import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rss_config import VALID_TAGS

# Debug flag
DEBUG = True

console = Console()


class ArticleViewer:
    def __init__(self, db_path: str = "newsy.db"):
        self.db_path = db_path
        self.time_ranges = {
            "1": ("Last 24 hours", timedelta(hours=24)),
            "2": ("Last 7 days", timedelta(days=7)),
            "3": ("Last 30 days", timedelta(days=30)),
            "4": ("Custom range", None),
        }
        self.relevance_filters = {
            "0": ("All articles", None),
            "1": ("Relevant (score = 1)", 1),
            "2": ("Maybe relevant (score = 0)", 0),
            "3": ("Not relevant (score = -1)", -1),
        }

    def get_db_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_articles(
        self, tags: List[str], start_time: datetime, relevance_score: int | None = None
    ) -> List[Dict]:
        conn = self.get_db_connection()

        # Build query based on selected tags and relevance
        query = """
            SELECT a.*, an.tags, an.summary, an.relevance
            FROM articles a
            LEFT JOIN article_analysis an ON a.id = an.article_id
            WHERE a.timestamp > ?
        """
        params = [start_time.isoformat()]

        if tags:
            tag_conditions = []
            for _ in tags:
                tag_conditions.append("an.tags LIKE ?")
            query += f" AND ({' OR '.join(tag_conditions)})"
            params.extend(f"%{tag}%" for tag in tags)

        if relevance_score is not None:
            query += " AND an.relevance = ?"
            params.append(relevance_score)

        query += " ORDER BY a.timestamp DESC"

        cursor = conn.execute(query, params)
        articles = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return articles

    def display_article(self, article: Dict) -> None:
        """Display full article details"""
        try:
            timestamp = datetime.fromisoformat(article["timestamp"])
            local_time = timestamp.astimezone()

            content = Panel(
                f"[cyan]Title:[/cyan] {article['title']}\n\n"
                f"[cyan]Published:[/cyan] {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n"
                f"[cyan]URL:[/cyan] {article['url']}\n\n"
                f"[cyan]Tags:[/cyan] {article.get('tags', 'No tags')}\n\n"
                f"[cyan]Summary:[/cyan]\n{article.get('summary', 'No summary available')}\n\n"
                f"[cyan]Content:[/cyan]\n{article.get('content_markdown', article.get('content_html2text', 'No content available'))}",
                title="Article Details",
                border_style="blue",
            )
            console.print(content)
        except Exception as e:
            console.print(f"[red]Error displaying article: {e}[/red]")

    def select_time_range(self) -> datetime:
        """Interactive time range selection"""
        console.print("\n[cyan]Select time range:[/cyan]")
        for key, (desc, _) in self.time_ranges.items():
            console.print(f"{key}. {desc}")

        choice = Prompt.ask("Enter choice", choices=list(self.time_ranges.keys()))

        if choice == "4":
            days = IntPrompt.ask("Enter number of days to look back", default=7)
            return datetime.now() - timedelta(days=days)
        else:
            _, delta = self.time_ranges[choice]
            return datetime.now() - delta

    def select_tags(self) -> List[str]:
        """Interactive tag selection"""
        console.print("\n[cyan]Available tags:[/cyan]")
        for i, tag in enumerate(VALID_TAGS, 1):
            console.print(f"{i}. {tag}")
        console.print("0. Done selecting")

        selected_tags = []
        while True:
            choice = IntPrompt.ask("Select tag number (0 to finish)", default=0)
            if choice == 0:
                break
            if 1 <= choice <= len(VALID_TAGS):
                tag = VALID_TAGS[choice - 1]
                if tag not in selected_tags:
                    selected_tags.append(tag)
                    console.print(f"[green]Added {tag}[/green]")
            else:
                console.print("[red]Invalid selection[/red]")

        return selected_tags

    def select_relevance_filter(self) -> int | None:
        """Interactive relevance score selection"""
        console.print("\n[cyan]Select relevance filter:[/cyan]")
        for key, (desc, _) in self.relevance_filters.items():
            console.print(f"{key}. {desc}")

        choice = Prompt.ask("Enter choice", choices=list(self.relevance_filters.keys()))
        return self.relevance_filters[choice][1]

    def display_articles(self, articles: List[Dict]) -> None:
        """Display articles in a table"""
        table = Table(title="Articles")
        table.add_column("Index", style="cyan", justify="right")
        table.add_column("Timestamp", style="magenta")
        table.add_column("Title", style="green")
        table.add_column("Tags", style="yellow")
        table.add_column("Relevance", style="blue")

        for i, article in enumerate(articles, 1):
            try:
                timestamp = datetime.fromisoformat(article["timestamp"])
                local_time = timestamp.astimezone()
                relevance = article.get("relevance", "N/A")
                if relevance == 1:
                    relevance_text = "✓"
                elif relevance == 0:
                    relevance_text = "?"
                elif relevance == -1:
                    relevance_text = "✗"
                else:
                    relevance_text = "N/A"

                table.add_row(
                    str(i),
                    local_time.strftime("%Y-%m-%d %H:%M"),
                    article["title"],
                    article.get("tags", "No tags"),
                    relevance_text,
                )
            except Exception as e:
                console.print(f"[red]Error displaying article {i}: {e}[/red]")

        console.print(table)

    def run(self) -> None:
        """Main interaction loop"""
        while True:
            console.clear()
            console.print("[bold cyan]Article Browser[/bold cyan]")

            # Get filters
            tags = self.select_tags()
            start_time = self.select_time_range()
            relevance_score = self.select_relevance_filter()

            # Fetch articles
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Fetching articles...", total=1)
                articles = self.get_articles(tags, start_time, relevance_score)
                progress.advance(task)

            if not articles:
                console.print("[yellow]No articles found matching criteria[/yellow]")
                if not Prompt.ask("Try again?", choices=["y", "n"], default="y") == "y":
                    break
                continue

            # Display articles
            self.display_articles(articles)

            # Article selection loop
            while True:
                choice = IntPrompt.ask(
                    "\nEnter article number to view (0 to start over)", default=0
                )
                if choice == 0:
                    break
                if 1 <= choice <= len(articles):
                    self.display_article(articles[choice - 1])
                    console.print("\nPress Enter to continue...")
                    input()
                else:
                    console.print("[red]Invalid article number[/red]")

            if (
                not Prompt.ask("Continue browsing?", choices=["y", "n"], default="y")
                == "y"
            ):
                break


def main() -> None:
    viewer = ArticleViewer()
    viewer.run()


if __name__ == "__main__":
    main()
