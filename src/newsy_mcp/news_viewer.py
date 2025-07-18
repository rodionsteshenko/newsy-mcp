#!/usr/bin/env python3
"""
News Article Viewer

Features:
1. Article Index View
   - Shows articles from last 24 hours
   - Displays title and abbreviated URL
   - Numbered index for selection
   - Rich formatting for readability

2. Article Detail View
   - Full article title, URL, summary
   - Navigation commands (back, similar)
   - Rich panel display format

3. Similar Articles View
   - Uses embedding cosine similarity
   - Shows top 10 similar articles
   - Displays similarity scores
   - Allows selection to view details

4. Navigation
   - Back command ('b') returns to previous view
   - Similar command ('s') shows related articles
   - Index selection to view articles
   - Clean menu interface

5. Article Stats
   - Dump command (-d) shows article statistics
   - Counts by time range (24h, all time)
   - Tracks articles with summaries and embeddings
   - Helps diagnose data pipeline issues

6. Similarity Analysis
   - Matrix view of similarities (-a N)
   - Similar pair finder (-s threshold)
   - Configurable similarity threshold
   - Displays article pairs above threshold
   - Shows titles and URLs for context

7. Article Clustering
   - Groups similar articles into clusters (-c threshold)
   - Agglomerative clustering approach
   - Shows cluster sizes and average similarity
   - Lists articles within each cluster
   - Sorted by timestamp within clusters
"""

import sqlite3
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
import numpy as np
from typing import List, Dict, Tuple
import argparse
import traceback

# Constants
DEBUG = False
HOURS_WINDOW = 24
MAX_SIMILAR = 10
MAX_ARTICLES = 100  # Increased from default

console = Console()


def init_database() -> None:
    """Initialize database tables if they don't exist"""
    try:
        from db_init import init_database as init_db

        init_db()
        console.print("[green]Database tables initialized[/green]")
    except Exception as e:
        console.print(f"[red]Error initializing database: {str(e)}[/red]")
        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")


def dump_article_stats() -> None:
    """Display article statistics and pipeline status"""
    conn = sqlite3.connect("newsy.db")
    try:
        # Get current time for 24h window
        cutoff = (datetime.now() - timedelta(hours=HOURS_WINDOW)).isoformat()

        # Get total article counts
        cursor = conn.execute("SELECT COUNT(*) FROM articles")
        total_articles = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT COUNT(*) FROM articles WHERE timestamp > ?", (cutoff,)
        )
        recent_articles = cursor.fetchone()[0]

        # Get summary counts
        cursor = conn.execute("SELECT COUNT(*) FROM article_analysis")
        total_summaries = cursor.fetchone()[0]

        cursor = conn.execute(
            """
            SELECT COUNT(*)
            FROM articles a
            JOIN article_analysis an ON an.article_id = a.id
            WHERE a.timestamp > ?
            """,
            (cutoff,),
        )
        recent_summaries = cursor.fetchone()[0]

        # Get embedding counts
        cursor = conn.execute("SELECT COUNT(*) FROM article_embeddings")
        total_embeddings = cursor.fetchone()[0]

        cursor = conn.execute(
            """
            SELECT COUNT(*)
            FROM articles a
            JOIN article_embeddings ae ON ae.article_id = a.id
            WHERE a.timestamp > ?
            """,
            (cutoff,),
        )
        recent_embeddings = cursor.fetchone()[0]

        # Display stats
        console.print("\n[bold cyan]Article Statistics[/bold cyan]")

        console.print("\n[bold]Total Articles[/bold]")
        console.print(f"Last {HOURS_WINDOW} hours: {recent_articles} articles")
        console.print(f"All time: {total_articles} articles")

        console.print("\n[bold]Processing Pipeline[/bold]")
        console.print(
            f"Articles with summaries (24h): {recent_summaries}/{recent_articles} articles ({recent_summaries / recent_articles * 100:.1f}% coverage)"
        )
        console.print(
            f"Articles with summaries (all): {total_summaries}/{total_articles} articles ({total_summaries / total_articles * 100:.1f}% coverage)"
        )
        console.print(
            f"Articles with embeddings (24h): {recent_embeddings}/{recent_articles} articles ({recent_embeddings / recent_articles * 100:.1f}% coverage)"
        )
        console.print(
            f"Articles with embeddings (all): {total_embeddings}/{total_articles} articles ({total_embeddings / total_articles * 100:.1f}% coverage)"
        )

    except Exception as e:
        console.print(f"[red]Error getting article stats: {e}[/red]")
    finally:
        conn.close()


def get_recent_articles() -> List[Dict]:
    """Get articles from last 24 hours"""
    conn = sqlite3.connect("newsy.db")
    cutoff = (datetime.now() - timedelta(hours=HOURS_WINDOW)).isoformat()

    cursor = conn.execute(
        """
        SELECT 
            a.id, 
            a.title, 
            a.url, 
            an.summary, 
            COALESCE(ae.id, 0) as embedding_id,
            (SELECT COUNT(*) FROM article_images ai WHERE ai.article_id = a.id) as image_count
        FROM articles a
        JOIN article_analysis an ON an.article_id = a.id
        LEFT JOIN article_embeddings ae ON ae.article_id = a.id
        WHERE a.timestamp > ?
        ORDER BY a.timestamp DESC
        LIMIT ?
        """,
        (cutoff, MAX_ARTICLES),
    )

    articles = [
        {
            "id": row[0],
            "title": row[1],
            "url": row[2],
            "summary": row[3],
            "embedding_id": row[4],
            "image_count": row[5],
        }
        for row in cursor.fetchall()
    ]
    conn.close()
    return articles


def show_article_index(articles: List[Dict]) -> None:
    """Display table of recent articles"""
    table = Table(title=f"Articles from last {HOURS_WINDOW} hours")
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Title", style="green")
    table.add_column("URL", style="blue")
    table.add_column("Images", style="magenta", justify="right")

    for i, article in enumerate(articles, 1):
        url = (
            article["url"][:50] + "..." if len(article["url"]) > 50 else article["url"]
        )
        table.add_row(str(i), article["title"], url, str(article["image_count"]))

    console.print(table)


def get_article_images(
    article_id: int, conn: sqlite3.Connection
) -> List[Dict[str, str]]:
    """Get images associated with an article"""
    cursor = conn.execute(
        """
        SELECT url, alt_text, caption 
        FROM article_images 
        WHERE article_id = ?
        ORDER BY id
        """,
        (article_id,),
    )
    return [
        {"url": row[0], "alt_text": row[1] or "", "caption": row[2] or ""}
        for row in cursor.fetchall()
    ]


def show_article_detail(article: Dict) -> None:
    """Display full article details"""
    # Get images for this article
    conn = sqlite3.connect("newsy.db")
    try:
        images = get_article_images(article["id"], conn)

        # Build content string with images
        content = (
            f"[cyan]Title:[/cyan] {article['title']}\n\n"
            f"[cyan]URL:[/cyan] {article['url']}\n\n"
        )

        # Add images section if any exist
        if images:
            content += "[cyan]Images:[/cyan]\n"
            for i, img in enumerate(images, 1):
                content += f"{i}. {img['url']}\n"
                if img["alt_text"]:
                    content += f"   Alt: {img['alt_text']}\n"
                if img["caption"]:
                    content += f"   Caption: {img['caption']}\n"
            content += "\n"

        content += f"[cyan]Summary:[/cyan]\n{article['summary']}"

        console.print(Panel(content, title="Article Detail", border_style="green"))

    finally:
        conn.close()


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Normalizes vectors first if they aren't already normalized.
    """
    # Ensure vectors are 2D
    if len(vec1.shape) == 1:
        vec1 = vec1.reshape(1, -1)
    if len(vec2.shape) == 1:
        vec2 = vec2.reshape(1, -1)

    # Normalize vectors
    vec1_norm = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
    vec2_norm = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)

    # Compute similarity
    return float(np.dot(vec1_norm, vec2_norm.T)[0, 0])


def get_similar_articles(
    article_id: int, embedding_id: int
) -> List[Tuple[Dict, float]]:
    """Find similar articles using direct similarity computation"""
    if embedding_id == 0:
        console.print(
            "[yellow]Similarity search not available for this article (no embedding)[/yellow]"
        )
        return []

    try:
        conn = sqlite3.connect("newsy.db")

        # Get the query embedding
        cursor = conn.execute(
            "SELECT embedding FROM article_embeddings WHERE id = ?", (embedding_id,)
        )
        result = cursor.fetchone()
        if not result:
            return []

        query_vector = np.frombuffer(result[0], dtype=np.float32)

        # Get all other articles with embeddings
        cursor = conn.execute(
            """
            SELECT a.id, a.title, a.url, an.summary, ae.id, ae.embedding
            FROM articles a
            JOIN article_analysis an ON an.article_id = a.id
            JOIN article_embeddings ae ON ae.article_id = a.id
            WHERE a.id != ?
            ORDER BY a.timestamp DESC
            """,
            (article_id,),
        )

        articles = []
        for row in cursor.fetchall():
            article = {
                "id": row[0],
                "title": row[1],
                "url": row[2],
                "summary": row[3],
                "embedding_id": row[4],
            }
            embedding = np.frombuffer(row[5], dtype=np.float32)
            similarity = compute_cosine_similarity(query_vector, embedding)
            articles.append((article, similarity))

        # Sort by similarity and return top MAX_SIMILAR
        return sorted(articles, key=lambda x: x[1], reverse=True)[:MAX_SIMILAR]

    except Exception as e:
        console.print(f"[red]Error finding similar articles: {e}[/red]")
        return []
    finally:
        if "conn" in locals():
            conn.close()


def show_similar_articles(similar: List[Tuple[Dict, float]]) -> None:
    """Display table of similar articles with scores"""
    table = Table(title="Similar Articles")
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Title", style="green")
    table.add_column("Similarity", style="magenta", justify="right")

    for i, (article, similarity) in enumerate(similar, 1):
        table.add_row(str(i), article["title"], f"{similarity:.2f}")

    console.print(table)


def analyze_similarities(n_articles: int = 10) -> None:
    """
    Analyze and display similarity scores between the first N articles.
    Shows a matrix of cosine similarities between article embeddings.
    """
    conn = sqlite3.connect("newsy.db")
    try:
        # Get the first N articles with embeddings
        cursor = conn.execute(
            """
            SELECT a.id, a.title, ae.id as embedding_id, ae.embedding 
            FROM articles a
            JOIN article_embeddings ae ON ae.article_id = a.id
            ORDER BY a.timestamp DESC
            LIMIT ?
            """,
            (n_articles,),
        )

        articles = []
        embeddings = []
        for row in cursor.fetchall():
            articles.append(
                {
                    "id": row[0],
                    "title": row[1][:50] + "..."
                    if len(row[1]) > 50
                    else row[1],  # Truncate long titles
                    "embedding_id": row[2],
                }
            )
            # Convert blob to numpy array
            embedding = np.frombuffer(row[3], dtype=np.float32)
            embeddings.append(embedding)

        if not articles:
            console.print("[yellow]No articles with embeddings found[/yellow]")
            return

        # Convert to numpy array
        embeddings_matrix = np.vstack(embeddings)

        # Compute all pairwise similarities
        similarities = np.zeros((len(articles), len(articles)))
        for i in range(len(articles)):
            for j in range(len(articles)):
                if i != j:  # Skip diagonal
                    similarities[i, j] = compute_cosine_similarity(
                        embeddings_matrix[i], embeddings_matrix[j]
                    )

        # Create table
        table = Table(title=f"Similarity Matrix for {len(articles)} Articles")

        # Add header row with article numbers
        table.add_column("Article", style="cyan")
        for i in range(len(articles)):
            table.add_column(f"#{i + 1}", justify="right")

        # Add rows with similarities
        for i, article in enumerate(articles):
            row = [f"#{i + 1} {article['title']}"]
            for j in range(len(articles)):
                if i == j:  # Skip diagonal
                    row.append("-")
                else:
                    sim = similarities[i][j]
                    # Updated color scheme
                    color = (
                        "green"
                        if sim > 0.8
                        else "blue"
                        if sim > 0.75
                        else "yellow"
                        if sim > 0.7
                        else "red"
                    )
                    row.append(f"[{color}]{sim:.2f}[/{color}]")
            table.add_row(*row)

        console.print(table)

        # Print article mapping for reference
        console.print("\nArticle Reference:")
        for i, article in enumerate(articles, 1):
            console.print(f"#{i}: {article['title']}")

    except Exception as e:
        console.print(f"[red]Error analyzing similarities: {e}[/red]")
    finally:
        conn.close()


def find_similar_pairs(threshold: float) -> None:
    """
    Find and display pairs of articles with similarity above threshold.
    Compares each article against all others that come after it.
    """
    conn = sqlite3.connect("newsy.db")
    try:
        # Get all articles with embeddings
        cursor = conn.execute(
            """
            SELECT a.id, a.title, a.url, ae.id as embedding_id, ae.embedding 
            FROM articles a
            JOIN article_embeddings ae ON ae.article_id = a.id
            ORDER BY a.timestamp DESC
            """
        )

        articles = []
        embeddings = []
        for row in cursor.fetchall():
            articles.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "url": row[2],
                    "embedding_id": row[3],
                }
            )
            embeddings.append(np.frombuffer(row[4], dtype=np.float32))

        if not articles:
            console.print("[yellow]No articles with embeddings found[/yellow]")
            return

        console.print(
            f"\n[bold cyan]Finding article pairs with similarity > {threshold}[/bold cyan]\n"
        )

        # Compare each article with all later articles
        for i in range(len(articles) - 1):
            for j in range(i + 1, len(articles)):
                similarity = compute_cosine_similarity(embeddings[i], embeddings[j])

                if similarity > threshold:
                    # Display the similar pair
                    console.print(
                        f"[bold green]Similarity: {similarity:.3f}[/bold green]"
                    )
                    console.print("\n[cyan]Article 1:[/cyan]")
                    console.print(f"Title: {articles[i]['title']}")
                    console.print(f"URL: {articles[i]['url']}\n")

                    console.print("[cyan]Article 2:[/cyan]")
                    console.print(f"Title: {articles[j]['title']}")
                    console.print(f"URL: {articles[j]['url']}")
                    console.print("\n" + "-" * 80 + "\n")

    except Exception as e:
        console.print(f"[red]Error finding similar pairs: {e}[/red]")
    finally:
        conn.close()


def find_article_clusters(threshold: float) -> None:
    """
    Group articles into clusters based on similarity threshold.
    Uses a simple agglomerative clustering approach.
    """
    conn = sqlite3.connect("newsy.db")
    try:
        # Get all articles with embeddings
        cursor = conn.execute(
            """
            SELECT a.id, a.title, a.url, ae.id as embedding_id, ae.embedding, a.timestamp
            FROM articles a
            JOIN article_embeddings ae ON ae.article_id = a.id
            ORDER BY a.timestamp DESC
            """
        )

        articles = []
        embeddings = []
        for row in cursor.fetchall():
            articles.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "url": row[2],
                    "embedding_id": row[3],
                    "timestamp": row[5],
                }
            )
            embeddings.append(np.frombuffer(row[4], dtype=np.float32))

        if not articles:
            console.print("[yellow]No articles with embeddings found[/yellow]")
            return

        console.print(
            f"\n[bold cyan]Clustering articles with similarity > {threshold}[/bold cyan]\n"
        )

        # Initialize clusters
        clusters = []
        used_articles = set()

        # Find clusters
        for i in range(len(articles)):
            if i in used_articles:
                continue

            # Start a new cluster
            cluster = {i}
            used_articles.add(i)

            # Find all similar articles for this cluster
            changed = True
            while changed:
                changed = False
                for j in range(len(articles)):
                    if j in used_articles:
                        continue

                    # Check similarity with all articles in current cluster
                    for cluster_idx in cluster:
                        similarity = compute_cosine_similarity(
                            embeddings[cluster_idx], embeddings[j]
                        )
                        if similarity > threshold:
                            cluster.add(j)
                            used_articles.add(j)
                            changed = True
                            break

            if len(cluster) > 1:  # Only keep clusters with multiple articles
                clusters.append(cluster)

        # Display clusters
        if not clusters:
            console.print(
                "[yellow]No article clusters found above similarity threshold[/yellow]"
            )
            return

        for idx, cluster in enumerate(clusters, 1):
            console.print(
                f"\n[bold green]Cluster #{idx} ({len(cluster)} articles)[/bold green]"
            )

            # Calculate average similarity within cluster
            similarities = []
            for i in cluster:
                for j in cluster:
                    if i < j:
                        sim = compute_cosine_similarity(embeddings[i], embeddings[j])
                        similarities.append(sim)
            avg_similarity = (
                sum(similarities) / len(similarities) if similarities else 0
            )

            console.print(f"[blue]Average similarity: {avg_similarity:.3f}[/blue]\n")

            # Sort cluster articles by timestamp
            cluster_articles = sorted(
                [articles[i] for i in cluster],
                key=lambda x: x["timestamp"],
                reverse=True,
            )

            # Display articles in cluster
            for article in cluster_articles:
                console.print(f"[cyan]Title:[/cyan] {article['title']}")
                console.print(f"[cyan]URL:[/cyan] {article['url']}")
                console.print()

            console.print("-" * 80)

    except Exception as e:
        console.print(f"[red]Error finding article clusters: {e}[/red]")
    finally:
        conn.close()


def get_articles_by_tag_and_similarity(
    tags: List[str], min_similarity: float = 0.75, hours: int = 24
) -> List[Dict]:
    """Get unread articles matching tags above similarity threshold, grouped by largest cluster.

    Args:
        tags: List of tags to filter articles by
        min_similarity: Minimum similarity threshold (default 0.75)
        hours: Time window in hours (default 24)

    Returns:
        List of articles in largest similar cluster
    """
    try:
        console.print(
            f"\n[bold cyan]Searching for articles with tags: {tags}[/bold cyan]"
        )
        console.print(f"[cyan]Minimum similarity threshold: {min_similarity}[/cyan]")
        console.print(f"[cyan]Time window: {hours} hours[/cyan]\n")

        conn = sqlite3.connect("newsy.db")

        # Calculate cutoff time
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        # Build tag conditions
        tag_conditions = []
        params = [cutoff]  # First param is cutoff time
        for tag in tags:
            tag_conditions.append("an.tags LIKE ?")
            params.append(f"%{tag}%")

        console.print("[yellow]Executing database query...[/yellow]")

        # Get unread articles within time window
        query = """
            SELECT 
                a.id,
                a.title, 
                a.url,
                an.summary,
                a.content,
                ae.embedding
            FROM articles a
            JOIN article_analysis an ON an.article_id = a.id 
            JOIN article_embeddings ae ON ae.article_id = a.id
            WHERE an.relevance > 0.3
            AND a.timestamp > ?
            AND ({})
            AND NOT EXISTS (
                SELECT 1 FROM sent_articles sa 
                WHERE sa.article_id = a.id
            )
            ORDER BY a.timestamp DESC
        """.format(" OR ".join(tag_conditions))

        cursor = conn.execute(query, params)
        articles = []
        embeddings = []

        for row in cursor.fetchall():
            articles.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "url": row[2],
                    "summary": row[3],
                    "content": row[4],
                }
            )
            embeddings.append(np.frombuffer(row[5], dtype=np.float32))

        console.print(f"[green]Found {len(articles)} matching articles[/green]")

        if not articles:
            console.print("[yellow]No articles found matching criteria[/yellow]")
            return []

        console.print("\n[yellow]Computing similarity clusters...[/yellow]")

        # Convert embeddings to numpy array
        embeddings_matrix = np.vstack(embeddings)

        # Find clusters
        clusters = []
        used_articles = set()

        for i in range(len(articles)):
            if i in used_articles:
                continue

            # Start new cluster
            cluster = {i}
            used_articles.add(i)

            console.print(
                f"\n[blue]Building cluster starting with article {i + 1}:[/blue]"
            )
            console.print(f"Title: {articles[i]['title']}")

            # Find similar articles
            changed = True
            while changed:
                changed = False
                for j in range(len(articles)):
                    if j in used_articles:
                        continue

                    # Check similarity
                    for cluster_idx in cluster:
                        similarity = compute_cosine_similarity(
                            embeddings_matrix[cluster_idx], embeddings_matrix[j]
                        )

                        if similarity > min_similarity:
                            cluster.add(j)
                            used_articles.add(j)
                            changed = True
                            console.print(
                                f"[green]Added article {j + 1} to cluster (similarity: {similarity:.3f})[/green]"
                            )
                            console.print(f"Title: {articles[j]['title']}")
                            break

            if len(cluster) > 1:
                clusters.append(cluster)
                console.print(
                    f"[cyan]Completed cluster with {len(cluster)} articles[/cyan]"
                )

        console.print(f"\n[bold green]Found {len(clusters)} clusters[/bold green]")

        # Find largest cluster
        if not clusters:
            console.print(
                "[yellow]No clusters found above similarity threshold[/yellow]"
            )
            return []

        largest_cluster = max(clusters, key=len)
        console.print(
            f"\n[bold green]Largest cluster has {len(largest_cluster)} articles:[/bold green]"
        )

        # Return articles from largest cluster
        cluster_articles = [articles[i] for i in largest_cluster]

        # Print cluster details
        for i, article in enumerate(cluster_articles, 1):
            console.print(f"\n[cyan]Article {i}:[/cyan]")
            console.print(f"Title: {article['title']}")
            console.print(f"URL: {article['url']}")

        return cluster_articles

    except Exception as e:
        console.print(
            f"[red]Error getting articles by tag and similarity: {str(e)}[/red]"
        )
        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        return []
    finally:
        conn.close()


def mark_articles_as_read(article_ids: List[int]) -> None:
    """Mark articles as read so they won't be returned in future queries.

    Args:
        article_ids: List of article IDs to mark as read
    """
    try:
        conn = sqlite3.connect("newsy.db")

        # Insert articles into sent_articles table
        for article_id in article_ids:
            conn.execute(
                """
                INSERT OR IGNORE INTO sent_articles 
                (article_id, sent_time)
                VALUES (?, CURRENT_TIMESTAMP)
            """,
                (article_id,),
            )

        conn.commit()
        console.print(f"Marked {len(article_ids)} articles as read")

    except Exception as e:
        console.print(f"Error marking articles as read: {str(e)}", error=True)
    finally:
        conn.close()


def main() -> None:
    """Main viewer interface"""
    # Initialize database tables
    init_database()

    parser = argparse.ArgumentParser(description="News Article Viewer")
    parser.add_argument(
        "--analyze",
        "-a",
        type=int,
        metavar="N",
        help="Analyze similarities between first N articles",
    )
    parser.add_argument(
        "--dump", "-d", action="store_true", help="Dump article statistics"
    )
    parser.add_argument(
        "--similar",
        "-s",
        type=float,
        metavar="THRESHOLD",
        help="Find article pairs with similarity above threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--cluster",
        "-c",
        type=float,
        metavar="THRESHOLD",
        help="Cluster articles with similarity above threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--tag-search", "-t", nargs="+", metavar="TAG", help="Search articles by tags"
    )
    parser.add_argument(
        "--similarity",
        "-m",
        type=float,
        default=0.75,
        help="Minimum similarity threshold for tag search (default: 0.75)",
    )
    parser.add_argument(
        "--hours",
        "-r",
        type=int,
        default=24,
        help="Time window in hours for tag search (default: 24)",
    )
    parser.add_argument(
        "--mark-read", action="store_true", help="Mark found articles as read"
    )
    args = parser.parse_args()

    if args.analyze:
        analyze_similarities(args.analyze)
        return

    if args.dump:
        dump_article_stats()
        return

    if args.similar is not None:
        if not 0 <= args.similar <= 1:
            console.print("[red]Threshold must be between 0.0 and 1.0[/red]")
            return
        find_similar_pairs(args.similar)
        return

    if args.cluster is not None:
        if not 0 <= args.cluster <= 1:
            console.print("[red]Threshold must be between 0.0 and 1.0[/red]")
            return
        find_article_clusters(args.cluster)
        return

    if args.tag_search:
        if not 0 <= args.similarity <= 1:
            console.print("[red]Similarity threshold must be between 0.0 and 1.0[/red]")
            return

        articles = get_articles_by_tag_and_similarity(
            args.tag_search, args.similarity, args.hours
        )

        if articles and args.mark_read:
            article_ids = [a["id"] for a in articles]
            mark_articles_as_read(article_ids)
            console.print("\n[green]Marked articles as read[/green]")

        return

    while True:
        # Get and show article index
        articles = get_recent_articles()
        show_article_index(articles)

        # Get user selection
        choice = Prompt.ask(
            "\nEnter article number to view, or 'q' to quit",
            choices=[str(i) for i in range(1, len(articles) + 1)] + ["q"],
        )

        if choice == "q":
            break

        # Show article detail view
        while True:
            article = articles[int(choice) - 1]
            show_article_detail(article)

            cmd = Prompt.ask(
                "\nCommands: [b]ack, [s]imilar, or [q]uit", choices=["b", "s", "q"]
            )

            if cmd == "b":
                break
            elif cmd == "q":
                return
            elif cmd == "s":
                # Show similar articles
                similar = get_similar_articles(article["id"], article["embedding_id"])
                if similar:
                    show_similar_articles(similar)
                    sim_choice = Prompt.ask(
                        "\nEnter article number to view, or 'b' to go back",
                        choices=[str(i) for i in range(1, len(similar) + 1)] + ["b"],
                    )
                    if sim_choice != "b":
                        article = similar[int(sim_choice) - 1][0]
                else:
                    console.print("[yellow]No similar articles found[/yellow]")


if __name__ == "__main__":
    main()
