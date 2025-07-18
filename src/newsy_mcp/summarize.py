#!/usr/bin/env python3
"""
Article Group Summarizer

Features:
1. Article Group Processing
   - Takes list of articles as input
   - Extracts text content and images
   - Configurable word limit for summary

2. Summary Generation
   - Uses OpenAI to create concise summary
   - Maintains key points from all articles
   - Respects word limit parameter

3. Image Selection
   - Downloads images from articles
   - Evaluates relevance to summary
   - Returns local path to best image
   - Cleans up unused images

4. Error Handling
   - Validates inputs
   - Handles API failures gracefully
   - Reports issues clearly
"""

import os
import tempfile
import requests
import sqlite3
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from bs4 import BeautifulSoup
from rich.console import Console
from utils import dprint
from content_util import analyze_image_relevance

# Constants
DEBUG = False
MODEL = "gpt-4o-mini"
IMAGE_DIR = "article_images"

console = Console()


def download_image(url: str, filename: str) -> Optional[str]:
    """Download image from URL to local file"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Create image directory if it doesn't exist
        os.makedirs(IMAGE_DIR, exist_ok=True)

        local_path = os.path.join(IMAGE_DIR, filename)
        with open(local_path, "wb") as f:
            f.write(response.content)

        return local_path
    except Exception as e:
        dprint(f"Failed to download image {url}: {e}")
        return None


def extract_images(articles: List[Dict]) -> List[Tuple[str, str]]:
    """Extract image URLs from articles"""
    images = []
    for article in articles:
        if "content_html" in article:
            soup = BeautifulSoup(article["content_html"], "html.parser")
            for img in soup.find_all("img"):
                src = img.get("src")
                if src:
                    images.append((src, article["title"]))
    return images


def generate_summary(articles: List[Dict], word_limit: int) -> str:
    """Generate summary of articles using OpenAI"""
    client = OpenAI()

    # Combine article content
    combined_content = "\n\n".join(
        f"Article: {a['title']}\n{a.get('content', '')}" for a in articles
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"Summarize these articles in {word_limit} words or less. Focus on key points and maintain coherence across articles.",
                },
                {"role": "user", "content": combined_content},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        console.print(f"[red]Failed to generate summary: {e}[/red]")
        return ""


def get_article_images(
    article_id: int, conn: sqlite3.Connection
) -> List[Dict[str, str]]:
    """Get all images associated with an article"""
    cursor = conn.execute(
        """
        SELECT url, alt_text, caption 
        FROM article_images 
        WHERE article_id = ?
        """,
        (article_id,),
    )
    return [
        {"url": row[0], "alt_text": row[1] or "", "caption": row[2] or ""}
        for row in cursor.fetchall()
    ]


def evaluate_image_relevance(
    summary: str, image_info: Dict[str, str], client: OpenAI
) -> float:
    """Evaluate how relevant an image is to the summary using vision capabilities"""
    try:
        # Download image and save to temp file
        response = requests.get(image_info["url"], timeout=10)
        response.raise_for_status()

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(response.content)
            temp_path = f.name

        try:
            # Use content_util's analyze_image_relevance
            is_relevant, _ = analyze_image_relevance(temp_path, summary, client)
            # Simple scoring - 1.0 if relevant, 0.2 if not
            return 1.0 if is_relevant else 0.2

        finally:
            # Clean up temp file
            os.unlink(temp_path)

    except Exception as e:
        dprint(f"Failed to evaluate image relevance: {e}")
        return 0.0


def summarize_articles(
    articles: List[Dict], word_limit: int = 150, num_images: int = 1
) -> Tuple[str, List[str]]:
    """Generate summary of articles and find most relevant images

    Args:
        articles: List of article dictionaries
        word_limit: Maximum number of words for summary
        num_images: Number of images to return

    Returns:
        Tuple of (summary text, list of image file paths)
    """
    if not articles:
        return "", []

    # Generate summary
    summary = generate_summary(articles, word_limit)
    if not summary:
        return "", []

    # Connect to database to get images
    conn = sqlite3.connect("newsy.db")
    try:
        client = OpenAI()
        image_scores = []  # List of (score, path) tuples
        images_needed = num_images

        # Process articles until we have enough images
        for article in articles:
            if images_needed <= 0:
                break

            images = get_article_images(article["id"], conn)
            dprint(f"Found {len(images)} images for article {article['title']}")

            # Only process as many images as we still need
            for image_info in images[:images_needed]:
                try:
                    # Download image and save to temp file
                    response = requests.get(image_info["url"], timeout=10)
                    response.raise_for_status()

                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                        f.write(response.content)
                        temp_path = f.name

                        score = evaluate_image_relevance(summary, image_info, client)
                        dprint(f"Image {image_info['url']} relevance score: {score}")
                        image_scores.append((score, temp_path))
                        images_needed -= 1

                        if images_needed <= 0:
                            break

                except Exception as e:
                    dprint(f"Error processing image {image_info['url']}: {str(e)}")
                    continue

        # Sort by score and return the paths
        image_scores.sort(reverse=True)
        best_image_paths = [path for _, path in image_scores]
        return summary, best_image_paths

    finally:
        conn.close()


def get_unsummarized_articles(limit: int = 10) -> List[Dict]:
    """Get articles from the database that haven't been summarized yet"""
    try:
        conn = sqlite3.connect("newsy.db")
        cursor = conn.cursor()
        
        # Select articles that don't have entries in article_analysis
        cursor.execute("""
            SELECT a.id, a.title, a.url, a.feed_domain, a.content_markdown
            FROM articles a
            LEFT JOIN article_analysis an ON a.id = an.article_id
            WHERE an.id IS NULL
            ORDER BY a.timestamp DESC
            LIMIT ?
        """, (limit,))
        
        articles = []
        for row in cursor.fetchall():
            articles.append({
                "id": row[0],
                "title": row[1],
                "url": row[2],
                "feed_domain": row[3],
                "content": row[4]
            })
        
        return articles
    except Exception as e:
        console.print(f"[red]Error fetching unsummarized articles: {e}[/red]")
        return []
    finally:
        conn.close()


def save_article_analysis(article_id: int, summary: str, tags: List[str] = None) -> bool:
    """Save analysis results to the database"""
    if not tags:
        tags = []
    
    try:
        conn = sqlite3.connect("newsy.db")
        cursor = conn.cursor()
        
        # Get article title and URL
        cursor.execute("SELECT title, url FROM articles WHERE id = ?", (article_id,))
        row = cursor.fetchone()
        if not row:
            console.print(f"[red]Article with ID {article_id} not found[/red]")
            return False
        
        title, url = row
        
        # Insert into article_analysis
        cursor.execute("""
            INSERT INTO article_analysis 
            (article_id, title, url, summary, tags)
            VALUES (?, ?, ?, ?, ?)
        """, (article_id, title, url, summary, ",".join(tags)))
        
        conn.commit()
        return True
    except Exception as e:
        console.print(f"[red]Error saving article analysis: {e}[/red]")
        return False
    finally:
        conn.close()


def extract_tags_from_summary(summary: str) -> List[str]:
    """Extract relevant tags from the summary using OpenAI"""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Extract 3-5 relevant tags from this article summary. Return only a comma-separated list of tags, nothing else."
                },
                {"role": "user", "content": summary}
            ]
        )
        
        # Parse comma-separated tags
        tags_text = response.choices[0].message.content
        tags = [tag.strip() for tag in tags_text.split(",")]
        return tags
    except Exception as e:
        console.print(f"[red]Failed to extract tags: {e}[/red]")
        return []


def main():
    """Process unsummarized articles"""
    console.print("[bold blue]Starting article summarization process...[/bold blue]")
    
    # Get unsummarized articles
    batch_size = 10
    articles = get_unsummarized_articles(batch_size)
    
    if not articles:
        console.print("[yellow]No unsummarized articles found.[/yellow]")
        return
    
    console.print(f"[green]Found {len(articles)} unsummarized articles[/green]")
    
    # Process each article individually
    for i, article in enumerate(articles):
        console.print(f"[bold]Processing article {i+1}/{len(articles)}: {article['title']}[/bold]")
        
        # Generate summary
        summary, _ = summarize_articles([article], word_limit=150)
        if not summary:
            console.print(f"[red]Failed to generate summary for article: {article['title']}[/red]")
            continue
        
        # Extract tags from summary
        tags = extract_tags_from_summary(summary)
        
        # Save analysis to database
        if save_article_analysis(article['id'], summary, tags):
            console.print(f"[green]Successfully processed article: {article['title']}[/green]")
            console.print(f"[blue]Summary: {summary}[/blue]")
            console.print(f"[blue]Tags: {', '.join(tags)}[/blue]")
        else:
            console.print(f"[red]Failed to save analysis for article: {article['title']}[/red]")
    
    console.print("[bold green]Article summarization process completed![/bold green]")


if __name__ == "__main__":
    main()
