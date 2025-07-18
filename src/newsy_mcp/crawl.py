#! /usr/bin/env python3
import asyncio
import argparse
from crawl4ai import AsyncWebCrawler


async def main(url: str):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
        )
        print(result.markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl and extract content from a URL")
    parser.add_argument(
        "url", help="URL to crawl", default="https://www.nbcnews.com/business"
    )
    args = parser.parse_args()

    asyncio.run(main(args.url))
