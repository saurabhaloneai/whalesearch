import argparse
import asyncio
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
import logging
import aiohttp
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def parse_feed_async(feed_url):
    try:
        feed = await asyncio.to_thread(feedparser.parse, feed_url)
        return [entry.link for entry in feed.entries]
    except Exception as e:
        logger.error(f"Error parsing feed {feed_url}: {e}")
        return []


async def fetch_content_async(session, url):
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        logger.error(f"Error fetching content from {url}: {e}")
        return None


async def process_feed_async(feed_url, session, executor):
    try:
        post_urls = await parse_feed_async(feed_url)
        tasks = [fetch_content_async(session, post_url) for post_url in post_urls]
        post_contents = await asyncio.gather(*tasks)
        cleaned_contents = await asyncio.gather(*[loop.run_in_executor(executor, clean_content, content) for content in post_contents if content])
        return list(zip(post_urls, cleaned_contents))
    except Exception as e:
        logger.error(f"Error processing feed {feed_url}: {e}")
        return []


def clean_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    cleaned_text = " ".join(chunk for chunk in chunks if chunk)
    return cleaned_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed-path")
    return parser.parse_args()


async def main_async(feed_file):
    async with aiohttp.ClientSession() as session:
        with open(feed_file, "r") as file:
            feed_urls = [line.strip() for line in file]

        with ThreadPoolExecutor() as executor:
            tasks = [process_feed_async(feed_url, session, executor) for feed_url in feed_urls]
            results = await asyncio.gather(*tasks)

    flattened_results = [item for sublist in results for item in sublist]
    df = pd.DataFrame(flattened_results, columns=["URL", "content"])
    df.to_parquet("output.parquet", index=False)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args.feed_path))
