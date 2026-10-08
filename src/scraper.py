"""
scraper.py - Extracts clean text content from any website.
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def fetch_website_text(url: str) -> str:
    """
    Fetch and extract clean text from a given URL.

    Args:
        url: Website URL (e.g., "https://example.com")

    Returns:
        Cleaned text content (str)

    Raises:
        RuntimeError: If the website cannot be fetched.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch website: {e}")

    soup = BeautifulSoup(response.text, "lxml")

    for tag in soup([
        "script", "style", "nav", "footer", "header",
        "aside", "noscript", "iframe", "form", "button",
        "svg", "img"
    ]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    clean_text = "\n".join(lines)

    return clean_text


def get_website_title(url: str) -> str:
    """
    Extract the <title> tag of a website.

    Used for displaying a friendly name in the UI.

    Args:
        url: Website URL

    Returns:
        Website title (str), or domain name if title is unavailable.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "lxml")
        title = soup.title.string if soup.title else urlparse(url).netloc
        return title.strip() if title else urlparse(url).netloc
    except Exception:
        return urlparse(url).netloc