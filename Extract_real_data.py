
import json
import logging
import time
import requests
import feedparser
from pathlib import Path
from typing import Dict, List, Set, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# ====================
# CONFIGURATION
# ====================

RSS_SOURCES = {
    "VietnamPlus": [
        "https://www.vietnamplus.vn/rss/thoi-su.rss",
        "https://www.vietnamplus.vn/rss/xa-hoi.rss",
        "https://www.vietnamplus.vn/rss/kinh-te.rss",
        "https://www.vietnamplus.vn/rss/the-gioi.rss",
        "https://www.vietnamplus.vn/rss/phap-luat.rss",
        "https://www.vietnamplus.vn/rss/doi-song.rss",
        "https://www.vietnamplus.vn/rss/van-hoa.rss",
        "https://www.vietnamplus.vn/rss/cong-nghe.rss",
    ],
    "VnExpress": [
        "https://vnexpress.net/rss/tin-moi-nhat.rss",
        "https://vnexpress.net/rss/thoi-su.rss",
        "https://vnexpress.net/rss/the-gioi.rss",
        "https://vnexpress.net/rss/kinh-doanh.rss",
        "https://vnexpress.net/rss/phap-luat.rss",
        "https://vnexpress.net/rss/giao-duc.rss",
        "https://vnexpress.net/rss/suc-khoe.rss",
        "https://vnexpress.net/rss/doi-song.rss",
    ],
    "VTVNews": [
        "https://vtv.vn/rss/trang-chu.rss",
        "https://vtv.vn/rss/thoi-su.rss",
        "https://vtv.vn/rss/xa-hoi.rss",
        "https://vtv.vn/rss/the-gioi.rss",
        "https://vtv.vn/rss/kinh-te.rss",
        "https://vtv.vn/rss/phap-luat.rss",
        "https://vtv.vn/rss/giao-duc.rss",
    ],
    "ThanhNien": [
        "https://thanhnien.vn/rss/thoi-su.rss",
        "https://thanhnien.vn/rss/the-gioi.rss",
        "https://thanhnien.vn/rss/kinh-te.rss",
        "https://thanhnien.vn/rss/doi-song.rss",
        "https://thanhnien.vn/rss/van-hoa.rss",
        "https://thanhnien.vn/rss/giai-tri.rss",
        "https://thanhnien.vn/rss/giao-duc.rss",
        "https://thanhnien.vn/rss/cong-nghe.rss",
    ],
    "TuoiTre": [
        "https://tuoitre.vn/rss/thoi-su.rss",
        "https://tuoitre.vn/rss/the-gioi.rss",
        "https://tuoitre.vn/rss/phap-luat.rss",
        "https://tuoitre.vn/rss/kinh-doanh.rss",
        "https://tuoitre.vn/rss/cong-nghe.rss",
        "https://tuoitre.vn/rss/van-hoa.rss",
    ],
    "DanTri": [
        "https://dantri.com.vn/rss/xa-hoi.rss",
        "https://dantri.com.vn/rss/the-gioi.rss",
        "https://dantri.com.vn/rss/kinh-doanh.rss",
        "https://dantri.com.vn/rss/bat-dong-san.rss",
        "https://dantri.com.vn/rss/van-hoa.rss",
        "https://dantri.com.vn/rss/lao-dong-viec-lam.rss",
        "https://dantri.com.vn/rss/tam-long-nhan-ai.rss",
        "https://dantri.com.vn/rss/suc-khoe.rss",
        "https://dantri.com.vn/rss/giao-duc-huong-nghiep.rss",
    ],
    "NguoiLaoDong": [
        "https://nld.com.vn/rss/thoi-su.rss",
        "https://nld.com.vn/rss/quoc-te.rss",
        "https://nld.com.vn/rss/kinh-te.rss",
        "https://nld.com.vn/rss/phap-luat.rss",
        "https://nld.com.vn/rss/cong-doan.rss",
        "https://nld.com.vn/rss/ban-doc.rss",
        "https://nld.com.vn/rss/giao-duc-khoa-hoc.rss",
    ],
}

IGNORED_URL_PATTERNS = [
    "rss.html",
    "static",
    "dieu-khoan",
    "chinh-sach",
    "gioi-thieu",
    "about",
    "contact",
    "video",      # Often video-only pages
    "podcast",
    "quiz",
]

# ====================
# CRAWLER IMPLEMENTATION
# ====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RSSCrawler")

class RSSNewsCrawler:
    def __init__(self, output_dir: str = "data_real", delay: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = self._init_session()

    def _init_session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,vi;q=0.8"
        })
        return s

    def is_valid_url(self, url: str) -> bool:
        """Check if URL contains ignored patterns."""
        if not url: return False
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in IGNORED_URL_PATTERNS):
            return False
        return True

    def fetch_and_clean_article(self, url: str) -> Optional[Dict]:
        """Fetches URL and extracts clean content."""
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # Use appropriate encoding, fallback to apparent_encoding if needed
            if response.encoding and response.encoding.lower() == 'iso-8859-1':
                 response.encoding = response.apparent_encoding
            
            html = response.text
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

        if not html: return None
        
        soup = BeautifulSoup(html, "html.parser")

        # 1. Remove unwanted elements
        for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()

        # 2. Extract Title
        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        
        if not title: # Fallback to title tag
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

        if not title:
            return None

        # 3. Content Extraction Strategy
        # Priority: <article> -> known classes -> generic div
        # Using a specialized selector list for Vietnamese news sites
        article_body = (
            soup.find("article") or
            soup.find("div", class_=lambda x: x and any(c in x for c in ["content-detail", "fck_detail", "article-body", "description", "post-content", "detail-content", "main-detail"])) or
            soup.find("div", id=lambda x: x and any(c in x for c in ["content", "article-content", "mainContent"])) or
            soup.find("main")
        )

        content = ""
        if article_body:
            # Extract paragraphs
            paragraphs = []
            for p in article_body.find_all("p"):
                text = p.get_text(strip=True)
                # Keep only paragraphs longer than 40 characters (filter captions/metadata)
                if len(text) > 40:
                    paragraphs.append(text)
            
            content = "\n\n".join(paragraphs)
        
        # 4. Data Quality Filter
        if len(content) < 200:
            logger.debug(f"Skipping {url}: Content too short ({len(content)} chars)")
            return None

        return {
            "title": title,
            "content": content,
            "url": url
        }

    def process_feed(self, feed_url: str) -> List[Dict]:
        """Parses a single RSS feed."""
        logger.info(f"Parsing feed: {feed_url}")
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            logger.error(f"Error parsing feed {feed_url}: {e}")
            return []

        if feed.bozo:
             logger.warning(f"Feed malformed {feed_url}: {feed.bozo_exception}")
             # Continue anyway as feedparser often extracts usable data even from malformed feeds

        items = []
        for entry in feed.entries:
            link = entry.get("link")
            if not link or not self.is_valid_url(link):
                continue

            item = {
                "link": link,
                "title": entry.get("title", ""),
                "published": entry.get("published", entry.get("updated", "")),
                "summary": entry.get("summary", "") # Use as fallback or metadata
            }
            items.append(item)
        
        return items

    def process_source(self, source_name: str, feed_urls: List[str]):
        """Orchestrates fetching for a single source."""
        logger.info(f"========== Starting {source_name} ==========")
        
        all_articles = []
        visited_urls = set() # Deduplicate within this run
        
        # 1. Collect all valid links from all feeds first
        potential_items = []
        for url in feed_urls:
            items = self.process_feed(url)
            potential_items.extend(items)
        
        logger.info(f"Found {len(potential_items)} items from RSS feeds for {source_name}")

        # 2. Fetch content for each item
        for item in potential_items:
            url = item["link"]
            
            if url in visited_urls:
                continue
            visited_urls.add(url)
            
            logger.info(f"Fetching: {url}")
            article_data = self.fetch_and_clean_article(url)
            
            if article_data:
                # Merge RSS metadata with scraped content
                article_data["published"] = item["published"]
                article_data["rss_summary"] = item["summary"]
                all_articles.append(article_data)

        # 3. Save results
        self.save_results(source_name, all_articles)

    def save_results(self, source_name: str, articles: List[Dict]):
        if not articles:
            logger.warning(f"No valid articles collected for {source_name}")
            return

        source_path = self.output_dir / source_name
        source_path.mkdir(exist_ok=True)
        
        filename = source_path / f"{source_name}_articles.json"
        
        output_data = {
            "source": source_name,
            "count": len(articles),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "articles": articles
        }
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(articles)} articles to {filename}")
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")

    def run_all(self):
        for name, feeds in RSS_SOURCES.items():
            self.process_source(name, feeds)

if __name__ == "__main__":
    crawler = RSSNewsCrawler()
    crawler.run_all()
