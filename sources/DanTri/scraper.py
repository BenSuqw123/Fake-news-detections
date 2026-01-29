
from core.base_scraper import NewsScraper

class DanTriScraper(NewsScraper):
    def start(self, limit=2000):
        super().run(
            name="DanTri",
            start_urls=["https://dantri.com.vn/xa-hoi.htm"],
            link_selector="a[href$='.htm']",
            pagination_pattern="https://dantri.com.vn/xa-hoi/trang-{page}.htm",
            limit=limit
        )
