
from core.base_scraper import NewsScraper

class ThanhNienScraper(NewsScraper):
    def start(self, limit=2000):
        super().run(
            name="ThanhNien",
            start_urls=["https://thanhnien.vn/thoi-su.htm"],
            link_selector="a[href$='.html']",
            pagination_pattern="https://thanhnien.vn/thoi-su/trang-{page}.html",
            limit=limit
        )
