
from core.base_scraper import NewsScraper

class TuoiTreScraper(NewsScraper):
    def start(self, limit=2000):
        super().run(
            name="TuoiTre",
            start_urls=["https://tuoitre.vn/thoi-su.htm"],
            link_selector="a[href$='.htm']",
            pagination_pattern="https://tuoitre.vn/thoi-su/trang-{page}.htm",
            limit=limit
        )
