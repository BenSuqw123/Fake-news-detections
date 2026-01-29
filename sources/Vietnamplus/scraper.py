
from core.base_scraper import NewsScraper

class VietnamplusScraper(NewsScraper):
    def start(self, limit=2000):
        super().run(
            name="Vietnamplus",
            start_urls=["https://www.vietnamplus.vn/thoi-su/"],
            link_selector="a[href*='.vnp']",
            pagination_pattern="https://www.vietnamplus.vn/thoi-su/page/{page}.vnp",
            limit=limit
        )
