
from core.base_scraper import NewsScraper

class VTCNewsScraper(NewsScraper):
    def start(self, limit=2000):
        super().run(
            name="VTCNews",
            start_urls=["https://vtc.vn/thoi-su.html"],
            link_selector="a[href$='.html']",
            pagination_pattern="https://vtc.vn/thoi-su/trang-{page}.html",
            limit=limit
        )
