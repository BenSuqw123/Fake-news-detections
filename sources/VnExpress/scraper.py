
from core.base_scraper import NewsScraper

class VnExpressScraper(NewsScraper):
    def start(self, limit=2000):
        super().run(
            name="VnExpress",
            start_urls=["https://vnexpress.net/thoi-su"],
            link_selector="a[href$='.html']",
            pagination_pattern="https://vnexpress.net/thoi-su-p{page}",
            limit=limit
        )
