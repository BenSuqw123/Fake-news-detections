
from core.base_scraper import NewsScraper

class VtvNewsScraper(NewsScraper):
    def start(self, limit=2000):
        super().run(
            name="VtvNews",
            start_urls=["https://vtv.vn/xa-hoi.htm"],
            link_selector="a[href$='.htm']",
            pagination_pattern="https://vtv.vn/xa-hoi/trang-{page}.htm",
            limit=limit
        )
