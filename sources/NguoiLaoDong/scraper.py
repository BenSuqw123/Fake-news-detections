
from core.base_scraper import NewsScraper

class NguoiLaoDongScraper(NewsScraper):
    def start(self, limit=2000):
        super().run(
            name="NguoiLaoDong",
            start_urls=["https://nld.com.vn/thoi-su.htm"],
            link_selector="a[href$='.htm']",
            pagination_pattern="https://nld.com.vn/thoi-su/trang-{page}.htm",
            limit=limit
        )
