import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

data = []

headers = {
    "User-Agent": "Mozilla/5.0"
}

for page in range(1, 60):

    url = f"https://www.politifact.com/factchecks/?page={page}"
    print("Crawling:", url)

    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    items = soup.find_all("li", class_="o-listicle__item")

    for item in items:

        claim = item.find("div", class_="m-statement__quote")

        if claim:
            claim = claim.text.strip()

        meter = item.find("div", class_="m-statement__meter")

        label = None
        if meter:
            img = meter.find("img")
            if img:
                label = img["alt"]

        if claim and label:
            data.append({
                "text": claim,
                "label": label
            })

    time.sleep(1)

df = pd.DataFrame(data)

print("Collected:", len(df))

df.to_csv("./Data/politifact_data.csv", index=False)

print("Saved!")