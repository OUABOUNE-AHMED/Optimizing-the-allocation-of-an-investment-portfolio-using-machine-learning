from matplotlib import ticker
import requests
from bs4 import BeautifulSoup
import os

def get_logo(tickers_list):
    images = []
    for ticker in tickers_list:
        url = f'https://www.tradingview.com/symbols/NASDAQ-{ticker}/'
        cookies = {
            '_sp_ses.cf1a': '*',
            '_sp_id.cf1a': 'fae78f9c-2ee8-4598-b504-df36037e640b.1646861479.3.1649003427.1647712930.a10c82a7-fc67-4f35-92a5-9b56646b5049',
        }
        headers = {
            'authority': 'www.tradingview.com',
            'cache-control': 'max-age=0',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-user': '?1',
            'sec-fetch-dest': 'document',
            'accept-language': 'en-US,en;q=0.9,fr;q=0.8',
        }
        r = requests.get(url, headers=headers, cookies=cookies)
        soup = BeautifulSoup(r.text, "html.parser")
        img = soup.find("div", {"class": "js-sticky-symbol-header-container tv-sticky-symbol-header"})
        img_url = img["data-logo-urls"]
        images.append(img_url)

    return images