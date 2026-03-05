import requests 
from bs4 import BeautifulSoup

urls = ["https://auto.mercadolibre.com.mx/MLM-2758230457-crafter-cargo-van-std-49-tlwb-_JM#polycard_client=search-desktop&search_layout=grid&position=8&type=item&tracking_id=9fef13dd-de80-4f29-ace3-18cf5a670731","https://auto.mercadolibre.com.mx/MLM-2726755513-audi-q6-e-tron-00-55-advanced-quattro-_JM#polycard_client=search-desktop&search_layout=grid&position=43&type=item&tracking_id=2c0c21c0-add2-47b8-8c4f-adc5583137d2",
"https://auto.mercadolibre.com.mx/MLM-4921510640-kia-niro-2022-hibrida-_JM#polycard_client=search-desktop&search_layout=grid&position=44&type=item&tracking_id=2c0c21c0-add2-47b8-8c4f-adc5583137d2", 
"https://auto.mercadolibre.com.mx/MLM-4928938572-nissan-urvan-2021-_JM#polycard_client=search-desktop&search_layout=grid&position=45&type=item&tracking_id=2c0c21c0-add2-47b8-8c4f-adc5583137d2",
"https://auto.mercadolibre.com.mx/MLM-4306836538-ford-f350-2026-_JM#polycard_client=search-desktop&search_layout=grid&position=46&type=item&tracking_id=2c0c21c0-add2-47b8-8c4f-adc5583137d2",
"https://auto.mercadolibre.com.mx/MLM-2758094861-chevrolet-onix-lt-2021-_JM#polycard_client=search-desktop&search_layout=grid&position=47&type=item&tracking_id=2c0c21c0-add2-47b8-8c4f-adc5583137d2",
"https://auto.mercadolibre.com.mx/MLM-3708187188-audi-rs5-sportback-29t-v6-quattro-at-_JM#polycard_client=search-desktop&search_layout=grid&position=48&type=item&tracking_id=2c0c21c0-add2-47b8-8c4f-adc5583137d2"]

for url in urls:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")

        attributes = soup.find_all("div", class_="andes-table__header__container")
        attribute_values = soup.find_all("span", class_="andes-table__column--value")
        at_dict = dict(zip(attributes, attribute_values))

        

    else:
        print("invalid connection")