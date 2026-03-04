import requests 
from bs4 import BeautifulSoup
import csv

try:
    with open('datos.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
        instances = 0
        for row in data:
            instances +=1
except:
    instances = 0

url = f'https://autos.mercadolibre.com.mx/autos_Desde{instances}_NoIndex_True'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
}
links = []
counter = 5

while counter:
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")

        link_cards = soup.find_all("a", class_="poly-component__title")
        for card in link_cards:
            links.append(card.get('href'))  

        url = f"https://autos.mercadolibre.com.mx/autos_Desde_{len(links)}_NoIndex_True"

    else:
        print("conection failed")

    counter -= 1

with open('urls.csv', 'a', newline='', encoding='utf-8') as f:
       writer = csv.writer(f)
       writer.writerow(['url'])
       for link in links:
           writer.writerow([link])
