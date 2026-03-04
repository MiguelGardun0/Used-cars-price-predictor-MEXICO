import time
import subprocess
import random

while True:
    print("Starting scraping...")
    print("Searching new links")
    subprocess.run(["python", "url_scraping.py"])

    print("Processing details")
    subprocess.run(["python", "cars_scraping.py"])

    minutes = 60
    print(f"😴 Sleeping for {minutes} minutes...")
    time.sleep(minutes * 60)