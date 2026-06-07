import os

import requests

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/Mylinear/Brazilian_E_Commerce_Public_Dataset_by_Olist/main"
FILES = [
    "olist_customers_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def download_file(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"File {dest_path} already exists. Skipping download.")
        return
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {dest_path}")
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")


def main():
    for f in FILES:
        url = f"{BASE_URL}/{f}"
        dest = os.path.join(DATA_DIR, f)
        download_file(url, dest)


if __name__ == "__main__":
    main()
