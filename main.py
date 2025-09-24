import sys
import datetime
import os
import requests
from pystac_client import Client
from geopy.geocoders import Nominatim
from tqdm import tqdm

# This script now supports searching and DIRECTLY downloading full products.

# --- User Credentials (Required for full product download) ---
COPERNICUS_EMAIL = "FILL_YOUR_MAIL_HERE"
COPERNICUS_PASSWORD = "FILL_YOUR_PASSWORD_HERE"

def get_auth_token(email, password):
    """Gets an authentication token for downloading protected data."""
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    auth_data = {
        "client_id": "cdse-public",
        "username": email,
        "password": password,
        "grant_type": "password",
    }
    try:
        response = requests.post(token_url, data=auth_data)
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        print(f"--> ERROR: Could not get authentication token. Check credentials. Details: {e}")
        return None

def download_full_product(item, auth_token):
    """Downloads the full, large PRODUCT asset with a progress bar."""
    product_asset = item.assets.get("PRODUCT")
    if not product_asset:
        print("--> ERROR: Full PRODUCT asset not found for this item.")
        return

    download_url = product_asset.href
    filename = f"{item.id}.zip"
    
    products_dir = "products"
    if not os.path.exists(products_dir):
        os.makedirs(products_dir)

    filepath = os.path.join(products_dir, filename)

    print(f"\nDownloading full product to {filepath}...")
    try:
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = requests.get(download_url, headers=headers, stream=True, timeout=120)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, "wb") as f, tqdm(
            desc=filename, total=total_size, unit='iB',
            unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        print(f"--> SUCCESS: Full product download complete.")
    except requests.exceptions.RequestException as e:
        print(f"--> FAILED to download full product: {e}")

def search_satellite_data(location_name):
    """
    Searches for data and prompts user to download the full product directly.
    """
    try:
        print(f"--> Finding coordinates for '{location_name}'...")
        geolocator = Nominatim(user_agent="project-manzar", timeout=20)
        location = geolocator.geocode(location_name)
        if not location: return

        print(f"--> Coordinates found: ({location.latitude:.4f}, {location.longitude:.4f})")
        
        STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
        catalog = Client.open(STAC_URL)
        print("--> Successfully connected to the data catalog.")

        search_bbox = [
            location.longitude - 0.1, location.latitude - 0.1,
            location.longitude + 0.1, location.latitude + 0.1,
        ]
        
        start_time = datetime.datetime(2025, 1, 1)
        end_time = datetime.datetime(2025, 3, 31)

        print(f"--> Performing search for images from {start_time.date()} to {end_time.date()}...")
        search = catalog.search(
            collections=["SENTINEL-2"], bbox=search_bbox,
            datetime=[start_time, end_time], limit=250 
        )
        all_items = list(search.item_collection())
        
        if not all_items:
            print("No images found for this period.")
            return

        print(f"\n--- Found Top {min(10, len(all_items))} Results ---")
        selectable_items = {}
        for i, item in enumerate(all_items[:10]):
            result_num = i + 1
            selectable_items[result_num] = item
            print(f"\n  #{result_num}:")
            print(f"    ID: {item.id}")
            print(f"    Date: {item.datetime.strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n" + "="*50)
        print("!!! WARNING: Cloud cover is UNKNOWN. Files are large (~1GB).")
        print("="*50)

        auth_token = None
        while True:
            choice = input("\nEnter the # of the image to download the full product (or 'q' to quit): ").strip()
            if choice.lower() == 'q': break
            
            try:
                choice_num = int(choice)
                if choice_num in selectable_items:
                    if not auth_token:
                        print("--> Getting authentication token...")
                        auth_token = get_auth_token(COPERNICUS_EMAIL, COPERNICUS_PASSWORD)
                        if not auth_token: break
                    
                    download_full_product(selectable_items[choice_num], auth_token)
                else:
                    print(f"--> Invalid number. Please choose between 1 and {len(selectable_items)}.")
            except ValueError:
                print("--> Invalid input. Please enter a number or 'q'.")

    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}")

if __name__ == "__main__":
    search_satellite_data("Nellore, India")

