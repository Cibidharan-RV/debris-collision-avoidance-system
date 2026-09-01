import os
import requests
from skyfield.api import Loader

load = Loader('data')

def fetch_active_satellites():
    print("Fetching active satellites from CelesTrak...")
    
    url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
    filename = 'data/active.txt'
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    if not os.path.exists(filename):
        print("Downloading from web with custom User-Agent...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)
        
        with open(filename, 'w') as f:
            f.write(response.text)
            
    print("Loading TLEs from local file...")
    satellites = load.tle_file('active.txt')
    
    print(f"Fetched {len(satellites)} active satellites")
    return satellites

if __name__ == "__main__":
    satellites = fetch_active_satellites()

    # Let's print the first 10 satellites to prove it works
    for satellite in satellites[:10]:
        print(satellite.name, satellite.epoch.utc_jpl())

