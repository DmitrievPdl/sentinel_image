INPUT_FILE = './data/polygon.json'

from sentinelhub import read_data
import numpy as np
from sentinel_img import SentinelImg, User
from shapely.geometry import shape
import datetime

if __name__ == "__main__":
    # Make user with client id
    # and client secret
    user = User()

    # Read geojson polygon
    geo_json = read_data(INPUT_FILE)
    polygons = []
    for polygon in geo_json['features']:
        polygons.append(np.array(shape(polygon['geometry']).boundary))

    start = datetime.datetime(2020, 5, 1)
    end = datetime.datetime(2020, 8, 1)
    n_chunks = 8
    tdelta = (end - start) / n_chunks
    edges = [(start + i*tdelta).date().isoformat() for i in range(n_chunks)]
    slots = [(edges[i], edges[i+1]) for i in range(len(edges)-1)]

    print('Monthly time windows:\n')
    for slot in slots:
        print(slot)
        img = SentinelImg(polygons, slot, user)
        print(f'img is available: {img.is_available}')
        print(f'{img.cloud_percent} - % cloudy')
        img.plot(type='true_color', factor=3.5/255, clip_range=(0,1))
