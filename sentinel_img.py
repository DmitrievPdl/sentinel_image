from collections import namedtuple
import numpy as np
from sentinelhub import (
    BBox, bbox_to_dimensions, CRS,\
    SentinelHubRequest, DataCollection,\
    MimeType, SHConfig, Geometry
)
from shapely.geometry import Polygon
from descartes import PolygonPatch
from matplotlib import pyplot as plt
import os
from evalscripts import *
import pandas as pd
import time
import functools

Coords = namedtuple("Coords",[
            "x_max", "y_max",
            "x_min", "y_min"
])
def attrs(obj):
    ''' Return public attribute values dictionary for an object '''
    return dict(i for i in vars(obj).items() if i[0][0] != '_')

def get_request(img, evalscript, **kwargs):
    """
    Build a query using the given
    evalscript and img.
    """

    return SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=img.time_interval,
                mosaicking_order='leastCC'
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.PNG)
        ],
        bbox=img.bbox,
        size=img.size,
        config=img.user.config,
        **kwargs
    )

class User():
    def __init__(self):
        # Set environment
        config = SHConfig()
        if os.getenv("CLIENT_ID") and os.getenv("CLIENT_SECRET"):
            config.sh_client_id = os.getenv("CLIENT_ID")
            config.sh_client_secret = os.getenv("CLIENT_SECRET")
        else:
            raise Exception("Client id and client secret is empty.")
        self.config = config

class SentinelImg():
    def __init__(self, polygons, time_interval,
                 user, zoom = 1.7, size = (1000, 1500)):
        # First polygon is external, other is inner.
        self.external_polygon = polygons[0]
        try:
            self.hole_polygons = polygons[1:]
        except IndexError:
            self.hole_polygons = None
        self.time_interval = time_interval
        self.user = user
        self.zoom = self.check_zoom(zoom)
        self.size = self.check_size(size)
        self._coords = None
        self._bbox = None
        self._true_color_map = None
        self._cloud_percent = None
        self._cloud_map = None
        self.is_available()

    @staticmethod
    def check_size(size):
        if  (
                isinstance(size, tuple) and len(size) == 2 and
                isinstance(size[0], int) and isinstance(size[1], int) and
                size[0] > 500 and size[0] < 2000 and
                size[1] > 500 and size[1] < 2000
            ):
            return size
        else:
            raise AttributeError(
                """
                Size attribute must be tuple (width, length).
                Where width and lenght is int and
                500 < wight < 2000 and 500 < lenght < 2000.
                """
                )

    @staticmethod
    def check_zoom(zoom):
        if  (
                isinstance(zoom, float) and
                zoom > 1.0 and zoom < 2.0
            ):
            return zoom
        else:
            raise AttributeError(
                """
                Zoom attribute must be float
                and 1.0 < zoom < 2.0.
                """
                )

    @property
    def coords(self):
        if self._coords == None:
            x_max, y_max = np.amax(self.external_polygon, axis=0)
            x_min, y_min = np.amin(self.external_polygon, axis=0)
            len_x, len_y = x_max - x_min, y_max - y_min
            self._coords = Coords(
                x_max=x_max + len_x * (self.zoom - 1), y_max=y_max + len_y * (self.zoom - 1),
                x_min=x_min - len_x * (self.zoom - 1), y_min=y_min - len_y * (self.zoom - 1)
            )
        return self._coords

    @property
    def bbox(self):
        if self._bbox == None:
            self._bbox = BBox(
                bbox=[
                    self.coords.x_min, self.coords.y_min,
                    self.coords.x_max, self.coords.y_max
                ],
                crs=CRS.WGS84)
        return self._bbox

    @property
    def true_color_map(self):
        """
        Download true colot map from API.
        """
        if not isinstance(self._true_color_map, np.ndarray):
            request = get_request(self, evalscript_true_color)
            self._true_color_map = request.get_data()[0]
        return self._true_color_map

    @property
    def cloud_map(self):
        """
        Download cloud map from API.
        """
        if not isinstance(self._cloud_map, pd.DataFrame):
            geometry = Geometry(Polygon(self.external_polygon), CRS.WGS84)
            request = get_request(
                self,
                evalscript=evalscript_cloud_mask,
                geometry=geometry
            )
            self._cloud_map = pd.DataFrame(request.get_data()[0])
        return self._cloud_map

    def is_available(self):
        """
        Check image is available.
        """
        request = get_request(self, evalscript_is_available)
        df_dataMask = pd.DataFrame(request.get_data()[0])
        dataMask_px = (df_dataMask == 1 ).sum().sum()
        self.is_available = dataMask_px == 0

    @property
    def cloud_percent(self):
        """
        Retrun сloud percent.
        """
        if self._cloud_percent == None:
            # the number of pixels in which 
            # the clouds fall is calculated
            # and divided by the total number of clouds
            no_data_px = (self.cloud_map == 0 ).sum().sum()
            no_cloud_px = (self.cloud_map == 1 ).sum().sum()
            cloud_px = (self.cloud_map == 2 ).sum().sum()
            total_px = cloud_px + no_cloud_px
            self._cloud_percent = int( cloud_px / total_px * 100)
        return self._cloud_percent

    def plot(self, type="true_color", factor=1.0, clip_range=None, **kwargs):

        if type == "true_color":
            image = self.true_color_map
        else:
            raise Exception("Only true-color type is aveliable.")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        extent = [
            self.coords.x_min, self.coords.x_max,
            self.coords.y_min, self.coords.y_max
        ]
        if clip_range is not None:
            ax.imshow(np.clip(image * factor, *clip_range), extent=extent, **kwargs)
        else:
            ax.imshow(image * factor, extent=extent, **kwargs)

        # Marks polygons areas
        if self.hole_polygons is not None:
            p = Polygon(
                self.external_polygon,
                holes=[hole[::-1] for hole in self.hole_polygons]
            )
        else:
            p = Polygon(self.external_polygon)
        patch = PolygonPatch(p, facecolor=(0,0,1,.2), edgecolor='black')
        ax.add_patch(patch)

        plt.axis("off")
        plt.savefig(f"./img/polygon_{type}_{self.cloud_percent}.png", bbox_inches = 'tight', pad_inches = 0)
    
    @property
    def public(self):
        ''' Return the public model attributes '''
        return attrs(self)

    def __str__(self):
        return f'Sentinel Image time interval: {self.time_interval}'

    def __repr__(self):
        return f'<SentinelImg {self.public}>'