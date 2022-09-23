import os
from pathlib import Path

import rasterio
from pandas import DataFrame
from rasterio.plot import reshape_as_image
import rasterio.mask
from rasterio.features import rasterize

import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union, unary_union

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from dataset import CSV_FILENAME, CSV_FIELDS, IMG_SIZE

RASTER_PATH = "T36UXV_20200406T083559_TCI_10m.jp2"
# DATA_FOLDER = "./data"
ROOT_DIR = './data/'
IMAGES_DIR = ROOT_DIR + 'images/'
MASKS_DIR = ROOT_DIR + 'masks/'
CRS = 'epsg:4322'  # EPSG:4322 Name:WGS 72


# rasterize works with polygons that are in image coordinate system
def poly_from_utm(polygon, transform):
    poly_pts = []

    # make a polygon from multipolygon
    poly = cascaded_union(polygon)
    for i in np.array(poly.exterior.coords):
        # transform polygon to image crs, using raster meta
        poly_pts.append(~transform * tuple(i))

    # make a shapely Polygon object
    new_poly = Polygon(poly_pts)
    return new_poly


def get_tiles(map: np.ndarray, box_size: int, include_cut_edges: bool = True) -> (np.ndarray, int, int, int, int):
    img_height, img_width = map.shape
    h = 0
    while h < img_height:
        w = 0
        while w < img_width:
            h2 = h + box_size
            w2 = w + box_size
            if h2 < img_height and w2 < img_width:
                yield map[h:h2, w:w2], h, h2, w, w2
            elif include_cut_edges:
                h1, h2, w1, w2 = h, h2, w, w2
                if h2 >= img_height:
                    h2, h1 = img_height - 1, img_height - 1 - box_size
                if w2 >= img_width:
                    w2, w1 = img_width - 1, img_width - 1 - box_size
                yield map[h1:h2, w1:w2], h1, h2, w1, w2
            w += box_size
        h += box_size


def main():
    src = rasterio.open(RASTER_PATH, 'r', driver="JP2OpenJPEG")
    raster_meta = src.meta

    # Load picture
    img = src.read()
    img = reshape_as_image(img)

    # Get mean and std for whole picture
    mean = np.mean(img, axis=(0, 1))
    std = np.std(img, axis=(0, 1))
    transform_zerocenter = {'mean': mean, 'std': std}
    transform_zerocenter_df = pd.DataFrame(data=transform_zerocenter)
    transform_zerocenter_df.to_csv('transform_zerocenter.csv')

    # Close file handle
    src.close()

    # Get mask database and converting GeoData frame to raster CRS
    train_df = gpd.read_file("masks/Masks_T36UXV_20190427.shp")
    train_df = train_df[train_df.geometry.notnull()]
    train_df.crs = {'init': CRS}
    train_df = train_df.to_crs({'init': raster_meta['crs']['init']})
    train_df = train_df.dropna(subset=['geometry'])  # get rid of null geometry rows

    # Creating binary mask for field/not_filed segmentation.
    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    for num, row in train_df.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)
        else:
            for p in row['geometry']:
                poly = poly_from_utm(p, src.meta['transform'])
                poly_shp.append(poly)

    mask = rasterize(shapes=poly_shp, out_shape=im_size)

    # Chop whole image to tiles and save those with mask on them
    Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    Path(MASKS_DIR).mkdir(parents=True, exist_ok=True)
    # tiles = []
    i = 0
    data_csv = DataFrame(columns=CSV_FIELDS)
    for mask_tile, min_height, max_height, min_width, max_width in get_tiles(mask, IMG_SIZE):
        # Check if there is mask on the tile
        if np.sum(mask_tile) > 0:
            # Get image and mask from the that area
            tile_img = Image.fromarray(img[min_height: max_height, min_width: max_width, :])
            tile_mask = Image.fromarray(mask_tile * 255)
            # Save them and add to dataframe (csv)
            name_stem = str(i).zfill(4)
            img_file_location = str(IMAGES_DIR + name_stem + '.png')
            mask_file_location = str(MASKS_DIR + name_stem + '_mask.png')
            tile_img.save(img_file_location, "png")
            tile_mask.save(mask_file_location, "png")
            data_csv = data_csv.append({CSV_FIELDS[0]: i,
                                        CSV_FIELDS[1]: img_file_location,
                                        CSV_FIELDS[2]: mask_file_location,
                                        CSV_FIELDS[3]: min_height,
                                        CSV_FIELDS[4]: max_height,
                                        CSV_FIELDS[5]: min_width,
                                        CSV_FIELDS[6]: max_width}
                                       , ignore_index=True)
            i += 1
    data_csv.to_csv(ROOT_DIR + CSV_FILENAME)


if __name__ == "__main__":
    main()
