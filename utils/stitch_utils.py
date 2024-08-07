import re
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import typer
from aicsimageio import AICSImage
from sklearn.cluster import AgglomerativeClustering


def infer_ext(img_dir: str | Path) -> str:
    # get file extension of all files in the directory
    exts = {i.suffix for i in Path(img_dir).glob("*")}
    # check if .tif and/or .nd2 files are present
    # if both are present, raise an error
    if ".tif" in exts and ".nd2" in exts:
        raise ValueError("Both .tif and .nd2 files are present in the directory")
    elif ".tif" in exts:
        ext = ".tif"
    elif ".nd2" in exts:
        ext = ".nd2"
    else:
        raise ValueError("No .tif or .nd2 files found in the directory")
    return ext


# For a directory of images, parse image names to create m2stitch-compatible dataframe
def get_images_dataframe(img_dir: str | Path, ext: str = ""):
    """
    Get a dataframe of image file paths and their corresponding stage positions
    """
    if not ext:
        ext = infer_ext(img_dir)
    if ext == ".tif":
        # TODO: This is a bespoke coordinate extraction for the specific naming convention
        # TODO: Convert to a general ome metadata reader (as below)
        coord_exp = r"x([\+\-][0-9]+)_y([\+\-][0-9]+)_"
        imgs = [
            str(im)
            for im in Path(img_dir).glob("*.tif")
            if re.search(coord_exp, str(im))
        ]

        img_df = pd.DataFrame(columns=["fpath", "x", "y"], index=np.arange(len(imgs)))
        img_df["fpath"] = imgs

        img_df["x"] = img_df["fpath"].apply(
            lambda x: float(re.search(coord_exp, x).groups()[0])
        )
        img_df["y"] = img_df["fpath"].apply(
            lambda x: float(re.search(coord_exp, x).groups()[1])
        )
        # hardcoded for compatibility
        img_df["x"] = img_df["x"] / 0.65
        img_df["y"] = img_df["y"] / 0.65
    elif ext == ".nd2":
        im_loc = []
        for im in Path(img_dir).glob("*.nd2"):
            img = AICSImage(im)
            # if image has multiple planes, get the stage position for all planes
            pos = [
                (plane.position_x, plane.position_y)
                for plane in img.ome_metadata.images[0].pixels.planes
            ]
            stage_pos = pos[0]
            # check that the stage position is the same for all planes
            assert all(p == stage_pos for p in pos)

            # get pixel size
            px, py = (
                img.ome_metadata.images[0].pixels.physical_size_x,
                img.ome_metadata.images[0].pixels.physical_size_y,
            )
            im_loc.append((im, *stage_pos))
        img_df = pd.DataFrame(im_loc, columns=["fpath", "x", "y"])
        img_df["x"] = img_df["x"] / px
        img_df["y"] = img_df["y"] / py
    else:
        raise ValueError("Image format not supported")

    return img_df


# From Eric/ChatGPT
def estimate_grid_size(img_df, cluster_threshold):
    x_cluster = AgglomerativeClustering(
        n_clusters=None, distance_threshold=cluster_threshold, linkage="single"
    )
    y_cluster = AgglomerativeClustering(
        n_clusters=None, distance_threshold=cluster_threshold, linkage="single"
    )

    x_labels = x_cluster.fit_predict(img_df[["x"]])
    y_labels = y_cluster.fit_predict(img_df[["y"]])

    num_rows = len(np.unique(x_labels))
    num_cols = len(np.unique(y_labels))

    return num_rows, num_cols


def get_m2stitch_coords_grid_based(
    img_df, debug=False, cluster_threshold=10, flip_x=False, flip_y=False
):
    grid_size = estimate_grid_size(img_df, cluster_threshold)

    # Load the first image to get the shape
    rdr = AICSImage(img_df.iloc[0]["fpath"])
    imarr = rdr.get_image_dask_data("CZYX", T=0)

    imzarr = da.zeros_like(
        None, shape=(img_df.shape[0],) + imarr.shape, dtype=imarr.dtype
    )

    # Normalize coordinates and create grid indices
    x_min, y_min = img_df["x"].min(), img_df["y"].min()
    x_max, y_max = img_df["x"].max(), img_df["y"].max()

    # Make sure that the maximum value is included in the last grid cell
    img_df["grid_x"] = np.clip(
        ((img_df["x"] - x_min) / (x_max - x_min) * grid_size[0]), 0, grid_size[0] - 1
    ).astype(int)
    img_df["grid_y"] = np.clip(
        ((img_df["y"] - y_min) / (y_max - y_min) * grid_size[1]), 0, grid_size[1] - 1
    ).astype(int)

    # Assign row and column based on grid
    img_df["row"] = img_df["grid_x"]
    img_df["col"] = img_df["grid_y"]

    # Fill in the dask array with image data
    for idx, row in img_df.iterrows():
        rdr = AICSImage(row.fpath)
        im = rdr.get_image_dask_data("CZYX")
        if flip_y:
            im = im[:, :, ::-1, :]
        if flip_x:
            im = im[:, :, :, ::-1]
        imzarr[idx, :, :, :, :] = im

    return imzarr, img_df["row"].tolist(), img_df["col"].tolist()


# Converts tiles into a whole-slide dask-array lazily
def stitch_tiles(tile_arr, stitch_df):
    size_y = tile_arr.shape[1]
    size_x = tile_arr.shape[2]

    stitch_size = (
        stitch_df["y_pos"].max() + size_y,
        stitch_df["x_pos"].max() + size_x,
    )

    stitched_image = da.empty_like(
        None,
        shape=stitch_size,
        dtype=np.uint32,  # Hard coded cuz im lazy
    )
    stitched_mask = da.empty_like(stitched_image, dtype=np.uint8)

    for i, row in stitch_df.iterrows():
        stitched_image[
            row.y_pos : (row.y_pos + size_y),
            row.x_pos : (row.x_pos + size_x),
        ] = tile_arr[row.tile_id, :, :]

        stitched_mask[
            row.y_pos : (row.y_pos + size_y),
            row.x_pos : (row.x_pos + size_x),
        ] = row.tile_id

    return stitched_image, stitched_mask


def cli_stitch(
    im_folder: Path,
    stitch_file: Path,
    out_path: Path,
    flip_x: bool = False,
    flip_y: bool = False,
):
    # for n, im_path in enumerate(im_folder.glob("*")):
    stitch_df = pd.read_csv(stitch_file)
    # segmented_dir = out_path.parent.joinpath("segmented")
    for n, row in stitch_df.iterrows():
        old_im_path = Path(row.fpath)
        im_path = im_folder.joinpath(old_im_path.stem + "_segmented.ome.tiff")
        im = AICSImage(im_path).get_image_dask_data("YX")
        if flip_x:
            im = im[:, ::-1]
        if flip_y:
            im = im[::-1, :]
        # get bit depth
        dtype = im.dtype
        print(dtype)
        if dtype == np.uint16:
            im = im.astype(np.uint32)
            im = (im + 2**16 * n) * (im > 0)
        elif dtype == np.uint8:
            im = im.astype(np.uint32)
            im = (im + 2**8 * n) * (im > 0)
        else:
            raise ValueError("Image bit depth not supported")
        print(n, im.mean().compute(), im.max().compute(), im.min().compute())
        if n == 0:
            imzarr = da.zeros_like(
                None,
                shape=(len(list(im_folder.glob("*"))),) + im.shape,
                dtype=np.uint32,
            )
        imzarr[n, :, :] = im
    stitched_image, stitched_mask = stitch_tiles(imzarr, stitch_df)
    # save stitched image
    stitched_image.to_zarr(out_path, overwrite=True)


if __name__ == "__main__":
    typer.run(cli_stitch)
