from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
from aicsimageio import AICSImage
from dask import compute, delayed
from dask.diagnostics import ProgressBar
from scipy.spatial import ConvexHull, qhull
from shapely.geometry import Polygon


def get_cell_boundaries(im: da.Array, min_size: int = 25) -> gpd.GeoDataFrame:
    @delayed
    def compute_convex_hull(im, label):
        if label == 0:
            return None
        masked = im == label
        if np.any(masked):
            points = np.transpose(np.nonzero(masked))
            if points.size < min_size:
                return None
            try:
                # Use the 'QJ' option for joggling the input to handle precision issues
                hull = ConvexHull(points, qhull_options="QJ")
                return Polygon(hull.points[hull.vertices])
            except qhull.QhullError as e:
                print(f"Failed to compute convex hull for label {label} due to: {e}")
                return None
        else:
            return None

    # Assuming `im` is a Dask array with integer labels
    unique_labels = da.unique(im).compute()

    # Create delayed tasks for each label's convex hull calculation
    delayed_tasks = [compute_convex_hull(im, label) for label in unique_labels]

    # Execute all tasks in parallel and retrieve results, with a progress bar
    with ProgressBar():
        polygons = compute(*delayed_tasks)

    # Filter out None results
    polygons = [polygon for polygon in polygons if polygon is not None]

    # Create a GeoDataFrame from the computed polygons
    gdf = gpd.GeoDataFrame({"geometry": polygons})
    return gdf


def cli(im_path: Path, out_path: Path, min_size: int = 25):
    """
    Compute the convex hulls of cell masks in an image and save them to a GeoDataFrame.

    Args:
        im_path: Path to the image file containing cell masks.
        out_path: Path to save the GeoDataFrame containing the convex hulls.
        min_size: Minimum number of pixels required to compute the convex hull for a cell mask.
    """
    if im_path.suffix == ".zarr":
        # TODO: Script written with the assumption that the zarr is a 2D image
        im = da.from_zarr(im_path)
    else:
        # TODO: Script written with the assumption that the image is a 3D image
        im = AICSImage(im_path).get_image_dask_data(
            "ZYX", C=0
        )  # Assuming the first channel is the cell mask

    boundaries = get_cell_boundaries(im, min_size)
    boundaries.to_parquet(out_path)


if __name__ == "__main__":
    import typer

    typer.run(cli)
