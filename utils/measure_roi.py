from pathlib import Path

import dask.array as da
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box


def measure_roi(
    gdf: gpd.GeoDataFrame, image: da.Array | np.ndarray, npartitions: int = 100
) -> gpd.GeoDataFrame:
    """
    Measure image statistics for each geometry in a GeoDataFrame or GeoSeries using Dask for parallelization.

    Args:
        gdf (gpd.GeoDataFrame or gpd.GeoSeries): Input GeoDataFrame or GeoSeries containing the geometries.
        image (da.Array or np.ndarray): 2D array representing the image to measure.
        npartitions (int): Number of partitions to use for parallel processing (default: 100).

    Returns:
        dgpd.GeoDataFrame: Dask GeoDataFrame with the same geometries as the input and additional columns
                          for the measured image statistics (sum, mean, median, std, pixel_count).
    """

    def get_values(geom):
        if geom.is_valid:
            bbox = geom.bounds
            y_min, x_min, y_max, x_max = (
                int(bbox[1]),
                int(bbox[0]),
                int(bbox[3]),
                int(bbox[2]),
            )
            cropped_im = image[
                max(y_min, 0) : min(y_max, image.shape[0]),
                max(x_min, 0) : min(x_max, image.shape[1]),
            ]
            if isinstance(cropped_im, da.Array):
                cropped_im = cropped_im.compute()
            cropped_geom = geom.intersection(
                box(
                    max(x_min, 0),
                    max(y_min, 0),
                    min(x_max, image.shape[1]),
                    min(y_max, image.shape[0]),
                )
            )
            raster_im = rasterize(
                [cropped_geom],
                out_shape=cropped_im.shape,
                fill=0,
                default_value=1,
                dtype="uint8",
                transform=from_bounds(
                    max(x_min, 0),
                    max(y_min, 0),
                    min(x_max, image.shape[1]),
                    min(y_max, image.shape[0]),
                    cropped_im.shape[1],
                    cropped_im.shape[0],
                ),
            )
            masked_values = raster_im * cropped_im
            valid_pixels = masked_values[masked_values != 0]
            if valid_pixels.size > 0:
                return pd.Series(
                    {
                        "sum": valid_pixels.sum(),
                        "mean": valid_pixels.mean(),
                        "median": np.median(valid_pixels),
                        "std": valid_pixels.std(),
                        "max": valid_pixels.max(),
                        "min": valid_pixels.min(),
                        "pixel_count": valid_pixels.size,
                    }
                )
            else:
                return pd.Series(
                    {
                        "sum": np.nan,
                        "mean": np.nan,
                        "median": np.nan,
                        "std": np.nan,
                        "max": np.nan,
                        "min": np.nan,
                        "pixel_count": 0,
                    }
                )
        else:
            return pd.Series(
                {
                    "sum": np.nan,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "max": np.nan,
                    "min": np.nan,
                    "pixel_count": 0,
                }
            )

    if isinstance(gdf, gpd.GeoSeries):
        geometry = gdf
    else:
        geometry = gdf.geometry

    dgdf = dgpd.from_geopandas(
        gpd.GeoDataFrame(geometry=geometry), npartitions=npartitions
    )

    meta = pd.Series(
        index=["sum", "mean", "median", "std", "max", "min", "pixel_count"],
        dtype=float,
    )
    meta["pixel_count"] = pd.Int64Dtype()

    with ProgressBar():
        stats = dgdf.geometry.apply(get_values, meta=meta).compute()
    gdf = gdf.assign(**stats)

    return gdf


def cli(
    in_geodf: Path,
    out_file: Path,
    image: Path,
    image_channel: int | None = None,
    transpose_image: bool = False,
    npartitions: int = 100,
):
    """
    Measure image statistics for each geometry in a GeoDataFrame using Dask for parallelization.

    Args:
        in_geodf (Path): Input GeoDataFrame or GeoSeries containing the geometries.
        out_file (Path): Output file path (.parquet) to save the measured GeoDataFrame.
        image (Path): 2D array representing the image to measure.
        image_channel (int): Image channel to use for measurement (default: None - Use all channels).
        transpose_iamge (bool): Transpose the image before measurement (default: False).
        npartitions (int): Number of partitions to use for parallel processing (default: 100).
    """
    gdf = gpd.read_file(in_geodf)
    im = da.from_zarr(image)
    if image_channel is not None:
        im = im[image_channel, ...]
    if transpose_image:
        im = im.T
    im = im.compute()
    gdf = measure_roi(gdf, im, npartitions)
    gdf.to_parquet(out_file)


if __name__ == "__main__":
    import typer

    typer.run(cli)
