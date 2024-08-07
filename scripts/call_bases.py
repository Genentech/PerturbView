import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import dask.array as da
import typer
from geopandas import GeoDataFrame

from utils import call_utils
from utils.bg_subtract import rolling_ball_dask


def main(
    # img should be a dask array or np array
    img: da.Array,
    peak_method: str = "spotiflow",
    transform_json: Optional[Path] = None,
    sigma: float = 1.6,
    grouping_radius: int = 2,
    min_distance: float = 5.0,
    temp_dir: Path = Path(""),
    xenium: bool = False,
    rolling_ball: bool = False,
    rolling_ball_radius: int = 10,
    save: bool = True,
    threshold: str = "None,None,None,None",
    dapi_channel: int = 0,
    date: str = "",
    time: str = "",
) -> GeoDataFrame:
    tform = call_utils.load_transforms(transform_json) if transform_json else None

    # move the DAPI channel to C=0 in array (C,Y,X)
    if dapi_channel != 0:
        # Construct a new channel order
        channels = list(range(img.shape[0]))
        channels.insert(0, channels.pop(dapi_channel))
        img = img[channels, ...]
        if transform_json:
            raise NotImplementedError(
                "Transforms are not yet implemented for non-DAPI channels"
            )

    cfactors = call_utils.get_af_factors(img)
    zimg = call_utils.correct_af(img, cfactors, temp_dir=temp_dir)

    if rolling_ball:
        bg_im = rolling_ball_dask(zimg, rolling_ball_radius)
    else:
        bg_im = zimg

    if peak_method == "spotiflow":
        parr = call_utils.detect_peaks_spotiflow(
            zimg[1:, ...], thresh=0.4, temp_dir=temp_dir
        )
        threshold_values = [
            None if x == "None" else int(x) for x in threshold.split(",")
        ]
        for c in range(parr.shape[0]):
            if threshold_values[c] is not None:
                parr[c, ...] = (
                    parr[c, ...] & (bg_im[c + 1, ...] >= threshold_values[c])
                ).compute()
    else:
        parr = call_utils.detect_peaks(
            zimg[1:, ...],
            sigma=sigma,
            min_distance=int(min_distance),
            temp_dir=temp_dir,
        )

    plabels = call_utils.label_peaks(
        parr, grouping_radius=grouping_radius, temp_dir=temp_dir
    )

    # Save peaks by default
    pdf = call_utils.peak_props(plabels, zimg, tform, xenium)
    if date and time:
        pdf["time"] = datetime.strptime("_".join([date, time]), "%Y%m%d_%H%M%S")

    if save:
        peak_zarr = output_dir.with_suffix(".zarr")
        peak_zarr.mkdir(parents=True, exist_ok=True)
        parr.to_zarr(peak_zarr, overwrite=True)

        peak_labels_zarr = peak_zarr.with_name(peak_zarr.stem + "_labels.zarr")
        peak_labels_zarr.mkdir(parents=True, exist_ok=True)
        plabels.to_zarr(peak_labels_zarr, overwrite=True)

        bg_zarr = output_dir.with_name(output_dir.stem + "_bg.zarr")
        bg_zarr.mkdir(parents=True, exist_ok=True)
        bg_im.to_zarr(bg_zarr, overwrite=True)
    return pdf


app = typer.Typer()


@app.command()
def cli_main(
    img_zarr: Path = typer.Argument(..., help="Stitched ZARR from stitching step"),
    peak_method: str = typer.Option(
        "spotiflow", "-p", "--peak_method", help="Peak finding method"
    ),
    transform_json: Path = typer.Option(
        None, "-j", "--transform_json", help="JSON that registered ZARR to Xenium"
    ),
    output_dir: Path = typer.Option(
        ..., "-o", "--output_dir", help="Output name of GeoJSON object"
    ),
    sigma: float = typer.Option(
        1.6,
        "-s",
        "--sigma",
        help="Standard deviation of Gaussian in LoG filter (only used if peak_method is not 'spotiflow')",
    ),
    grouping_radius: int = typer.Option(
        2,
        "-r",
        "--grouping_radius",
        help="Radius (in spatial coordinates) at which peaks across channels are grouped together into a single peak",
    ),
    min_distance: float = typer.Option(
        5.0,
        "-d",
        "--min_distance",
        help="Minimum distance between peaks (only used if peak_method is not 'spotiflow')",
    ),
    temp_dir: Path = typer.Option(
        "", "-t", "--temp_dir", help="Temporary directory for intermediate files"
    ),
    xenium: bool = typer.Option(
        False,
        "--xenium/--no-xenium",
        help="Register to Xenium Image (True). If false assumes that target image was taken at same resolution as ISS",
    ),
    rolling_ball: bool = typer.Option(
        False, "--rolling_ball/--no-rolling_ball", help="Use rolling ball filter"
    ),
    rolling_ball_radius: int = typer.Option(
        10, "--rolling_ball_radius", help="Radius of rolling ball"
    ),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Save intermediate image files"
    ),
    threshold: str = typer.Option(
        "None,None,None,None",
        "--threshold",
        help="Threshold values for calling peaks in each channel (Cy3,594,Cy5,Cy5.5) (comma-separated, use 'None' to skip a channel)",
    ),
    dapi_channel: int = typer.Option(0, "--dapi_channel", help="DAPI channel index"),
):
    date, time, id = re.search(
        r"([0-9]+)_([0-9]+)_([0-9]+)",
        "_".join(img_zarr.name.split("_")[:-1]),
    ).groups()
    # Call the main function with all arguments
    imarr = da.from_zarr(img_zarr)
    pdf = main(
        img=imarr,
        peak_method=peak_method,
        transform_json=transform_json,
        sigma=sigma,
        grouping_radius=grouping_radius,
        min_distance=min_distance,
        temp_dir=temp_dir,
        xenium=xenium,
        rolling_ball=rolling_ball,
        rolling_ball_radius=rolling_ball_radius,
        save=save,
        threshold=threshold,
        dapi_channel=dapi_channel,
        date=date,
        time=time,
    )
    pdf.to_file(output_dir, driver="GeoJSON")


if __name__ == "__main__":
    app()
