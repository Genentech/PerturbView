from typing import Callable

import dask.array as da
import typer
from dask_image.ndfilters import laplace

app = typer.Typer()


# Focus functions
def laplacian_variance(image: da.Array) -> da.Array:
    """Calculate the variance of the Laplacian (sharpness) of the image."""
    laplacian = laplace(image)
    variance = laplacian.var()
    return variance


def sum_of_squares(image: da.Array) -> da.Array:
    """Calculate the sum of squared differences from the mean."""
    flattened = image.ravel()
    mean_intensity = flattened.mean()
    ssd = ((flattened - mean_intensity) ** 2).sum()
    return ssd


# Mapping of focus function names to their implementations
FOCUS_METRICS = {
    "sum_of_squares": sum_of_squares,
    "laplacian_variance": laplacian_variance,
}


def select_best_focus(
    img: da.Array, metric_func: Callable[[da.Array], da.Array]
) -> da.Array:
    if img.ndim != 4:
        raise ValueError(
            "Image dimensionality must be 4D (C,Z,Y,X) for independent channel focus selection."
        )

    best_focus_slices = []

    for channel_index in range(img.shape[0]):
        channel_img = img[channel_index, ...]  # Shape: (Z, Y, X)
        focus_metrics = da.stack(
            [
                metric_func(channel_img[z_slice, ...])
                for z_slice in range(channel_img.shape[0])
            ]
        )

        # best focus slice
        best_focus_index = da.argmax(focus_metrics)
        # get the array slice with the best focus
        best_focus_slice = da.take(channel_img, best_focus_index, axis=0)
        best_focus_slices.append(best_focus_slice)

    best_focus_img = da.stack(best_focus_slices, axis=0)
    return best_focus_img


@app.command()
def main(
    zarr_path: str = typer.Argument(
        ..., help="Path to the Zarr file containing the image data."
    ),
    focus_metric: str = typer.Option(
        "sum_of_squares",
        "--focus-metric",
        "-m",
        help="Focus metric to use for selecting the best focus slice.",
        case_sensitive=False,
    ),
    output_path: str = typer.Option(
        ..., "--output-path", "-o", help="Path to save the best focus image as Zarr."
    ),
):
    img = da.from_zarr(zarr_path)

    # Select the focus metric function based on user input
    if focus_metric in FOCUS_METRICS:
        metric_func = FOCUS_METRICS[focus_metric]
    else:
        raise typer.Exit(
            f"Focus metric '{focus_metric}' is not recognized. Available options are: {list(FOCUS_METRICS.keys())}"
        )

    # Perform focus selection
    best_focus_img = select_best_focus(img, metric_func=metric_func)

    # Save the resulting best focus image
    best_focus_img.to_zarr(output_path)
    typer.echo(f"Best focus image saved to {output_path}")


if __name__ == "__main__":
    app()
