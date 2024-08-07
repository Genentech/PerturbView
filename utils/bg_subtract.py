from pathlib import Path

import dask.array as da
from skimage.restoration import rolling_ball


def rolling_ball_dask(zimg: da.Array, radius: int):
    zimg_rechunked = zimg.rechunk((1, "auto", "auto")).astype("float32")

    def apply_rolling_ball(block):
        # The block will have shape (1, Y, X), but rolling_ball expects (Y, X), so we squeeze and unsqueeze the array
        return rolling_ball(block.squeeze(axis=0), radius=radius).reshape(
            (1, *block.shape[1:])
        )

    bg_im = zimg_rechunked.map_blocks(
        apply_rolling_ball,
        dtype=zimg_rechunked.dtype,
        chunks=zimg_rechunked.chunksize,
    ).compute()

    # convert back to dask array
    bg_im = da.from_array(bg_im)  # , chunks=zimg_rechunked.chunksize)

    bg_im2 = zimg_rechunked - bg_im

    bg_im2 = bg_im2.rechunk("auto").astype("float32")

    return bg_im2


def cli(image_in: Path, image_out: Path, radius: int):
    zimg = da.from_zarr(image_in)
    bg_im2 = rolling_ball_dask(zimg, radius).astype("uint16")
    bg_im2.to_zarr(image_out, overwrite=True)


if __name__ == "__main__":
    import typer

    typer.run(cli)
