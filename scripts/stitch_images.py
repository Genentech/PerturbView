# Script that will correct for flatfield distortion and align individual tiles and construct a WS image
import argparse
from pathlib import Path

import dask.array as da
import m2stitch
import pandas as pd
from basicpy import BaSiC

from utils import stitch_utils


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("img_dir", help="Directory of images")
    parser.add_argument(
        "-f",
        "--flatfield_smoothness",
        default=0.5,
        help="Smoothness of flatfield correction",
    )
    parser.add_argument(
        "-o", "--output_dir", default="None", help="Output name of stitched image."
    )
    parser.add_argument("-x", "--flip_x", default=False, help="Flip x coordinates")
    parser.add_argument("-y", "--flip_y", default=False, help="Flip y coordinates")
    args = parser.parse_args()

    return {
        "img_dir": args.img_dir,
        "out_name": args.output_dir,
        "smooth": float(args.flatfield_smoothness),
        "flip_x": args.flip_x,
        "flip_y": args.flip_y,
    }


def main():
    adict = parse_args()

    # make the output directory
    Path(adict["out_name"]).mkdir(parents=True, exist_ok=True)

    # Parse tile positions
    img_df = stitch_utils.get_images_dataframe(adict["img_dir"])
    img_df["y"] = -img_df["y"]

    # Get coords of every image
    img, rows, cols = stitch_utils.get_m2stitch_coords_grid_based(
        img_df, flip_x=adict["flip_x"], flip_y=adict["flip_y"]
    )
    img_df["row"] = rows
    img_df["col"] = cols

    img_df.index = pd.MultiIndex.from_frame(img_df.loc[:, ["row", "col"]])

    # Flatfield correction
    basic = BaSiC(get_darkfield=False, smoothness_flatfield=adict["smooth"])

    # Get DAPI image and flatten ilumination
    # Images are TCZYX
    # TODO: instead of max projecting can we use the most infocus plane?
    img = img.max(2)
    fimg = img[:, 0, :, :].compute()
    basic.fit(fimg)
    fimg = basic.transform(fimg)

    # TODO: Automatically extract the pixel size from the metadata
    # Find position of each dapi image
    result_df, _ = m2stitch.stitch_images(
        fimg,
        rows,
        cols,
        position_initial_guess=img_df.loc[:, ["y", "x"]],
        ncc_threshold=0.1,
        row_col_transpose=False,
    )

    result_df["tile_id"] = result_df.index
    result_df.index = pd.MultiIndex.from_frame(result_df.loc[:, ["row", "col"]])
    result_df = result_df.drop(["row", "col"], axis=1)
    result_df = result_df.join(img_df)

    result_df.to_csv(adict["out_name"] + "_stitching_results.csv")

    # Create stitched images
    stitched_dapi, stitched_mask = stitch_utils.stitch_tiles(fimg, result_df)

    all_chs = [
        stitched_dapi,
    ]
    # Create stitched images for all other channels (not DAPI) using the DAPI coordinates
    for ix in range(img.shape[1] - 1):
        cimg_corr = basic.transform(img[:, ix + 1, :, :])
        stitched_chan, _ = stitch_utils.stitch_tiles(cimg_corr, result_df)
        all_chs.append(stitched_chan)

    etk_out = da.stack(all_chs, axis=0)
    etk_out.to_zarr(adict["out_name"], overwrite=True)

    return


if __name__ == "__main__":
    main()
