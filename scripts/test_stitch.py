# %%
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# %%
from utils import stitch_utils

img_df = stitch_utils.get_images_dataframe(
    "/gnet/is1/p01/shares/ctg-microscopy/YC/DISCO-FISH/20240202_DISCO_SL1_top_AUTOFOCUS2/20240202_164726_211",
    ext=".nd2",
)
# img_df["y"] = -img_df["y"]
# %%
import pandas as pd
from basicpy import BaSiC

img, rows, cols = stitch_utils.get_m2stitch_coords_grid_based(img_df)
# WARNING: Z is flattened to 1 here

img_df.index = pd.MultiIndex.from_frame(img_df.loc[:, ["row", "col"]])

# Flatfield correction
basic = BaSiC(get_darkfield=False, smoothness_flatfield=0.5)

fimg = img[:, 0, :, :].compute()
basic.fit(fimg)

fimg = basic.transform(fimg)
# %%
# plot a low-res version of the image before stitching
# use the rows and cols to get the position of each image

# %%
import m2stitch

# Find position of each dapi image
result_df, _ = m2stitch.stitch_images(
    fimg,
    rows,
    cols,
    position_initial_guess=img_df.loc[:, ["y", "x"]]
    / 0.65,  # TODO: is this correct for the nd2 reader?
    ncc_threshold=0.01,
    row_col_transpose=False,
)
# %%

result_df["tile_id"] = result_df.index
result_df.index = pd.MultiIndex.from_frame(result_df.loc[:, ["row", "col"]])
result_df = result_df.drop(["row", "col"], axis=1)
result_df = result_df.join(img_df)

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

# %%
fimg.shape
# %%
img
# %%
# %%
img_dir = "/gnet/is1/p01/shares/ctg-microscopy/YC/DISCO-FISH/20240202_DISCO_SL1_top_AUTOFOCUS2/20240202_164726_211"
# stitch_utils.get_images_dataframe(img_dir)
from pathlib import Path

from aicsimageio import AICSImage

ims = Path(img_dir).glob("*.nd2")

# Get an AICSImage object
img = AICSImage(next(ims))
# image may have multiple planes, get the stage position for all planes
pos = [
    (plane.position_x, plane.position_y)
    for plane in img.ome_metadata.images[0].pixels.planes
]
# check that the stage position is the same for all planes
stage_pos = pos[0]
assert all(p == stage_pos for p in pos)

# aicsimageio.metadata.utils.get_coords_from_ome(img.ome_metadata, 0)

# %%
