# From Conrad for paper
# %%
import gc
import json
import os
import re
from pathlib import Path

import dask.array as da

# import dask
import m2stitch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import scipy as sp
import seaborn as sns
import SimpleITK as sitk

# import skimage as ski
import skimage as skim
import zarr
from aicsimageio import AICSImage
from aicsimageio.readers import OmeTiffReader
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from aicsimageio.writers.ome_zarr_writer import OmeZarrWriter
from basicpy import BaSiC
from sklearn.cluster import AgglomerativeClustering

# from skimage.color import rgb2hed
from tqdm import tqdm
from wsireg import WsiReg2D
from wsireg.reg_images.loader import reg_image_loader
from wsireg.reg_transforms import RegTransform, RegTransformSeq
from wsireg.utils.reg_utils import _prepare_reg_models, sitk_pmap_to_dict

# %%
# Load Xenium
xdir = "/gnet/is1/p01/shares/ctg-microscopy/EL/xenium/raw/20231027__231309__20231027_DLD1_multi/"

# ix = 1
# rdr = OmeTiffReader(xdir + 'output-XETG00043__0011577__12-s2__20231027__232106/morphology_mip.ome.tif')

# ix = 2
# rdr = OmeTiffReader(xdir + 'output-XETG00043__0011577__12-s3__20231027__232106/morphology_mip.ome.tif')

# ix = 3
# rdr = OmeTiffReader(xdir + 'output-XETG00043__0011577__14-s2__20231027__232106/morphology_mip.ome.tif')

ix = 4
rdr = OmeTiffReader(
    xdir + "output-XETG00043__0011577__14-s3__20231027__232106/morphology_mip.ome.tif"
)

imarr = rdr.get_image_dask_data("YX", C=0, Z=0, T=0)
# Input directory
wdir = "/gstore/data/ctgbioinfo/kudot3/eTK128/raw/"
sdir = wdir + f"eTK128_{ix}/"

# Stitching parameters
out_dir = f"/gstore/scratch/u/lubecke/etk_{ix}_stitching/"
slist = [x for x in os.listdir(sdir) if len(x.split("_")) > 1]

# Registration params for Xenium <-> ISS
reg_dir = f"/gstore/scratch/u/lubecke/etk_{ix}_stitching/registration/"
os.makedirs(reg_dir, exist_ok=True)

# Base calling results
os.makedirs(f"{out_dir}/base_calling/", exist_ok=True)

# Temporary directory for intermediate arrays.
temp_dir = "/gstore/scratch/u/lubecke/"

# Reference gRNA barcodes
ref_table_csv = "/gstore/scratch/u/lubecke/etk_1_stitching/gRNAs_mouseMinimal1.csv"

# %% [markdown]
# # Stitching and alignment
# This proceeds in 3 steps:
#
# 1. Correct the flatfield distortion using BaSiC
# 2. Stitch the resultant tiles using m2stitch; we feed the stage readout x/y coordinates as prior information to constrain the transform
# 3. Align each stitched DAPI image to the Xenium WSI DAPI image.
#
# Further refinements include restitching the aligned image by projecting each tile individually onto a masked Xenium image, or using the overlapped peak calls to refine the rigid distortion.

# %%

# %% [markdown]
# ## Util fns


# %%
def get_images_dataframe(img_dir):
    coord_exp = r"x([\+\-][0-9]+)_y([\+\-][0-9]+)_"
    imgs = [
        img_dir + x
        for x in os.listdir(img_dir)
        if (os.path.splitext(x)[-1] == ".tif") and re.search(coord_exp, x)
    ]

    img_df = pd.DataFrame(columns=["fpath", "x", "y"], index=np.arange(len(imgs)))
    img_df["fpath"] = imgs

    img_df["x"] = img_df["fpath"].apply(
        lambda x: float(re.search(coord_exp, x).groups()[0])
    )
    img_df["y"] = img_df["fpath"].apply(
        lambda x: float(re.search(coord_exp, x).groups()[1])
    )

    return img_df


def get_m2stitch_coords_varying_rows_cols(img_df, debug=False):
    rdr = OmeTiffReader(img_df.iloc[0]["fpath"])
    imarr = rdr.get_image_dask_data("CYX", Z=0, T=0)

    imzarr = da.zeros_like(
        None, shape=(img_df.shape[0],) + imarr.shape, dtype=imarr.dtype
    )

    # Group by one axis and get unique values in the other
    grouped_by_y = img_df.groupby("y")["x"].unique()
    grouped_by_x = img_df.groupby("x")["y"].unique()

    # Create mappings for row and column indices
    row_idx = {val: key for key, vals in enumerate(grouped_by_y) for val in vals}
    col_idx = {val: key for key, vals in enumerate(grouped_by_x) for val in vals}

    rows, cols = [], []
    cidx = 0
    for _, row in img_df.iterrows():
        rows.append(row_idx[row.x])
        cols.append(col_idx[row.y])

        rdr = OmeTiffReader(row.fpath)
        imzarr[cidx, :, :, :] = rdr.get_image_dask_data("CYX")
        cidx += 1

    return imzarr, rows, cols


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
        dtype=np.uint16,  # Hard coded cuz im lazy
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


def load_transforms(tpath):
    with open(tpath, "r") as f:
        tform_composite = json.load(f)

    full_tform_seq = RegTransformSeq()

    if tform_composite["pre_reg_transforms"] is not None:
        pre_tforms = [
            sitk_pmap_to_dict(tf) for tf in tform_composite["pre_reg_transforms"]
        ]

        pre_tforms_rt = [RegTransform(t) for t in pre_tforms]
        pre_tforms_idx = [0 for _ in pre_tforms_rt]

        pre_transforms = RegTransformSeq(pre_tforms_rt, pre_tforms_idx)
        full_tform_seq.append(pre_transforms)

    # Convert registration transform into WSIReg format.
    reg_tforms = [sitk_pmap_to_dict(tf) for tf in tform_composite["reg_transforms"]]

    reg_tforms_rt = [RegTransform(t) for t in reg_tforms]
    reg_tforms_idx = [0 for _ in reg_tforms_rt]

    transforms = RegTransformSeq(reg_tforms_rt, reg_tforms_idx)
    full_tform_seq.append(transforms)

    return full_tform_seq


# %% [markdown]
# ## Pipeline functions

# %%


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


import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aicsimageio.readers import OmeTiffReader
from matplotlib.colors import ListedColormap
from sklearn.cluster import AgglomerativeClustering


def get_m2stitch_coords_grid_based(img_df, debug=False, cluster_threshold=10):
    grid_size = estimate_grid_size(img_df, cluster_threshold)

    # Load the first image to get the shape
    rdr = OmeTiffReader(img_df.iloc[0]["fpath"])
    imarr = rdr.get_image_dask_data("CYX", Z=0, T=0)

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

    # If debug is True, plot the positions
    if debug:
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        # Generate a color map with a color for each point
        colors = ListedColormap(plt.cm.hsv(np.linspace(0, 1, len(img_df)))).colors

        # Original positions with colors
        axs[0].scatter(img_df["x"], img_df["y"], alpha=0.7, c=colors, edgecolor="black")
        axs[0].set_title("Original Positions")
        axs[0].set_xlabel("X position")
        axs[0].set_ylabel("Y position")

        # Inferred grid positions with jitter and the same colors
        jitter = np.random.normal(0, 0.05, size=img_df[["grid_x", "grid_y"]].shape)
        axs[1].scatter(
            img_df["grid_x"] + jitter[:, 0],
            img_df["grid_y"] + jitter[:, 1],
            alpha=0.7,
            c=colors,
            edgecolor="black",
        )
        axs[1].set_title("Inferred Grid Positions")
        axs[1].set_xlabel("Grid X")
        axs[1].set_ylabel("Grid Y")

        plt.tight_layout()
        plt.show()

    # Fill in the dask array with image data
    for idx, row in img_df.iterrows():
        rdr = OmeTiffReader(row.fpath)
        imzarr[idx, :, :, :] = rdr.get_image_dask_data("CYX")

    return imzarr, img_df["row"].tolist(), img_df["col"].tolist()


# Hardcoded assumptions : 0.65um/pix; DAPI is always C=0; 7 cols and 4 rows of stage coordinates; 0.5 smoothness
def stitch_etk_stack(img_dir, debug=False):
    # Load tile data
    img_df = get_images_dataframe(img_dir)
    img_df["y"] = -img_df["y"]

    img, rows, cols = get_m2stitch_coords_grid_based(img_df, debug=True)

    img_df["row"] = rows
    img_df["col"] = cols

    img_df.index = pd.MultiIndex.from_frame(img_df.loc[:, ["row", "col"]])

    # Flatfield correction
    basic = BaSiC(get_darkfield=False, smoothness_flatfield=0.5)

    fimg = img[:, 0, :, :].compute()
    basic.fit(fimg)

    fimg = basic.transform(fimg)

    # Calculate stitching parameters
    result_df, _ = m2stitch.stitch_images(
        fimg,
        rows,
        cols,
        position_initial_guess=img_df.loc[:, ["y", "x"]] / 0.65,
        ncc_threshold=0.01,
        row_col_transpose=False,
    )

    result_df["tile_id"] = result_df.index
    result_df.index = pd.MultiIndex.from_frame(result_df.loc[:, ["row", "col"]])
    result_df = result_df.drop(["row", "col"], axis=1)
    result_df = result_df.join(img_df)

    # Create stitched images
    stitched_dapi, stitched_mask = stitch_tiles(fimg, result_df)

    all_chs = [
        stitched_dapi,
    ]
    for ix in range(img.shape[1] - 1):
        cimg_corr = basic.transform(img[:, ix + 1, :, :])
        stitched_chan, _ = stitch_tiles(cimg_corr, result_df)
        all_chs.append(stitched_chan)

    etk_out = da.stack(all_chs, axis=0)

    return etk_out, stitched_mask, result_df, basic


# %% [markdown]
# ## Mass pipeline

# %%
from pathlib import Path

for fname in tqdm(slist):
    img, mask, rdf, ffmdl = stitch_etk_stack(
        str(Path(sdir).joinpath(fname)) + "/", debug=True
    )  # trailing slash is necessary

    fid = fname.split("_")[-1]

    img.to_zarr(f"{out_dir}eTK_1_{fid}_stitched.zarr", overwrite=True)
    mask.to_zarr(f"{out_dir}eTK_1_{fid}_mask.zarr", overwrite=True)
    rdf.to_csv(f"{out_dir}eTK_1_{fid}_stitch.csv")
    ffmdl.save_model(f"{out_dir}ffmodel_{fid}/", overwrite=True)


# %%
# # Post correction (not needed after fixing first one)
# for fname in tqdm(slist):
#     img_dir = sdir + f"/{fname}/"
#     fid = fname.split("_")[-1]

#     # Load tile data
#     img_df = get_images_dataframe(img_dir)
#     img_df["y"] = -img_df["y"]

#     img, rows, cols = get_m2stitch_coords(img_df, n_cols=7, n_rows=4)
#     img_df["row"] = rows
#     img_df["col"] = cols

#     img_df.index = pd.MultiIndex.from_frame(img_df.loc[:, ["row", "col"]])

#     # Redo this bc of load error in BaSiC
#     basic = BaSiC(get_darkfield=False, smoothness_flatfield=0.5)

#     fimg = img[:, 0, :, :].compute()
#     basic.fit(fimg)

#     fimg = basic.transform(fimg)

#     # Load stitch results
#     result_df = pd.read_csv(f"{out_dir}eTK_1_{fid}_stitch.csv")
#     result_df = result_df.drop("Unnamed: 0", axis=1)
#     result_df["tile_id"] = result_df.index

#     result_df.index = pd.MultiIndex.from_frame(result_df.loc[:, ["row", "col"]])
#     result_df = result_df.drop(["row", "col"], axis=1)

#     result_df = result_df.join(img_df, lsuffix="_R")

#     # Create stitched images
#     stitched_dapi, stitched_mask = stitch_tiles(fimg, result_df)

#     all_chs = [
#         stitched_dapi,
#     ]
#     for ix in range(img.shape[1] - 1):
#         cimg_corr = basic.transform(img[:, ix + 1, :, :])
#         stitched_chan, _ = stitch_tiles(cimg_corr, result_df)
#         all_chs.append(stitched_chan)

#     etk_out = da.stack(all_chs, axis=0)
#     etk_out.to_zarr(f"{out_dir}eTK_1_{fid}_stitched.zarr", overwrite=True)

# %% [markdown]
# ## Align to Xenium GT

# %%
ids = [x.split("_")[-1] for x in slist]
imgs = [da.from_zarr(f"{out_dir}eTK_1_{x}_stitched.zarr") for x in ids]

# %%
# Target image (Xenium) to align to
hi, lo = da.percentile(imarr.ravel(), [2, 98])
narr = (imarr > lo) * (imarr - lo) / (hi - lo)
narr = (narr * 32768).astype(np.int16)
np.save(reg_dir + "tgt_img.npy", narr.compute())

if_dict = dict(
    image_fp=reg_dir + "tgt_img.npy",
    image_res=rdr.physical_pixel_sizes.X,
    channel_names=["he"],
    channel_colors=["red"],
    preprocessing={
        "image_type": "FL",
        "ch_indices": [0],
        "as_uint8": False,
        "contrast_enhance": False,
    },
)

# %%
# Source images (ISS over time)
he_dicts = []
for id, img in zip(ids, imgs):
    # Normalize image
    hi, lo = da.percentile(img[0, :, :].ravel(), [2, 98])
    narr = (img[0, :, :] > lo) * (img[0, :, :] - lo) / (hi - lo)
    narr = (narr * 32768).astype(np.int16)
    np.save(reg_dir + f"src_img_{id}.npy", narr.compute())

    # Params
    he_dict = dict(
        image_fp=reg_dir + f"src_img_{id}.npy",
        image_res=0.65,
        channel_names=["he"],
        channel_colors=["red"],
        preprocessing={
            "image_type": "FL",
            "ch_indices": [
                0
            ],  # Assume DAPI is channel 0 (but have some metadata parsing if it's available)...
            "as_uint8": False,
            "contrast_enhance": False,
            #'flip': 'h', #Add flip
            "rot_cc": -90,  # Add rotation
        },
    )

    he_dicts.append(he_dict)


# %%
# Registration graphs
def construct_graph(if_dict, he_dict, output_dir):
    reg_graph = WsiReg2D()

    reg_graph.add_modality(
        "HE", **{x: he_dict[x] for x in he_dict if not re.search("mask", x)}
    )

    reg_graph.add_modality(
        "IF", **{x: if_dict[x] for x in if_dict if not re.search("mask", x)}
    )

    reg_graph.add_reg_path("HE", "IF", thru_modality=None, reg_params=["rigid", "nl"])

    reg_graph.add_merge_modalities("aligned", ["HE", "IF"])
    reg_graph.cache_images = False

    # Get the default parameter dictionary from the WSIReg graph.
    reg_params = _prepare_reg_models(
        reg_graph.reg_graph_edges[0]["params"]
    )  # Here we are loading one of the registration tasks within the graph.
    # reg_params[1]['MaximumStepLength'] = ['30','30','20','20','20','10','10','10','1','1'] #Custom parameter sweep for nonlinear

    # Custom parameter changes
    for pmap in reg_params:
        pmap["UseMultiThreadingForMetrics"] = ["false"]
        pmap["MaximumNumberOfIterations"] = [
            "5000"
        ]  # This was key for this alignment task. Unfortunately this means it takes 1 hour to align instead of 10mins...
        pmap["MaximumNumberOfSamplingAttempts"] = ["1000"]
        # pmap['FixedImagePyramid'] = ['FixedSmoothingImagePyramid']
        # pmap['MovingImagePyramid'] = ['MovingSmoothingImagePyramid']
        # pmap['AutomaticTransformInitialization'] = ['false']

    with open(output_dir + "reg_params.json", "w") as f:
        json.dump(reg_params, f)

    return reg_params


for id, hdict in zip(ids, he_dicts):
    os.makedirs(f"{reg_dir}/slide{id}/", exist_ok=True)
    rparams = construct_graph(if_dict, hdict, reg_dir + f"/slide{id}/")
    with open(f"{reg_dir}/slide{id}/src_params.json", "w") as f:
        json.dump(hdict, f)

with open(f"{reg_dir}/tgt_params.json", "w") as f:
    json.dump(if_dict, f)

# %%
import subprocess

for id in ids:
    subprocess.run(
        [
            "sbatch",
            "general_registration.sh",  # Baysor script
            reg_dir,
            reg_dir + f"/slide{id}/src_params.json",
            reg_dir + "tgt_params.json",
            reg_dir + f"/slide{id}/reg_params.json",
            reg_dir + f"/slide{id}/",
        ]
    )

# %% [markdown]
# ## Construct fully aligned DAPI stack
# Needed for intensity correction.

# %%
from pathlib import Path

rdr = OmeTiffReader(
    xdir + "output-XETG00043__0011577__12-s2__20231027__232106/morphology_mip.ome.tif"
)

zstore = zarr.DirectoryStore(out_dir + "DAPI_aligned.zarr", dimension_separator="/")
zimg = zarr.zeros(
    shape=(len(ids),) + rdr.get_image_dask_data("YX", C=0, Z=0, T=0).shape,
    store=zstore,
    chunks=(1, 8192, 8192),
    dtype=np.uint16,
)

for ix, (id, img) in enumerate(zip(ids, imgs)):
    tform = load_transforms(Path(reg_dir).joinpath(f"slide{id}/transforms.json"))

    r_img = reg_image_loader(
        img[0, :, :],
        0.65,
    )
    r_img.preprocessing.as_uint8 = False
    r_img.preprocessing.max_int_proj = False
    r_img.read_reg_image()

    timg = tform.resampler.Execute(r_img.reg_image)
    timg = sitk.GetArrayFromImage(timg)

    zimg[ix, :, :] = timg

# %%
from aicsimageio.writers import OmeZarrWriter

writer = OmeZarrWriter(
    f"{out_dir}/dapi_reg_v3_z.ome.zarr",
)

writer.write_image(
    zimg,
    image_name="DAPI",
    physical_pixel_sizes=rdr.physical_pixel_sizes,
    channel_names=["DAPI"],
    channel_colors=None,
    dimension_order="ZYX",
    chunk_dims=(1, 8192, 8192),
    scale_factor=2,
    scale_num_levels=5,
)

# %% [markdown]
# # Base calling
# Adapted from Eric's file

# %%

# %%
import gc
import json
import os
import re

import dask.array as da
import numpy as np
import pandas as pd
import seaborn as sns
import trackpy as tp
import zarr
from dask.diagnostics import ProgressBar
from dask_image import ndfilters as dask_ndi

# %%
from dask_image.ndmeasure import label
from dask_image.ndmorph import binary_dilation
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scry.reads import _channel_crosstalk_matrix
from skimage.feature import peak_local_max
from tqdm.auto import tqdm

# %% [markdown]
# ## Functions


# %%
def get_af_factors(img, af_img):
    corr_factor = np.zeros(img.shape[:2])
    for t in tqdm(range(af_img.shape[0])):
        af_signal = af_img[t, ...].ravel()[:, np.newaxis]
        for c in range(img.shape[1]):
            csignal = img[t, c, ...].ravel()
            p, _, _, _ = np.linalg.lstsq(
                af_signal[::100].compute(), csignal[::100].compute(), rcond=None
            )  # No need to use all the data
            corr_factor[t, c] = p

    return corr_factor


def correct_af(img, af_img, af_coeffs):
    imgs = []
    for t in range(img.shape[0]):
        timgs = []
        for c in range(img.shape[1]):
            timgs.append(
                (img[t, c, ...] - af_coeffs[t, c] * af_img[t, ...]).astype(np.float32)
            )
        timgs = da.stack(timgs, axis=0)
        imgs.append(timgs)
    img = da.stack(imgs, axis=0)

    return img * (img > 0)


# %%
# Detects peaks in an image
def detect_peaks(
    img, sigma=3, min_distance=15, threshold=None, temp_dir="/gstore/scratch/u/lubecke/"
):
    img = img.astype(np.float32)

    # Morphological footprint for peak detection
    size = 2 * min_distance + 1
    selem = np.ones((size,) * 2, dtype=bool)  # Assume 2 spatial dimensions

    border_width = skim.feature.peak._get_excluded_border_width(
        img[0, 0, :, :], min_distance, exclude_border=True
    )

    # Calculate peak intensity thresholds if we need to.
    if threshold is None:
        print("Calculating thresholds...")
        threshold = np.zeros(img.shape[:2])
        for t in tqdm(range(img.shape[0])):
            for c in range(img.shape[1]):
                cthresh = skim.filters.threshold_multiotsu(
                    img[t, c, ::min_distance, ::min_distance].compute(), classes=4
                )  # Downsample for speed
                threshold[t, c] = cthresh[1]
    else:
        threshold = np.ones(img.shape[:2]) * threshold

    # Temp zarr store for peak mask. This is a very large image (even as a bool), so we stream to disk.
    peak_zarr = zarr.TempStore(
        suffix="tmp2",
        prefix="peakmask",
        dir=temp_dir,
    )
    peaks = zarr.create(
        store=peak_zarr, shape=img.shape, dtype=bool, chunks=(1, 1, 2048, 2048)
    )

    # LoG-max filter - just do it in 2D.
    print("Detecting raw peaks...")
    # Due to channel crosstalk and non-uniform background intensity, I am restricting this to 2D
    # For this reason, we have to look at the negative d^2/dx^2 to highlight local maxima.
    # There is some weirdness where dask takes >30mins to generate the computation graph if I don't compute...probably a lot of edge pruning
    for t in tqdm(range(img.shape[0])):
        for c in range(img.shape[1]):
            cloged = -dask_ndi.gaussian_laplace(
                img[t, c, :, :].astype(np.float32), sigma=sigma
            )
            logmax = dask_ndi.maximum_filter(
                cloged,
                footprint=selem,
                mode="nearest",
            )

            peaks[t, c, ...] = (cloged == logmax).compute()
    gc.collect()

    # Get peaks (see skimage docs)
    print("Filtering peaks...")
    peak_arr = da.from_zarr(peaks)
    for t in tqdm(range(img.shape[0])):
        for c in range(img.shape[1]):
            peaks[t, c, ...] = (
                peak_arr[t, c, ...] & (img[t, c, ...] > threshold[t, c])
            ).compute()
    gc.collect()

    # Hardcode 2D spatial border exclusion region
    peak_arr[..., slice(None, border_width[0]), :] = 0
    peak_arr[..., slice(-border_width[0], None), :] = 0

    peak_arr[..., slice(None, border_width[1])] = 0
    peak_arr[..., slice(-border_width[1], None)] = 0

    return peak_arr


# %%
# Function that groups peaks across channels (but not time)
def label_peaks(
    peak_mask,
    grouping_radius=30,
):
    # The structuring element is a flat disk
    selem = skim.morphology.disk(radius=grouping_radius)

    # For each peak, we dilate a disk around it; overlapping disks are then merged.
    label_image = da.zeros_like(
        None,
        shape=(peak_mask.shape[0],) + peak_mask.shape[-2:],
        dtype=np.uint32,
    )

    print("Labelling peaks...")
    for t in tqdm(range(peak_mask.shape[0])):
        # Dilate peaks
        peak_dilated = binary_dilation(
            peak_mask[t, ...].max(axis=0),
            selem,
        )

        # Label peaks
        peak_labels, n_peaks = label(
            peak_dilated,
        )

        label_image[t, ...] = peak_labels

    return label_image


# %%
# Get the peak properties
def peak_props(peak_labels, img):
    peaks = []
    for t in tqdm(range(peak_labels.shape[0])):
        limg = peak_labels[t, :, :].compute()
        mimg = img[t, :, :, :].compute().transpose((1, 2, 0))

        rprops = skim.measure.regionprops(
            limg,
            mimg,
            cache=False,
        )

        peaks_df = pd.DataFrame(columns=["area", "x", "y"])
        # Area (used for poor alignment quality across time)
        peaks_df["area"] = [x["area"] for x in rprops]

        # Centroid coordinates (weighted by intensity)
        coords = [x["centroid_weighted"] for x in rprops]
        peaks_df["y"] = [np.nanmean(x[0]) for x in coords]
        peaks_df["x"] = [np.nanmean(x[1]) for x in coords]

        # Peak intensities per channel
        peak_intensities = [x["intensity_max"] for x in rprops]
        n_chans = np.max([len(x) for x in peak_intensities])
        for i in range(n_chans):
            peaks_df[f"ch_{i}"] = [x[i] for x in peak_intensities]

        # Time
        peaks_df["time"] = t

        peaks.append(peaks_df)

        del limg, mimg
        gc.collect()

    return pd.concat(peaks, ignore_index=True, join="outer")


# %% [markdown]
# ## Pipeline


# %%
# This uses a max filter on the LoG to detect peaks
# Also possible to use white tophat instead of LoG so we're using ints instad of floats for memory efficiency
def find_peaks(
    img,
    log_sigma=3,  # The std of the Gaussian used for LoG filter (or radius of selem for white tophat)
    int_threshold=None,  # Intensity threhsold for peaks; the original image is used. Automatically determined if not provided.
    min_dist=10,  # The size of the maximum filter structuring element (larger = sparser peaks)
    grouping_radius=7,  # Peaks that are less than grouping_radius apart (in space, across all channels) are merged
    temp_dir="/gstore/scratch/u/lubecke/",  # Scratch directory to hold larger-than-memory intermediate arrays.
):
    # Binary mask of peaks.
    peak_arr = detect_peaks(
        img,
        sigma=log_sigma,
        min_distance=min_dist,
        temp_dir=temp_dir,
        threshold=int_threshold,
    )

    # Label individual peaks
    peak_labels = label_peaks(peak_arr, grouping_radius)

    # This is significantly faster than computing per slice on the fly (as labels are shared across time)
    print("Caching labels...")
    lstore = zarr.TempStore(
        prefix="label",
        suffix="tmp",
        dir=temp_dir,
    )
    with ProgressBar():
        peak_labels.to_zarr(lstore, overwrite=True)
    peak_labels = da.from_zarr(lstore)
    gc.collect()

    # Get regionprops of peaks.
    all_peaks = peak_props(peak_labels, img)
    return all_peaks


# %% [markdown]
# ## Run

# %%
# Create aligned transcript images for base calling
for img_name in tqdm(slist):
    fid = img_name.split("_")[-1]

    img = da.from_zarr(f"{out_dir}eTK_1_{fid}_stitched.zarr")

    zstore = zarr.DirectoryStore(
        f"{out_dir}eTK_1_{fid}_aligned.zarr", dimension_separator="/"
    )
    zimg = zarr.zeros(
        shape=(img.shape[0] - 1,)
        + rdr.get_image_dask_data(
            "YX", Z=0, T=0
        ).shape,  # rdr is going to be loaded from above.
        store=zstore,
        chunks=(1, 2048, 2048),
        dtype=np.uint16,
    )

    tform = load_transforms(f"{reg_dir}/slide{fid}/transforms.json")

    for cix in range(img.shape[0] - 1):
        r_img = reg_image_loader(
            img[cix + 1, :, :],
            0.65,
        )
        r_img.preprocessing.as_uint8 = False
        r_img.preprocessing.max_int_proj = False
        r_img.read_reg_image()

        timg = tform.resampler.Execute(r_img.reg_image)
        timg = sitk.GetArrayFromImage(timg)

        zimg[cix, :, :] = timg

# %%
# Load the transformed images
imgs = []
ordering = []

# Hardcoded on my end. But maybe just parse datetime strings at some point as N -> large
# etk_1
temporal_order = {"160": 0, "597": 1, "788": 2, "257": 3, "176": 4, "862": 5}

for img_name in tqdm(slist):
    fid = img_name.split("_")[-1]
    imgs.append(da.from_zarr(f"{out_dir}eTK_1_{fid}_aligned.zarr"))
    ordering.append(temporal_order[fid])

base_img = da.stack(imgs, axis=0)
base_img = base_img[np.argsort(ordering), ...]
del imgs
gc.collect()

# Correct for DAPI signal bleedthrough
dapi_img = AICSImage(f"{out_dir}dapi_reg_v3_z.ome.zarr").get_image_dask_data("ZYX")

# %%
# Correct DAPI bleedthrough
print("Calculating correction factors...")
factors = get_af_factors(base_img, dapi_img)

print("Filtering ISS...")
cimg = correct_af(base_img.astype(np.float32), dapi_img.astype(np.float32), factors)

with ProgressBar():
    cimg.to_zarr(f"{out_dir}/base_calling/bg_corrected.zarr", overwrite=True)
cimg = da.from_zarr(f"{out_dir}/base_calling/bg_corrected.zarr")
gc.collect()

# %%
cimg = da.from_zarr(f"{out_dir}/base_calling/bg_corrected.zarr")

# Get peaks
all_peaks = find_peaks(
    cimg,
    log_sigma=5,  # Gaussian sigma for derivatives
    int_threshold=None,
    min_dist=7,  # Minimum distance between peaks (note that this isn't exactly true for my version if 2 peaks are exactly the same intensity within min_dist.)
    grouping_radius=7,  # Radius of element that groups peaks together
    temp_dir=temp_dir,
)

# all_peaks.to_csv(f'{out_dir}/base_calling/peak_props_v2.csv')

# %%
import geopandas
import shapely

# %%
all_peaks.to_csv(f"{out_dir}/base_calling/peak_props_v3.csv")

# %%
# Add NCC information from Raj's notebook
ncc_df = pd.read_csv(f"{out_dir}/base_calling/ncc_alignment.csv")
ncc_df = ncc_df.set_index("label").drop("Unnamed: 0", axis=1)
ncc_df = ncc_df.fillna(-1)

ncc_geometries = geopandas.GeoSeries(
    data=[
        shapely.Polygon(
            [
                (x["bbox-1"], x["bbox-0"]),
                (x["bbox-3"], x["bbox-0"]),
                (x["bbox-3"], x["bbox-2"]),
                (x["bbox-1"], x["bbox-2"]),
            ]
        )
        for _, x in ncc_df.iterrows()
    ],
    index=ncc_df.index,
)

ncc_gd = geopandas.GeoDataFrame(ncc_df, geometry=ncc_geometries)

ncc_gs = geopandas.GeoDataFrame(
    data=ncc_gd.loc[:, [f"xenium_iss{ix}_ncc" for ix in range(6)]].min(axis=1),
    geometry=ncc_geometries,
)

# %%
# GeoPandas for peaks
peak_geometries = geopandas.GeoSeries(
    data=[shapely.Point([x.x, x.y]) for _, x in all_peaks.iterrows()],
    index=all_peaks.index,
)

peak_gd = geopandas.GeoDataFrame(all_peaks, geometry=peak_geometries)
peak_gd["ncc"] = -2.0

# Transfer NCC information
for _, pgon in tqdm(ncc_gs.iterrows(), total=ncc_gs.shape[0]):
    peak_gd.loc[pgon.geometry.contains(peak_gd.geometry), "ncc"] = pgon[0]

# %%
peak_gd.to_file(f"{out_dir}peaks_v3.geojson", driver="GeoJSON")
all_peaks = peak_gd.copy()

# %% [markdown]
# ## Viz

# %%
cimg = da.from_zarr(f"{out_dir}/base_calling/bg_corrected.zarr")
plt.imshow(
    cimg[1, :, 5000:6000, 3000:4000].max(axis=0),
)
plt.colorbar()
plt.show()

# %%
f, a = plt.subplots(1, 5)
f.set_size_inches(10, 30)
for ix in range(5):
    a[ix].imshow(base_img[ix, 3, 5000:6000, 3000:4000], vmax=4000, vmin=0)
    a[ix].set_xticks([])
    a[ix].set_yticks([])
    a[ix].set_title(f"Time={ix}")
plt.show()

# %%
f, a = plt.subplots(1, 5)
f.set_size_inches(10, 30)
for ix in range(5):
    a[ix].imshow(dapi_img[ix, 5000:6000, 3000:4000], vmax=100, vmin=0)
    a[ix].set_xticks([])
    a[ix].set_yticks([])
    a[ix].set_title(f"Time={ix}")
plt.show()

# %%
f, a = plt.subplots(1, 5)
f.set_size_inches(10, 30)
for ix in range(5):
    a[ix].imshow(base_img[ix, 2, ::10, ::10].T, vmax=4000)
    a[ix].set_xticks([])
    a[ix].set_yticks([])
    a[ix].set_title(f"Time={ix}")
plt.show()

# %% [markdown]
# # Track peaks across time

# %%
import trackpy as tp

# %%
# Params
thresholded_intensity = 0
high_intensity_threshold = 0
search_radius = 15

reference_table = pd.read_csv(ref_table_csv, index_col=0)
reference_barcodes = reference_table.spacer.str[:6]
reference_barcode_arrays = np.array(
    [[ord(char) for char in barcode] for barcode in reference_barcodes]
)

# %%
ccols = [f"ch_{ix}" for ix in range(4)]
all_peaks["max_int"] = all_peaks.loc[:, ccols].max(axis=1)

# Track particles
filt_peaks = all_peaks.loc[all_peaks.max_int > thresholded_intensity, :]
tracked_particles = tp.link(filt_peaks, search_radius, t_column="time", memory=1)

# Find particles/barcodes that exist across time.
frequent_particles = tracked_particles["particle"].value_counts()
frequent_particles = frequent_particles[frequent_particles >= 5].index
frequent_particle_tracks = tracked_particles.loc[
    tracked_particles.particle.isin(frequent_particles), :
]

# Remove channel crosstalk using scry
# Didn't see much change in the fit even with the low intensity peaks. But maybe I need to just exhaustively test this.
high_intensity_tracks = frequent_particle_tracks[
    (frequent_particle_tracks.loc[:, ccols] > high_intensity_threshold).any(axis=1)
]

crosstalk_matrix = _channel_crosstalk_matrix(high_intensity_tracks.loc[:, ccols].values)
compensated_data = crosstalk_matrix.dot(
    frequent_particle_tracks.loc[:, ccols].values.T
).T
compensated_particle_tracks = frequent_particle_tracks.copy()
compensated_particle_tracks.loc[:, ccols] = compensated_data

# Add some info for each peak.
compensated_particle_tracks["max_channel"] = compensated_particle_tracks.loc[
    :, ccols
].idxmax(axis=1)
compensated_particle_tracks["max_intensity"] = compensated_particle_tracks.apply(
    lambda x: x[x.max_channel], axis=1
)

# Reconstruct particles
particle_df = pd.DataFrame(
    index=np.unique(compensated_particle_tracks.particle),
    columns=[
        "x",
        "y",
        "ncc",
        "barcode",
    ]
    + [f"i_{ix}" for ix in range(6)],
)

pgroups = compensated_particle_tracks.groupby("particle")

particle_df.loc[:, "x"] = pgroups.x.mean()
particle_df.loc[:, "y"] = pgroups.y.mean()
particle_df.loc[:, "ncc"] = pgroups.ncc.min()

channel_map = {
    "ch_0": "G",
    "ch_1": "T",
    "ch_2": "A",
    "ch_3": "C",
    "0": "N",
}  # This is also hardcoded. But prob can get by parsing names again.

bcodes = (
    compensated_particle_tracks.groupby(["time", "particle"])["max_channel"]
    .first()
    .unstack()
)
bcodes = bcodes.fillna("0")
bcodes = bcodes.apply(lambda x: "".join([channel_map[z] for z in x]))

particle_df.loc[:, "barcode"] = bcodes

max_int = (
    compensated_particle_tracks.groupby(["time", "particle"])["max_intensity"]
    .first()
    .unstack()
    .T
)
for c in max_int:
    particle_df.loc[:, f"i_{c}"] = max_int[c]

# Call barcodes
observed_barcode_arrays = np.array(
    [[ord(char) for char in barcode] for barcode in particle_df["barcode"]]
)

hamming_distances = cdist(
    observed_barcode_arrays, reference_barcode_arrays, metric="hamming"
)
min_hamming_distances = hamming_distances.min(axis=1) * len(reference_barcodes[0])
closest_matches = hamming_distances.argmin(axis=1)
matched_barcodes = reference_barcodes.values[closest_matches]

particle_df["matched_barcode"] = matched_barcodes
particle_df["hamming_distance"] = min_hamming_distances

print(
    f' Call Rate: {100*len(particle_df.query("hamming_distance < 2")) / len(particle_df)}'
)
passed = particle_df.query("hamming_distance < 2")
print(f"Passed: {len(passed)}")

# %%
# Convert to geopandas
called_peaks = particle_df
peak_gd = geopandas.GeoSeries(
    data=[shapely.Point([x.x, x.y]) for _, x in called_peaks.iterrows()],
    index=called_peaks.index,
)
called_peaks = called_peaks.drop(["x", "y"], axis=1)

called_peaks = geopandas.GeoDataFrame(data=called_peaks, geometry=peak_gd)

# %%
called_peaks.to_file(
    f"{out_dir}/base_calling/peaks_called_v3.geojson", driver="GeoJSON"
)

# %% [markdown]
# ## Viz

# %%
sns.pairplot(
    frequent_particle_tracks.iloc[:3000][ccols], plot_kws={"alpha": 0.3}
).fig.suptitle("Initial data")

# %%
sns.pairplot(
    compensated_particle_tracks.sample(5000).loc[:, ccols], plot_kws={"alpha": 0.3}
).fig.suptitle("Compensated data")


# %%
called_peaks.plot(
    "hamming_distance",
    alpha=0.7,
    s=1,
)
plt.title("Hamming distance")

# %%
called_peaks.plot("ncc", alpha=0.3, s=0.1, vmin=0.90, vmax=1, cmap="coolwarm")
plt.title("NCC")

# %%
sns.kdeplot(data=called_peaks, x="ncc", hue="hamming_distance", common_norm=False)

# %% [markdown]
# # ROI NL refinement

# %%
from skimage.morphology import binary_dilation

# %%
tile_arr = basic.transform(img_perturb[:, 0, :, :])

for ix in tqdm(range(28)):
    size_y = tile_arr.shape[1]
    size_x = tile_arr.shape[2]
    row = result_df.loc[ix]

    stitch_size = (
        result_df["y_pos"].max() + size_y,
        result_df["x_pos"].max() + size_x,
    )

    stitched_mask = da.empty_like(None, shape=stitch_size, dtype=np.uint8)
    stitched_mask[
        max(row.y_pos, 0) : min(row.y_pos + size_y, stitch_size[0]),
        max(row.x_pos, 0) : min(row.x_pos + size_x, stitch_size[1]),
    ] = 1

    stitched_image = da.empty_like(stitched_mask, dtype=np.uint16)
    hi, lo = da.percentile(tile_arr[ix, :, :].ravel(), [5, 98])
    stitched_image[
        max(row.y_pos, 0) : min(row.y_pos + size_y, stitch_size[0]),
        max(row.x_pos, 0) : min(row.x_pos + size_x, stitch_size[1]),
    ] = (tile_arr[ix, :, :] > lo) * (tile_arr[ix, :, :] - lo) / (hi - lo)

    # Handle mask
    r_img = reg_image_loader(stitched_mask, 0.65)
    r_img.read_reg_image()
    # Apply transform
    trans_image = full_tform_seq.resampler.Execute(r_img.reg_image)
    trans_image = sitk.GetArrayFromImage(trans_image)
    trans_image = (trans_image > 0).astype(np.uint8)

    np.save(f"{output_dir}/tgt_mask_{ix}.npy", trans_image)
    del trans_image
    gc.collect()

    # Handle image
    r_img = reg_image_loader(stitched_image, 0.65)
    r_img.read_reg_image()
    # Apply transform
    trans_image = full_tform_seq.resampler.Execute(r_img.reg_image)
    trans_image = sitk.GetArrayFromImage(trans_image)

    np.save(f"{output_dir}/src_img_{ix}.npy", trans_image)
    del trans_image
    gc.collect()

# %%
rdr = OmeTiffReader(
    xdir + "output-XETG00043__0011577__12-s2__20231027__232106/morphology_mip.ome.tif"
)
fov_dicts = []
for ix in range(28):
    he_dict = dict(
        image_fp=f"{output_dir}/src_img_{ix}.npy",
        image_res=rdr.physical_pixel_sizes.X,
        channel_names=["membrane"],
        channel_colors=["blue"],
        preprocessing={
            "image_type": "FL",
            "ch_indices": [0],
            "as_uint8": False,
            "contrast_enhance": False,
            #'rot_cc':-90,
        },
        image_mask=f"{output_dir}/tgt_mask_{ix}.npy",
    )

    with open(output_dir + f"/src_params_{ix}.json", "w") as f:
        json.dump(he_dict, f)
    fov_dicts.append(he_dict)

fov_tgt_dicts = []
for ix in range(28):
    if_dict = dict(
        image_fp=f"{output_dir}/tgt_img.npy",
        image_res=rdr.physical_pixel_sizes.X,
        channel_names=["he"],
        channel_colors=["red"],
        preprocessing={
            "image_type": "FL",
            "ch_indices": [0],
            "as_uint8": False,
            "contrast_enhance": False,
        },
        # image_mask=f'{output_dir}/tgt_mask_{ix}.npy',
    )

    with open(output_dir + f"/tgt_params_{ix}.json", "w") as f:
        json.dump(if_dict, f)
    fov_tgt_dicts.append(if_dict)

# %%
for ix in range(len(fov_dicts)):
    os.makedirs(f"{s_dir}/ROI{ix}/", exist_ok=True)
    rparams = construct_graph(
        fov_tgt_dicts[ix], fov_dicts[ix], output_dir + f"/ROI{ix}/"
    )

    rparams[0]["MaximumStepLength"] = [
        "10",
        "9",
        "7",
        "5",
        "4",
        "3",
        "2",
        "1",
        "1",
        "1",
    ]
    # rparams[0]['ImageSampler'] = ['Full']

    with open(s_dir + f"/ROI{ix}/reg_params.json", "w") as f:
        json.dump(rparams, f)

    subprocess.run(
        [
            "sbatch",
            "/gne/data/t3imagedata/conrad_store/modules/slurm_scripts/general_registration.sh",  # Baysor script
            s_dir,
            s_dir + f"src_params_{ix}.json",
            s_dir + f"tgt_params_{ix}.json",
            s_dir + f"/ROI{ix}/reg_params.json",
            s_dir + f"/ROI{ix}/",
        ]
    )


# %%
def load_transforms(tpath):
    with open(tpath, "r") as f:
        # with open('/gstore/scratch/u/lubecke/H2023_307/S0_transforms_v3.json','r') as f:
        tform_composite = json.load(f)

    full_tform_seq = RegTransformSeq()

    if tform_composite["pre_reg_transforms"] is not None:
        pre_tforms = [
            sitk_pmap_to_dict(tf) for tf in tform_composite["pre_reg_transforms"]
        ]

        pre_tforms_rt = [RegTransform(t) for t in pre_tforms]
        pre_tforms_idx = [0 for _ in pre_tforms_rt]

        pre_transforms = RegTransformSeq(pre_tforms_rt, pre_tforms_idx)
        full_tform_seq.append(pre_transforms)

    # Convert registration transform into WSIReg format.
    reg_tforms = [sitk_pmap_to_dict(tf) for tf in tform_composite["reg_transforms"]]

    reg_tforms_rt = [RegTransform(t) for t in reg_tforms]
    reg_tforms_idx = [0 for _ in reg_tforms_rt]

    transforms = RegTransformSeq(reg_tforms_rt, reg_tforms_idx)
    full_tform_seq.append(transforms)

    return full_tform_seq


# %%
mfi = tile_arr.sum(axis=2).sum(axis=1).compute()
plt.plot(mfi, "-bo")
plt.plot(
    [0, 1, 3, 8, 9, 13, 15, 16, 21, 26], mfi[[0, 1, 3, 8, 9, 13, 15, 16, 21, 26]], "-ro"
)

# %%
plt.imshow(tile_arr[5, :, :])
plt.show()

# %%
plt.imshow(tgt_img[::10, ::10])

# %%
[
    x
    for x in os.listdir(s_dir)
    if os.path.isdir(s_dir + "/" + x)
    and os.path.exists(s_dir + "/" + x + "/transforms.json")
]

# %%
tform = load_transforms(s_dir + "/ROI20/transforms.json")

# %%
src_img = np.load(s_dir + "/src_img_0.npy")
r_img = reg_image_loader(
    src_img,
    rdr.physical_pixel_sizes.X,
)
r_img.read_reg_image()

trans_image = tform.resampler.Execute(r_img.reg_image)
trans_image = sitk.GetArrayFromImage(trans_image)

# %%
f = plt.figure(dpi=300)
plt.imshow(trans_image[5000:10000:10, 10000:15000:10], alpha=0.7, cmap="inferno")
plt.imshow(tgt_img[5000:10000:10, 10000:15000:10], alpha=0.3)
plt.show()

# %%
tgt_img = np.load(s_dir + "tgt_img.npy")
plt.imshow(np.sqrt(tgt_img[::10, ::10]))
plt.show()

# %%
smask = np.load(s_dir + "tgt_mask_0.npy")
plt.imshow(smask[::10, ::10])
plt.show()

# %%
tgt_img = np.load(s_dir + "tgt_img.npy")
plt.imshow(tgt_img[::10, ::10])


# %%
def construct_graph(if_dict, he_dict, output_dir):
    reg_graph = WsiReg2D()

    reg_graph.add_modality(
        "HE", **{x: he_dict[x] for x in he_dict if not re.search("mask", x)}
    )

    reg_graph.add_modality(
        "IF", **{x: if_dict[x] for x in if_dict if not re.search("mask", x)}
    )

    reg_graph.add_reg_path("HE", "IF", thru_modality=None, reg_params=["nl"])

    reg_graph.add_merge_modalities("aligned", ["HE", "IF"])
    reg_graph.cache_images = False

    # Get the default parameter dictionary from the WSIReg graph.
    reg_params = _prepare_reg_models(
        reg_graph.reg_graph_edges[0]["params"]
    )  # Here we are loading one of the registration tasks within the graph.
    # reg_params[1]['MaximumStepLength'] = ['30','30','20','20','20','10','10','10','1','1'] #Custom parameter sweep for nonlinear

    # Custom parameter changes
    for pmap in reg_params:
        pmap["UseMultiThreadingForMetrics"] = ["false"]
        pmap["MaximumNumberOfIterations"] = ["200"]
        pmap["MaximumNumberOfSamplingAttempts"] = ["1000"]
        pmap["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
        pmap["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]

    with open(output_dir + "reg_params.json", "w") as f:
        json.dump(reg_params, f)

    return reg_params


# %% [markdown]
# # Test

# %%
rechunked_data = cimg.rechunk((1, 1, 1000, 1000))

# Make sure rechunking was successful
assert rechunked_data.chunksize == (1, 1, 1000, 1000)

dim_order = ["t", "c", "y", "x"]
im = rechunked_data
sigma_values = np.hstack(
    [np.zeros(im.ndim - 2), np.array((3, 3))]
)  # No filtering for the first three dimensions
# %%
# explicitly tell xarray to get every channel except '408'
loged = dask_ndi.gaussian_laplace(im.astype(np.float32), sigma=sigma_values)


def get_peaks(im_block):
    # Apply peak_local_max
    original_shape = im_block.shape
    squeezed_im_block = im_block.squeeze()
    squeezed_shape = squeezed_im_block.shape
    peaks = peak_local_max(squeezed_im_block, min_distance=5, threshold_abs=800)

    # Create a zeros array of the same shape as im_block
    result = np.zeros_like(im_block, dtype=np.bool_)

    # Determine how many dimensions were squeezed out
    diff_dims = len(original_shape) - len(squeezed_shape)

    # Extract the intensity of the peaks
    for peak in peaks:
        # Expanding the peak dimensions to match the im_block
        full_peak = tuple([0] * diff_dims) + tuple(peak)
        result[full_peak] = True

    return result


with ProgressBar():
    y = loged.map_blocks(get_peaks, dtype=np.bool_)  # .compute()
    # Drop the first channel to match indexing of loged

    # Find the coordinates where y is True
    y_summed_over_c = y.sum(axis=1)

    # If you want to force it to be a boolean array
    y_bool = y_summed_over_c > 0  # This will be True if a peak was found in any channel

    max_size = 9
    # Create a boolean footprint array with ones only in the last two dimensions
    footprint_shape = [1] * (im.ndim - 2) + [max_size, max_size]
    footprint = np.ones(footprint_shape, dtype=bool)

    # Apply the maximum filter using the footprint
    maxed = dask_ndi.maximum_filter(im, footprint=footprint)
    maxed = maxed.rechunk((1, -1, "auto", "auto"))

dfs = []

n_rounds = y.shape[0]
for t in tqdm(range(n_rounds), desc="Extracting intensities"):
    # Pre-compute the time slice for maxed and y_bool
    maxed_slice = maxed[t].compute()
    ys_slice = y_bool[t].compute()

    # Get the coordinates where ys is True
    coords = np.argwhere(ys_slice)

    # Initialize a list to store intensity values
    vals_list = []

    # Loop through each channel to get the intensity values
    n_channels = maxed.shape[1]
    for ch in range(n_channels):
        channel_values = maxed_slice[ch, ys_slice]
        vals_list.append(channel_values)

    # Stack the lists horizontally to get a 2D array of shape (n_points, n_channels)
    vals_array = np.column_stack(vals_list)

    # Create DataFrame for the current time slice
    df = pd.DataFrame(
        vals_array, columns=[f"Channel_{i}" for i in range(1, n_channels + 1)]
    )
    df["time"] = t
    df[["y", "x"]] = coords
    # Group points that are within a certain radius
    tree = cKDTree(df[["y", "x"]].values)
    # Define the radius
    group_radius = 5  # You can set this based on your specific requirements
    # Perform the query_pairs which will return pairs of points within the radius
    pairs = tree.query_pairs(group_radius)

    # Create an empty dictionary to store the group index for each point
    groups = {}
    # Loop through the pairs to assign group indices
    group_idx = 0
    for i, j in pairs:
        if i not in groups and j not in groups:
            groups[i] = groups[j] = group_idx
            group_idx += 1
        elif i in groups:
            groups[j] = groups[i]
        elif j in groups:
            groups[i] = groups[j]

    # For unpaired points, assign them unique group indices
    for i in df.index:
        if i not in groups:
            groups[i] = group_idx
            group_idx += 1

    # Assign the group indices back to the DataFrame
    df["group"] = df.index.map(groups).fillna(-1)

    # Group by the 'group' column and take the max over the channels
    result = (
        df.groupby("group")
        .agg(
            {
                "Channel_1": "max",
                "Channel_2": "max",
                "Channel_3": "max",
                "Channel_4": "max",
                "y": "mean",
                "x": "mean",
                "time": "mean",  # You can also take 'min' or 'max' depending on what you want
            }
        )
        .reset_index()
    )
    # remove the group column
    del result["group"]
    # Add to list of DataFrames
    dfs.append(result)
    # Assertion to ensure that the size is as expected

# Concatenate all the DataFrames into one
final_df = pd.concat(dfs, ignore_index=True)

# %%
# Load and preprocess initial DataFrame
initial_points = final_df.copy()
initial_points.rename(
    columns={
        "Channel_1": "545",
        "Channel_2": "594",
        "Channel_3": "647",
        "Channel_4": "680",
    },
    inplace=True,
)
initial_points["max_channel"] = initial_points.iloc[:, :4].idxmax(axis=1)

# Filter rows based on intensity threshold
thresholded_intensity = 1200
above_threshold = initial_points.iloc[:, :4] > thresholded_intensity
filtered_points = initial_points[above_threshold.any(axis=1)]

# Perform particle tracking
search_radius = 10
tracked_particles = tp.link(filtered_points, search_radius, t_column="time", memory=1)

# Keep only frequently occurring particles
frequent_particles = tracked_particles["particle"].value_counts()
frequent_particles = frequent_particles[frequent_particles >= 5].index
frequent_particle_tracks = tracked_particles[
    tracked_particles["particle"].isin(frequent_particles)
]

# Perform initial data visualization
sns.pairplot(
    frequent_particle_tracks.iloc[:3000, :4], plot_kws={"alpha": 0.3}
).fig.suptitle("Initial data")

# Compensate for channel crosstalk
high_intensity_threshold = 1500
high_intensity_tracks = frequent_particle_tracks[
    (frequent_particle_tracks.iloc[:, :4] > high_intensity_threshold).any(axis=1)
]
crosstalk_matrix = _channel_crosstalk_matrix(high_intensity_tracks.iloc[:, :4].values)
compensated_data = crosstalk_matrix.dot(frequent_particle_tracks.iloc[:, :4].values.T).T
compensated_particle_tracks = frequent_particle_tracks.copy()
compensated_particle_tracks.iloc[:, :4] = compensated_data

# Visualize compensated data
sns.pairplot(
    compensated_particle_tracks.sample(5000).iloc[:, :4], plot_kws={"alpha": 0.3}
).fig.suptitle("Compensated data")

# Update particle tracks based on compensation
compensated_particle_tracks["max_channel"] = compensated_particle_tracks.iloc[
    :, :4
].idxmax(axis=1)
compensated_particle_tracks["max_intensity"] = compensated_particle_tracks.apply(
    lambda row: row[row["max_channel"]], axis=1
)

# Create summary dataframes for max channel and intensity
summary_channel = (
    compensated_particle_tracks.groupby(["time", "particle"])["max_channel"]
    .first()
    .reset_index()
)
summary_intensity = (
    compensated_particle_tracks.groupby(["time", "particle"])["max_intensity"]
    .first()
    .reset_index()
)
pivot_channels = summary_channel.pivot(
    index="particle", columns="time", values="max_channel"
).reset_index()
pivot_intensity = summary_intensity.pivot(
    index="particle", columns="time", values="max_intensity"
).reset_index()

# Merge the summary dataframes
pivot_channels.columns = ["particle"] + [
    f"c_{col}" for col in pivot_channels.columns[1:]
]
pivot_intensity.columns = ["particle"] + [
    f"i_{col}" for col in pivot_intensity.columns[1:]
]
combined_summary = pd.concat(
    [pivot_channels, pivot_intensity.drop("particle", axis=1)], axis=1
)
combined_summary.to_csv("calls.csv", index=False)

# Generate barcodes
channel_map = {"545": "G", "594": "T", "647": "A", "680": "C", "0": "N"}
combined_summary["barcode"] = (
    combined_summary[["c_0", "c_1", "c_2", "c_3", "c_4", "c_5"]]
    .fillna("0")
    .apply(lambda x: "".join([channel_map[str(i)] for i in x]), axis=1)
)

# Compute Hamming distances
reference_table = pd.read_csv(
    "/gstore/scratch/u/lubecke/etk_1_stitching/gRNAs_mouseMinimal1.csv", index_col=0
)
reference_barcodes = reference_table.spacer.str[:6]
reference_barcode_arrays = np.array(
    [[ord(char) for char in barcode] for barcode in reference_barcodes]
)
observed_barcode_arrays = np.array(
    [[ord(char) for char in barcode] for barcode in combined_summary["barcode"]]
)

hamming_distances = cdist(
    observed_barcode_arrays, reference_barcode_arrays, metric="hamming"
)
min_hamming_distances = hamming_distances.min(axis=1) * len(reference_barcodes[0])
closest_matches = hamming_distances.argmin(axis=1)
matched_barcodes = reference_barcodes.values[closest_matches]

combined_summary["matched_barcode"] = matched_barcodes
combined_summary["hamming_distance"] = min_hamming_distances

# Get unique 'particle', 'x', and 'y' from the original tracked DataFrame
unique_particles_coords = frequent_particle_tracks[
    ["particle", "x", "y"]
].drop_duplicates(subset=["particle"])

# Merge back 'x' and 'y' into the final DataFrame based on 'particle'
combined_summary_with_coords = pd.merge(
    combined_summary, unique_particles_coords, on="particle", how="left"
)

print(
    f' Call Rate: {100*len(combined_summary_with_coords.query("hamming_distance < 2")) / len(combined_summary_with_coords)}'
)
passed = combined_summary_with_coords.query("hamming_distance < 2")
print(f"Passed: {len(passed)}")

# %%
print(
    f' Call Rate: {100*len(combined_summary_with_coords.query("hamming_distance < 2")) / len(combined_summary_with_coords)}'
)
passed = combined_summary_with_coords.query("hamming_distance < 2")
print(f"Passed: {len(passed)}")

# %%
sns.histplot(combined_summary_with_coords.hamming_distance)
plt.show()

sns.histplot(called_peaks.hamming_distance)
plt.show()

# %%
imarr = dapi_img

f = plt.figure(dpi=300)
plt.imshow(dask_ndi.gaussian_filter(imarr[0, 10000:11000, 3000:4000], 15))
plt.show()

# %%
imarr = dapi_img

p, _, _, _ = np.linalg.lstsq(
    dask_ndi.gaussian_filter(imarr[0, 10000:11000, 3000:4000], 15)
    .ravel()[:, np.newaxis]
    .compute(),
    base_img[0, -1, 10000:11000, 3000:4000].ravel().compute(),
)
z = base_img[0, -1, 10000:11000, 3000:4000] - p * dask_ndi.gaussian_filter(
    imarr[0, 10000:11000, 3000:4000], 15
)
z[z < 0] = 0

# %%
zn = dask_ndi.gaussian_laplace(
    base_img[0, -1, 10000:11000, 3000:4000].astype(np.float32), sigma=[5, 5]
)

# %%
rechunked_data = base_img.rechunk((1, 1, 1000, 1000))

# Make sure rechunking was successful
assert rechunked_data.chunksize == (1, 1, 1000, 1000)

dim_order = ["t", "c", "y", "x"]
im = rechunked_data
sigma_values = np.hstack(
    [np.zeros(im.ndim - 2), np.array((3, 3))]
)  # No filtering for the first three dimensions
# %%
# explicitly tell xarray to get every channel except '408'
loged = dask_ndi.gaussian_laplace(im.astype(np.float32), sigma=sigma_values)


# %%
f, a = plt.subplots(2, 2)
f.set_dpi(300)

a[0, 0].imshow(np.sqrt(base_img[0, -1, 10000:11000, 3000:4000]))
a[1, 0].imshow(
    dask_ndi.gaussian_laplace(base_img.astype(np.float32), sigma=[0, 0, 3, 3])[
        0, -1, 10000:11000, 3000:4000
    ]
)
a[0, 1].imshow(np.sqrt(cimg[0, -1, 10000:11000, 3000:4000]))
a[1, 1].imshow(
    dask_ndi.gaussian_laplace(cimg.astype(np.float32), sigma=[0, 0, 3, 3])[
        0, -1, 10000:11000, 3000:4000
    ]
)

a[0, 0].set_title("Uncorrected")
a[0, 1].set_title("DAPI-corrected")
a[0, 0].set_ylabel("Raw image")
a[1, 0].set_ylabel("LoG filter")
plt.show()

# %%
sigma = 3
min_distance = 15

sigma_values = [0 if i < base_img.ndim - 2 else sigma for i in range(base_img.ndim)]

# LoG filter
loged = dask_ndi.gaussian_laplace(base_img.astype(np.float32), sigma=sigma_values)
# with ProgressBar():
#    loged.to_zarr('/gstore/scratch/u/lubecke/temp_log.zarr',compute=True,overwrite=True)

# Morphological footprint for peak detection
size = 2 * min_distance + 1
selem = np.ones(
    (1,) * (loged.ndim - 2) + (size,) * 2, dtype=bool
)  # Assume 2 spatial dimensions

border_width = skim.feature.peak._get_excluded_border_width(
    loged[0, 0, :, :], min_distance, exclude_border=True
)

# Maximum morph. filter
image_max = dask_ndi.maximum_filter(loged, footprint=selem, mode="nearest")

# %%
immax = dask_ndi.maximum_filter(imcorr, footprint=selem[0, ...], mode="nearest")

# %%
testmax = dask_ndi.maximum_filter(
    imcorr[-1, :, :], footprint=selem[0, 0, ...], mode="nearest"
)

# %%
plt.imshow(immax[-1, :, :])
plt.show()

# %%
plt.imshow(testmax)
plt.show()

# %%
spots = -dask_ndi.gaussian_laplace(z.astype(np.float32), sigma=[5, 5])
spots[spots < 0] = 0

spots_max = dask_ndi.maximum_filter(spots, footprint=selem.squeeze(), mode="nearest")

# %%
plt.imshow(spots)
plt.show()

# %%
thresh = skim.filters.threshold_multiotsu(spots.compute())

# %%
y, x = np.where((spots == spots_max) & (spots > thresh[-1]))
y = y.compute()
x = x.compute()

# %%
f = plt.figure(dpi=300)
plt.imshow(spots, vmin=0)
plt.scatter(x, y, s=1, c="r", marker="x", alpha=0.3)
plt.show()

# %%
