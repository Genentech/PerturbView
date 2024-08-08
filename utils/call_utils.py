import gc
import json
from pathlib import Path

import dask.array as da
import geopandas
import numpy as np
import pandas as pd
import shapely
import SimpleITK as sitk
import skimage as skim
import xarray as xr
import zarr
from dask_image import ndfilters as dask_ndi
from dask_image.ndmeasure import label
from dask_image.ndmorph import binary_dilation
from skimage.feature import peak
from spotiflow.model import Spotiflow
from tqdm.auto import tqdm
from wsireg.reg_images.loader import reg_image_loader
from wsireg.reg_transforms import RegTransform, RegTransformSeq
from wsireg.utils.reg_utils import sitk_pmap_to_dict


# Takes an image (with it's autofluorescence/DAPI channel) and attempts to
# correct for the bleedthrough from that channel into the cyclic probe channels
# with a linear model (img_meas[c,...] = corr_factor[c] * img[af_channel,..] + img[c,...])
# Image is a CxHxW array
def get_af_factors(img, af_channel=0):
    corr_factor = np.zeros(img.shape[0])
    af_signal = img[af_channel, ...].ravel()[:, np.newaxis]
    for c in range(img.shape[0]):
        if c == af_channel:
            corr_factor[c] = 0
            continue
        csignal = img[c, ...].ravel()
        p, _, _, _ = np.linalg.lstsq(
            af_signal[::100].compute(), csignal[::100].compute(), rcond=None
        )  # No need to use all the data
        corr_factor[c] = p

    return corr_factor


# Apply correction factors to an image.
def correct_af(img, af_coeffs, af_channel=0, temp_dir="/tmp"):
    cstore = zarr.TempStore(
        suffix="_dapicorr",
        dir=temp_dir,
    )
    cimg = zarr.create(
        store=cstore, shape=img.shape, dtype=np.float32, chunks=(1, 2048, 2048)
    )

    for c in range(img.shape[0]):
        cimg[c, ...] = (
            (img[c, ...] - af_coeffs[c] * img[af_channel, ...])
            .astype(np.float32)
            .compute()
        )

    img = da.from_zarr(cimg)
    return img * (
        img > 0
    )  # We threshold the resulting image so that anything < AF is 0.


def detect_peaks_spotiflow(
    img: da.Array, thresh: float = 0.5, temp_dir: Path | str = "/tmp"
) -> da.Array:
    """
    Use Spotiflow to detect peaks in an image.

    Args:
        img (dask.array): Image to detect peaks in.
        thresh (float, optional): Threshold for peak detection. Defaults to 0.5.
        temp_dir (Path | str, optional): Temporary directory to store intermediate results. Defaults to "/tmp".
    """
    fpath = Path(__file__).resolve()
    model = Spotiflow.from_folder(
        str(fpath.parents[1].joinpath("models", "general"))
    )
    peak_zarr = zarr.TempStore(
        suffix="_peakmask",
        dir=temp_dir,
    )
    peaks = zarr.create(
        store=peak_zarr, shape=img.shape, dtype=bool, chunks=(1, 2048, 2048)
    )
    for c in tqdm(range(img.shape[0])):
        points, details = model.predict(
            img[c].compute(),
            prob_thresh=thresh,
            subpix=False,
            peak_mode="fast",
            min_distance=20,
            n_tiles=(10, 10),
        )
        y, x = points[:, 0].astype(int), points[:, 1].astype(int)
        peaks[c, :, :] = details.heatmap > thresh
        peaks[c, y, x] = True
    return da.from_zarr(peaks)


# Detects peaks in an image
def detect_peaks(img, sigma=3, min_distance=15, threshold=None, temp_dir="/tmp"):
    img = img.astype(np.float32)

    # Morphological footprint for peak detection
    size = 2 * min_distance + 1
    selem = np.ones((size,) * 2, dtype=bool)  # Assume 2 spatial dimensions

    border_width = peak._get_excluded_border_width(
        img[0, :, :], min_distance, exclude_border=True
    )

    # Calculate peak intensity thresholds if we need to.
    if threshold is None:
        threshold = np.zeros(img.shape[0])
        for c in range(img.shape[0]):
            cthresh = skim.filters.threshold_multiotsu(
                img[c, ::min_distance, ::min_distance].compute(), classes=4
            )  # Downsample for speed
            threshold[c] = cthresh[1]
    else:
        threshold = np.ones(img.shape[0]) * threshold

    # Temp zarr store for peak mask. This is a very large image (even as a bool), so we stream to disk.
    peak_zarr = zarr.TempStore(
        suffix="_peakmask",
        dir=temp_dir,
    )
    peaks = zarr.create(
        store=peak_zarr, shape=img.shape, dtype=bool, chunks=(1, 2048, 2048)
    )

    # LoG-max filter - just do it in 2D.
    # Due to channel crosstalk and non-uniform background intensity, I am restricting this to 2D
    # For this reason, we have to look at the negative d^2/dx^2 to highlight local maxima.
    # There is some weirdness where dask takes >30mins to generate the computation graph if I don't compute...probably a lot of edge pruning
    for c in range(img.shape[0]):
        cloged = -dask_ndi.gaussian_laplace(
            img[c, :, :].astype(np.float32), sigma=sigma
        )
        logmax = dask_ndi.maximum_filter(
            cloged,
            footprint=selem,
            mode="nearest",
        )

        peaks[c, ...] = (cloged == logmax).compute()
    gc.collect()

    # Get peaks (see skimage docs)
    peak_arr = da.from_zarr(peaks)
    for c in range(img.shape[0]):
        peaks[c, ...] = (peak_arr[c, ...] & (img[c, ...] > threshold[c])).compute()
    gc.collect()

    # Hardcode 2D spatial border exclusion region
    peak_arr[..., slice(None, border_width[0]), :] = 0
    peak_arr[..., slice(-border_width[0], None), :] = 0

    peak_arr[..., slice(None, border_width[1])] = 0
    peak_arr[..., slice(-border_width[1], None)] = 0

    return peak_arr


# Function that groups peaks across channels
def label_peaks(peak_mask, grouping_radius=30, temp_dir="/tmp"):
    # The structuring element is a flat disk
    selem = skim.morphology.disk(radius=grouping_radius)

    # For each peak, we dilate a disk around it; overlapping disks are then merged.
    lstore = zarr.TempStore(
        suffix="_label",
        dir=temp_dir,
    )

    # Dilate peaks
    peak_dilated = binary_dilation(
        peak_mask.max(axis=0),
        selem,
    )

    # Label peaks
    peak_labels, n_peaks = label(
        peak_dilated,
    )

    larr = peak_labels.to_zarr(lstore, return_stored=True)

    return larr


# Get the peak properties from the peak label image and the intensity image
def peak_props(
    peak_labels,
    img,
    ctransform=None,
    align_to_xenium=True,
):
    # Get intensities from original image.
    int_img = xr.DataArray(
        img * (peak_labels > 0),
        coords={
            "C": np.arange(0, img.shape[0]),
            "Y": np.arange(0, peak_labels.shape[0]),
            "X": np.arange(0, peak_labels.shape[1]),
        },
    )

    if ctransform is not None:
        r_img = reg_image_loader(
            peak_labels,
            0.65,
        )
        r_img.preprocessing.as_uint8 = False
        r_img.preprocessing.max_int_proj = False
        r_img.read_reg_image()

        ctransform.resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        limg = ctransform.resampler.Execute(r_img.reg_image)
        limg = sitk.GetArrayFromImage(limg)
    else:
        limg = peak_labels.compute()

    idx = np.argsort(limg.ravel())
    szt = limg.ravel()[idx]

    rois, roi_idx = np.unique(szt, return_index=True)
    x, y = np.unravel_index(
        idx[roi_idx[1] :], shape=limg.shape
    )  # Points in Xenium pixels

    if ctransform is not None and align_to_xenium:
        xen_to_ISS = ctransform.composite_transform
        tpts = np.array(
            [
                xen_to_ISS.TransformPoint(
                    (yy.astype(np.float64) * 0.2125, xx.astype(np.float64) * 0.2125)
                )
                for xx, yy in zip(x, y)
            ]
        )
        tpts = tpts / 0.65  # Points in ISS pixels
    else:
        tpts = np.array([y, x]).T

    p_int = int_img.interp(
        Y=("pts", tpts[:, 1]), X=("pts", tpts[:, 0]), method="linear"
    )
    pdat = p_int.data.compute()

    peaks = pd.DataFrame(
        columns=[
            "x",
            "y",
            "area",
        ]
        + [f"ch_{i}" for i in range(img.shape[0])],
        index=rois[1:],
    )

    for ix, roi in tqdm(enumerate(rois[:-1]), total=len(rois) - 1):
        if ix == 0:
            continue

        # Channel intensities
        cint = pdat[
            :, (roi_idx[ix] - roi_idx[1]) : (roi_idx[ix + 1] - roi_idx[1])
        ]  # ch_intensities[ix-1,:]

        # (Weighted) centroid coordinates. We calculate the intensity at each point, which is directly mappable back to coordinates (x,y) in Xenium.
        p_int = cint.max(axis=0)
        x_cent = (
            p_int * x[(roi_idx[ix] - roi_idx[1]) : (roi_idx[ix + 1] - roi_idx[1])]
        ).sum() / p_int.sum()
        y_cent = (
            p_int * y[(roi_idx[ix] - roi_idx[1]) : (roi_idx[ix + 1] - roi_idx[1])]
        ).sum() / p_int.sum()

        peaks.loc[roi, ["x", "y"]] = [x_cent, y_cent]
        peaks.loc[roi, "area"] = cint.shape[1]
        peaks.loc[roi, [f"ch_{i}" for i in range(img.shape[0])]] = cint.mean(axis=1)

    cint = pdat[:, (roi_idx[-1] - roi_idx[1]) :]

    p_int = cint.max(axis=0)
    x_cent = (p_int * x[(roi_idx[-1] - roi_idx[1]) :]).sum() / p_int.sum()
    y_cent = (p_int * y[(roi_idx[-1] - roi_idx[1]) :]).sum() / p_int.sum()

    peaks.loc[rois[-1], ["x", "y"]] = [x_cent, y_cent]
    peaks.loc[rois[-1], [f"ch_{i}" for i in range(img.shape[0])]] = cint.mean(axis=1)
    peaks.loc[rois[-1], "area"] = cint.shape[1]

    # Convert to GeoPandas for transform manipulation/integration across time.
    peak_geometries = geopandas.GeoSeries(
        data=[shapely.Point([x.x, x.y]) for _, x in peaks.iterrows()], index=peaks.index
    )
    peak_gd = geopandas.GeoDataFrame(
        peaks.loc[
            :,
            [
                "area",
            ]
            + [f"ch_{i}" for i in range(img.shape[0])],
        ],
        geometry=peak_geometries,
    )

    return peak_gd


# Load transforms
def load_transforms(tpath):
    with open(tpath, "r") as f:
        # with open('/gstore/scratch/u/fooc5/H2023_307/S0_transforms_v3.json','r') as f:
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


# Takes a list of registration transforms and applies them to shapely polygons.
# Dfield res is the resolution (in microns) of the inverted displacement field. Increase this to create a coarser inverse that uses less memory
# You should choose an appropriate resolution for the field based on the quality of alignment/desired precision of ROI annotations
def transform_points(
    geometries,
    rtforms,
    dfield_res=1.0,
    geom_scale=1.0,
):
    gcoords = []
    for geom in geometries:
        coords = geom.coords
        coords = np.array(
            [np.array(coords.xy[0]) * geom_scale, np.array(coords.xy[1]) * geom_scale]
        ).T
        gcoords.append(coords)

    gcoords = np.concatenate(gcoords, axis=0)

    # Create inverse displacement fields for non-linear diffeomorphisms.
    tform_list = rtforms.reg_transforms
    output_res = tform_list[-1].output_spacing[
        0
    ]  # Assume x and y spacing are the same.

    for tform in tform_list:
        if tform.inverse_transform:
            continue

        size_ratio = dfield_res / tform.output_spacing[0]

        dfield = sitk.TransformToDisplacementFieldFilter()

        dfield.SetOutputSpacing([dfield_res, dfield_res])
        dfield.SetOutputOrigin(tform.output_origin)
        dfield.SetOutputDirection(tform.output_direction)
        dfield.SetSize([int(x // size_ratio) for x in tform.output_size])

        df = dfield.Execute(tform.itk_transform)
        df = sitk.InvertDisplacementField(df)
        df = sitk.DisplacementFieldTransform(df)

        tform.inverse_transform = df

    # Apply transforms to ROI coordinates
    transformed_geometries = []
    for ix, cc in enumerate(gcoords):
        tc = tform_list[0].inverse_transform.TransformPoint(cc)
        for tform in tform_list[1:]:
            tc = tform.inverse_transform.TransformPoint(tc)

        transformed_geometries.append(
            shapely.Point(tc[0] / output_res, tc[1] / output_res)
        )

    return geopandas.GeoSeries(data=transformed_geometries), output_res


# # %%
# img = da.from_zarr(
#     "/gnet/is1/p01/shares/ctg-microscopy/EL/perturbview_output/eTK146A/intermediate_outputs/20240225_173513_765_registered.zarr"
# )
# peaks = detect_peaks_spotiflow(img, thresh=0.4)
# # %%
# peak_zarr = Path(
#     "/gnet/is1/p01/shares/ctg-microscopy/EL/perturbview_output/eTK146A/test_spotiflow_peaks_0p4.zarr"
# )
# da.from_zarr(peaks).to_zarr(peak_zarr, overwrite=True)
# # %%
# arr = da.from_zarr(peak_zarr)
# plabel = label_peaks(arr, grouping_radius=2)
# lpeak_zarr = Path(
#     "/gnet/is1/p01/shares/ctg-microscopy/EL/perturbview_output/eTK146A/test_spotiflow_peak_labels_erode1.zarr"
# )
# plabel.to_zarr(lpeak_zarr, overwrite=True)
