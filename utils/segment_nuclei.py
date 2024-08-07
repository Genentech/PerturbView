from pathlib import Path

import numpy as np
import typer
import xarray as xr
from aicsimageio import AICSImage, writers
from csbdeep.utils import normalize
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology, segmentation
from stardist.models import StarDist2D
from stardist.plot import render_label
from tqdm import tqdm

# # Import CuPy or fallback to scipy.ndimage if CuPy is not installed or if no GPU is available
# try:
#     import cupy as cp
#     import cupyx as cpx
#     from cucim.skimage import morphology, segmentation

#     cp.cuda.Device()  # Will raise an exception if no GPU is found
#     ndi = cpx.scipy.ndimage  # Use CuPy's ndimage equivalent functions where possible
#     print("Using CuPy for GPU acceleration")
# except (ImportError, cp.cuda.runtime.CUDARuntimeError, AttributeError):
#     from scipy import ndimage as ndi
#     from skimage import morphology, segmentation

#     print("Falling back to scipy.ndimage for CPU processing")


def segment_3d_binary_stack(binary_volume, height=5, min_size=50):
    # 1. Compute the distance transform
    distance = ndi.distance_transform_edt(binary_volume)

    # Apply Gaussian smoothing to the distance map to reduce noise
    smoothed_distance = ndi.gaussian_filter(
        distance, sigma=1
    )  # sigma may need to be adjusted

    # Use h-maxima transform to suppress shallow maxima
    h_maxima = morphology.h_maxima(smoothed_distance, height)

    # Label the local maxima
    markers, _ = ndi.label(h_maxima)

    # Apply the watershed algorithm
    labels = segmentation.watershed(-smoothed_distance, markers, mask=binary_volume)

    # Remove small objects
    cleaned_volume = morphology.remove_small_objects(
        labels, min_size=min_size
    )  # Increase min_size if necessary

    return cleaned_volume


def stardist_2d_stack(im: xr.DataArray, target_channel: int, zoom: float = 0.0):
    """
    Segment a 3D image stack using the StarDist2D model.

    Parameters:
    - im_root: Path to the directory containing the image files.
    - target_channel: Index of the target channel (DAPI) to be segmented.
    - zoom: Zoom factor for the image. Default is 0.0 (no zoom).

    Returns:
    - segmented_volume: A 3D numpy array of the segmented volume.
    """
    # search for nd2 files recursively

    # Initialize the model
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    # List to hold segmented slices
    segmented_slices = []

    # Iterate over each Z-slice and segment
    n_slices = im.sizes["Z"]
    for z in tqdm(range(n_slices), desc="Segmenting slices"):
        img = im.isel(C=target_channel, Z=z).values
        if zoom != 0.0:
            img = ndi.zoom(img, zoom)
        normalized_img = normalize(
            img, 1, 99.8
        )  # Adjust normalization percentiles as needed
        labels, _ = model.predict_instances(normalized_img)
        # invert the zoom
        if zoom != 0.0:
            labels = ndi.zoom(labels, 1 / zoom)
        segmented_slices.append(labels)

    # Convert list of 2D arrays into a 3D numpy array
    segmented_volume = np.stack(segmented_slices, axis=0)

    return segmented_volume


def save_segmentation(
    segmented_volume: np.ndarray,
    file_path: Path,
    pixel_sizes,
):
    """
    Save the segmented 3D volume as an 8-bit TIFF image using aicsimageio.

    Parameters:
    - segmented_volume: Numpy array of the segmented 3D volume.
    - file_path: Path (including filename) where the TIFF file will be saved.
    """
    # Ensure the segmented volume is in 16-bit format
    seg_vol = segmented_volume.astype(np.uint16)

    # Save the volume as a TIFF file using aicsimageio
    writers.ome_tiff_writer.OmeTiffWriter().save(
        seg_vol,
        file_path,
        dimension_order="ZYX",
        pixel_sizes=pixel_sizes,
    )


def plot_debug(im, segmented_volume, cleaned_volume, z=10):
    """
    Plot a 2x2 comparison of original image slices, initial segmentation, and watershed-enhanced segmentation.

    Parameters:
    - im: The original image volume.
    - segmented_volume: The initial segmented volume before watershed.
    - cleaned_volume: The volume after applying watershed segmentation.
    - z: The specific Z slice index to display.
    """
    # Original image slice for specific Z
    img_slice = im.isel(C=4, Z=z).values

    # Maximum intensity projection (MIP) across Z for the original image
    img_mip = im.isel(C=4).max(axis=0).values

    plt.figure(figsize=(12, 12))  # Adjust figure size for a 2x2 grid

    # Top left: Initial segmentation overlay on a specific Z slice
    plt.subplot(2, 2, 1)
    plt.imshow(render_label(segmented_volume[z], img=img_slice))
    plt.axis("off")
    plt.title("Initial Prediction + Input Overlay (Slice {})".format(z))

    # Top right: Watershed segmentation overlay on the same Z slice
    plt.subplot(2, 2, 2)
    plt.imshow(render_label(cleaned_volume[z], img=img_slice))
    plt.axis("off")
    plt.title("Watershed Prediction + Input Overlay (Slice {})".format(z))

    # Bottom left: Initial segmentation overlay on MIP
    plt.subplot(2, 2, 3)
    plt.imshow(render_label(segmented_volume.max(axis=0), img=img_mip))
    plt.axis("off")
    plt.title("Initial Prediction + Input Overlay (MIP)")

    # Bottom right: Watershed segmentation overlay on MIP
    plt.subplot(2, 2, 4)
    plt.imshow(render_label(cleaned_volume.max(axis=0), img=img_mip))
    plt.axis("off")
    plt.title("Watershed Prediction + Input Overlay (MIP)")

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def main(
    im_path: Path,
    target_channel: int,
    save_path: Path,
    dimensionality: str = "3D",
    z_slice: str = "auto",
    flip_x: bool = False,
    flip_y: bool = False,
    height: int = 5,
    min_size: int = 50,
    zoom: float = 0.0,
    debug: bool = False,
):
    """
    Segment nuclei in a 2D slice or a 3D image stack using StarDist2D followed by a watershed algorithm.

    Parameters:
    - im_path (Path): Path to the 4D image stack file (CZYX format).
    - target_channel (int): Index of the target channel (e.g., DAPI) to be segmented.
    - save_path (Path): Path (including filename) where the segmented TIFF file will be saved.
    - dimensionality (str): Specifies the dimensionality for segmentation ('2D' or '3D').
      '2D' processes a single Z-slice, while '3D' processes the entire stack.
    - z_slice (int, optional): Index of the specific Z slice to use for 2D segmentation or "auto" for the best slice. Required if `dimensionality` is '2D'.
    - flip_x (bool): If True, horizontally flip the image(s) before segmentation.
    - flip_y (bool): If True, vertically flip the image(s) before segmentation.
    - height (int): Height parameter for the h-maxima transform in the watershed algorithm.
    - min_size (int): Minimum size of objects to keep after applying the watershed algorithm.
    - zoom (float): Zoom factor for the image. Default is 0.0 (no zoom).
    - debug (bool): If True, displays a 2x2 comparison of the original image, initial segmentation,
      and watershed-enhanced segmentation for debugging purposes.

    Processes an image file for segmentation, applying optional flips, and either segments
    a specified 2D slice or the entire 3D volume based on the `dimensionality` parameter.
    The result is saved as an 8-bit TIFF image. Debug visualizations can be enabled for
    further analysis.

    Raises:
    - ValueError: If `dimensionality` is '2D' but `z_slice` is not specified.
    - ValueError: If an invalid `dimensionality` value is provided.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw_im = AICSImage(im_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        blank = (
            np.zeros([1, 2072, 2072], dtype=np.uint8)
            if dimensionality == "2D"
            else np.zeros([2072, 2072, 12], dtype=np.uint8)
        )
        save_segmentation(blank, save_path, None)
        return

    im = raw_im.get_xarray_dask_stack().squeeze()

    # Apply flips if specified
    if flip_x:
        im = im[:, :, ::-1]  # Flip horizontally
    if flip_y:
        im = im[:, ::-1, :]  # Flip vertically

    # Adjust for 2D or 3D processing
    if dimensionality == "2D":
        match z_slice:
            case "auto":
                segmented_volume = stardist_2d_stack(im, target_channel, zoom)
                # get the z slice with the most nuclei
                max_z = np.argmax(segmented_volume.sum(axis=(1, 2)))
                segmented_volume = np.expand_dims(segmented_volume[max_z, ...], 0)
            case _ if z_slice.isdigit():
                z_slice = int(z_slice)  # type: ignore
                # Select the specified Z-slice and expand the dimension to put it back
                im = im.isel(Z=z_slice).expand_dims("Z")
                segmented_volume = stardist_2d_stack(im, target_channel, zoom)
            case _:
                raise ValueError(
                    "For 2D segmentation, z_slice must be 'auto' or an integer value."
                )
    elif dimensionality == "3D":
        print("3d")
        segmented_volume = stardist_2d_stack(im, target_channel, zoom)
        # Apply watershed to the segmented volume only for 3D segmentation
        cleaned_volume = segment_3d_binary_stack(segmented_volume, height, min_size)
    else:
        raise ValueError("Invalid dimensionality. Choose either '2D' or '3D'.")

    # Debug visualization
    if debug:
        if dimensionality == "3D":
            plot_debug(im, segmented_volume, cleaned_volume, z=10)
        else:
            # Debug visualization adjustments for 2D might be needed
            print("Debug visualization for 2D is not implemented.")

    # Save the segmented volume as a TIFF file
    pixel_sizes = raw_im.physical_pixel_sizes
    save_segmentation(
        segmented_volume if dimensionality == "2D" else cleaned_volume,
        save_path,
        pixel_sizes,
    )


if __name__ == "__main__":
    typer.run(main)

# # %%

# from pathlib import Path

# # # Load the data
# im_root = "/gstore/home/lubecke/Desktop/Link to YC/DISCO-FISH/20X/"
# # search for nd2 files recursively
# nd2_files = list(Path(im_root).rglob("*.nd2"))
# # load the first one
# main(nd2_files[0], 4, Path("~", "segmented.tif").expanduser(), 5, 50, True)
