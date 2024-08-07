# %%
import shutil
from pathlib import Path

import dask.array as da
import SimpleITK as sitk
import typer
import zarr
from aicsimageio import AICSImage
from wsireg.reg_images.loader import reg_image_loader

from utils.call_utils import load_transforms


def register_dask_im(
    zarr_file: Path,
    transform_json: Path,
    output_zarr_path: Path,
    pixel_size: float = 0.65,
) -> da.Array:
    # cut the name at _stitched and add _transforms.json
    tform = load_transforms(transform_json)
    if str(zarr_file).endswith(".zarr") and not str(zarr_file).endswith(".ome.zarr"):
        img = da.from_zarr(zarr_file)
    else:
        img = AICSImage(zarr_file).get_image_dask_data("CZYX")

    # Ensure image is either 3D or 4D, raise an error otherwise
    if img.ndim < 3 or img.ndim > 4:
        raise ValueError("Only 3D (CYX) and 4D (CZYX) images are supported.")

    tform.resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    if output_zarr_path.exists():
        shutil.rmtree(output_zarr_path)

    # Placeholder for the shape of the transformed image, to initialize Zarr store
    transformed_shape = None

    # Process each channel and Z slice if applicable, writing results directly to disk
    for z_index in range(img.shape[1]) if img.ndim == 4 else [0]:
        for channel_index in range(img.shape[0]):
            channel_img = (
                img[channel_index, z_index].compute()
                if img.ndim == 4
                else img[channel_index].compute()
            )

            # Prepare the image for registration
            r_img = reg_image_loader(channel_img, pixel_size)
            r_img.preprocessing.as_uint8 = False
            r_img.preprocessing.max_int_proj = False
            r_img.read_reg_image()

            # Apply the transformation
            transformed_img_sitk = tform.resampler.Execute(r_img.reg_image)
            transformed_img_np = sitk.GetArrayFromImage(transformed_img_sitk)

            # Initialize the Zarr store with the shape of the first transformed image
            if transformed_shape is None:
                transformed_shape = (
                    (img.shape[0],)
                    + (img.shape[1] if img.ndim == 4 else 1,)
                    + transformed_img_np.shape
                )
                _ = zarr.zeros(
                    transformed_shape,
                    chunks=True,
                    dtype=transformed_img_np.dtype,
                    store=str(output_zarr_path),
                )

            # Write directly to the Zarr store on disk
            zarr_array = zarr.open(str(output_zarr_path), mode="a")
            if img.ndim == 4:
                zarr_array[channel_index, z_index] = transformed_img_np
            else:
                zarr_array[channel_index] = transformed_img_np
    # TODO: rechunk the Zarr store if needed
    return zarr_array


def run(
    input_zarr: Path, out_zarr: Path, transform_json: Path, pixel_size: float = 0.65
):
    out_path = Path(out_zarr)
    # Save the transformed multi-channel image as Zarr
    out_path.mkdir(parents=True, exist_ok=True)
    register_dask_im(input_zarr, transform_json, out_path, pixel_size)


if __name__ == "__main__":
    typer.run(run)
