# %%
from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
from skimage.morphology import dilation, disk


def make_nd_image(ref_images: list[Path], out_path: Path):
    ims = (da.from_zarr(im) for im in ref_images)
    im_stack = da.stack(ims, axis=0)
    im_stack.to_zarr(out_path)


def make_image(point_file: Path, out_path: Path, ref_images: list[Path]):
    im_shapes = [da.from_zarr(im).shape for im in ref_images]

    assert len(set(im_shapes)) == 1

    y, x = im_shapes[0][1:]

    points = gpd.read_file(point_file)

    # Convert matched_barcode to categorical and then to integer codes
    points["matched_barcode"] = points["matched_barcode"].astype("category")
    points["barcode_int"] = (
        points["matched_barcode"].cat.codes + 1
    )  # +1 to avoid using 0 as a code

    # Create an empty numpy array instead of a Dask array for the initial operation
    point_image = np.zeros((y, x), dtype=np.int32)

    # Convert geometry coordinates to integer indices
    px = points.geometry.x.astype(int)
    py = points.geometry.y.astype(int)

    # Use numpy advanced indexing to assign values
    point_image[px, py] = points["barcode_int"].values

    image_np = dilation(point_image, disk(5))
    # Convert the updated numpy array to a Dask array for further operations or saving

    image_da = da.from_array(image_np)
    image_da.to_zarr(out_path)


if __name__ == "__main__":
    import typer

    typer.run(make_image)


# point_file = Path(
#     "/gnet/is1/p01/shares/ctg-microscopy/EL/perturbview_output/eTK146A/intermediate_outputs/particles.geojson"
# )
# images = Path(
#     "/gnet/is1/p01/shares/ctg-microscopy/EL/perturbview_output/eTK146A/intermediate_outputs/"
# ).glob("*_registered.zarr")
