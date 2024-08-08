# %%
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from aicsimageio import AICSImage, exceptions, types
from aicsimageio.dimensions import DEFAULT_DIMENSION_ORDER, DimensionNames
from aicsimageio.metadata import utils
from aicsimageio.utils import io_utils
from dask.diagnostics import ProgressBar
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
from zarr.storage import default_compressor


# Copied from: https://allencellmodeling.github.io/aicsimageio/aicsimageio.writers.html.  Minor tweaks for compatibility.
class OmeZarrWriter:
    def __init__(self, uri: types.PathLike):
        """
        Constructor.

        Parameters
        ----------
        uri: types.PathLike
            The URI or local path for where to save the data.
        """
        # Resolve final destination
        fs, path = io_utils.pathlike_to_fs(uri)

        # Save image to zarr store!
        self.store = parse_url(uri, mode="w").store
        self.root_group = zarr.group(store=self.store)

    @staticmethod
    def build_ome(
        size_z: int,
        image_name: str,
        channel_names: List[str],
        channel_colors: List[int],
        channel_minmax: List[Tuple[float, float]],
    ) -> Dict:
        """
        Create the omero metadata for an OME zarr image

        Parameters
        ----------
        size_z:
            Number of z planes
        image_name:
            The name of the image
        channel_names:
            The names for each channel
        channel_colors:
            List of all channel colors
        channel_minmax:
            List of all (min, max) pairs of channel intensities

        Returns
        -------
        Dict
            An "omero" metadata object suitable for writing to ome-zarr
        """
        ch = []
        for i in range(len(channel_names)):
            ch.append(
                {
                    "active": True,
                    "coefficient": 1,
                    "color": f"{channel_colors[i]:06x}",
                    "family": "linear",
                    "inverted": False,
                    "label": channel_names[i],
                    "window": {
                        "end": float(channel_minmax[i][1]),
                        "max": float(channel_minmax[i][1]),
                        "min": float(channel_minmax[i][0]),
                        "start": float(channel_minmax[i][0]),
                    },
                }
            )

        omero = {
            "id": 1,  # ID in OMERO
            "name": image_name,  # Name as shown in the UI
            "version": "0.4",  # Current version
            "channels": ch,
            "rdefs": {
                "defaultT": 0,  # First timepoint to show the user
                "defaultZ": size_z // 2,  # First Z section to show the user
                "model": "color",  # "color" or "greyscale"
            },
            # TODO: can we add more metadata here?
            # # from here down this is all extra and not part of the ome-zarr spec
            # "meta": {
            #     "projectDescription": "20+ lines of gene edited cells etc",
            #     "datasetName": "aics_hipsc_v2020.1",
            #     "projectId": 2,
            #     "imageDescription": "foo bar",
            #     "imageTimestamp": 1277977808.0,
            #     "imageId": 12,
            #     "imageAuthor": "danielt",
            #     "imageName": "AICS-12_143.ome.tif",
            #     "datasetDescription": "variance dataset after QC",
            #     "projectName": "aics cell variance project",
            #     "datasetId": 3
            # },
        }
        return omero

    @staticmethod
    def _build_chunk_dims(
        chunk_dim_map: Dict[str, int],
        dimension_order: str = DEFAULT_DIMENSION_ORDER,
    ) -> Tuple[int, ...]:
        return tuple(chunk_dim_map[d] for d in dimension_order)

    def write_image(
        self,
        # TODO how to pass in precomputed multiscales?
        image_data: types.ArrayLike,  # must be 3D, 4D or 5D
        image_name: str,
        physical_pixel_sizes: Optional[types.PhysicalPixelSizes],
        channel_names: Optional[List[str]],
        channel_colors: Optional[List[int]],
        chunk_dims: Optional[Tuple] = None,
        scale_num_levels: int = 1,
        scale_factor: float = 2.0,
        dimension_order: Optional[str] = None,
        channel_minmax: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """
        Write a data array to a file.
        NOTE that this API is not yet finalized and will change in the future.

        Parameters
        ----------
        image_data: types.ArrayLike
            The array of data to store. Data arrays must have 2 to 6 dimensions. If a
            list is provided, then it is understood to be multiple images written to the
            ome-tiff file. All following metadata parameters will be expanded to the
            length of this list.
        image_name: str
            string representing the name of the image
        physical_pixel_sizes: Optional[types.PhysicalPixelSizes]
            PhysicalPixelSizes object representing the physical pixel sizes in Z, Y, X
            in microns.
            Default: None
        channel_names: Optional[List[str]]
            Lists of strings representing the names of the data channels
            Default: None
            If None is given, the list will be generated as a 0-indexed list of strings
            of the form "Channel:image_index:channel_index"
        channel_colors: Optional[List[int]]
            List of rgb color values per channel or a list of lists for each image.
            These must be values compatible with the OME spec.
            Default: None
        scale_num_levels: Optional[int]
            Number of pyramid levels to use for the image.
            Default: 1 (represents no downsampled levels)
        scale_factor: Optional[float]
            The scale factor to use for the image. Only active if scale_num_levels > 1.
            Default: 2.0
        dimension_order: Optional[str]
            The dimension order of the data. If None is given, the dimension order will
            be guessed from the number of dimensions in the data according to TCZYX
            order.

        Examples
        --------
        Write a TCZYX data set to OME-Zarr

        >>> image = numpy.ndarray([1, 10, 3, 1024, 2048])
        ... writer = OmeZarrWriter("/path/to/file.ome.zarr")
        ... writer.write_image(image)

        Write multi-scene data to OME-Zarr, specifying channel names

        >>> image0 = numpy.ndarray([3, 10, 1024, 2048])
        ... image1 = numpy.ndarray([3, 10, 512, 512])
        ... writer = OmeZarrWriter("/path/to/file.ome.zarr")
        ... writer.write_image(image0, "Image:0", ["C00","C01","C02"])
        ... writer.write_image(image1, "Image:1", ["C10","C11","C12"])
        """
        ndims = len(image_data.shape)
        if ndims < 2 or ndims > 5:
            raise exceptions.InvalidDimensionOrderingError(
                f"Image data must have 2, 3, 4, or 5 dimensions. "
                f"Received image data with shape: {image_data.shape}"
            )
        if dimension_order is None:
            dimension_order = DEFAULT_DIMENSION_ORDER[-ndims:]
        if len(dimension_order) != ndims:
            raise exceptions.InvalidDimensionOrderingError(
                f"Dimension order {dimension_order} does not match data "
                f"shape: {image_data.shape}"
            )
        if (len(set(dimension_order) - set(DEFAULT_DIMENSION_ORDER)) > 0) or len(
            dimension_order
        ) != len(set(dimension_order)):
            raise exceptions.InvalidDimensionOrderingError(
                f"Dimension order {dimension_order} is invalid or contains"
                f"unexpected dimensions. Only {DEFAULT_DIMENSION_ORDER}"
                f"currently supported."
            )
        xdimindex = dimension_order.find(DimensionNames.SpatialX)
        ydimindex = dimension_order.find(DimensionNames.SpatialY)
        zdimindex = dimension_order.find(DimensionNames.SpatialZ)
        cdimindex = dimension_order.find(DimensionNames.Channel)
        if cdimindex > min(i for i in [xdimindex, ydimindex, zdimindex] if i > -1):
            raise exceptions.InvalidDimensionOrderingError(
                f"Dimension order {dimension_order} is invalid. Channel dimension "
                f"must be before X, Y, and Z."
            )

        if chunk_dims is not None and len(chunk_dims) != ndims:
            raise exceptions.UnexpectedShapeError(
                f"Chunk dimensions:{chunk_dims} do not match data. "
                f"Expected chunk dimension length:{ndims}"
            )

        if physical_pixel_sizes is None:
            pixelsizes = (1.0, 1.0, 1.0)
        else:
            pixelsizes = (
                physical_pixel_sizes.Z if physical_pixel_sizes.Z is not None else 1.0,
                physical_pixel_sizes.Y if physical_pixel_sizes.Y is not None else 1.0,
                physical_pixel_sizes.X if physical_pixel_sizes.X is not None else 1.0,
            )
        if channel_names is None:
            # TODO this isn't generating a very pretty looking name but it will be
            # unique
            channel_names = (
                [
                    utils.generate_ome_channel_id(image_id=image_name, channel_id=i)
                    for i in range(image_data.shape[cdimindex])
                ]
                if cdimindex > -1
                else [utils.generate_ome_channel_id(image_id=image_name, channel_id=0)]
            )
        if channel_colors is None:
            # TODO generate proper colors or confirm that the underlying lib can handle
            # None
            channel_colors = (
                [i for i in range(image_data.shape[cdimindex])]
                if cdimindex > -1
                else [0]
            )
        # Chunk spatial dimensions
        scale_dim_map = {
            DimensionNames.Time: 1.0,
            DimensionNames.Channel: 1.0,
            DimensionNames.SpatialZ: pixelsizes[0],
            DimensionNames.SpatialY: pixelsizes[1],
            DimensionNames.SpatialX: pixelsizes[2],
        }
        transforms = [
            [
                # the voxel size for the first scale level
                {
                    "type": "scale",
                    "scale": [scale_dim_map[d] for d in dimension_order],
                }
            ]
        ]
        # TODO precompute sizes for downsampled also.
        plane_size = (
            image_data.shape[xdimindex]
            * image_data.shape[ydimindex]
            * image_data.itemsize
        )

        target_chunk_size = 16 * (1024 * 1024)  # 16 MB
        # this is making an assumption of chunking whole XY planes.

        if chunk_dims is None:
            nplanes_per_chunk = int(math.ceil(target_chunk_size / plane_size))
            nplanes_per_chunk = (
                min(nplanes_per_chunk, image_data.shape[zdimindex])
                if zdimindex > -1
                else 1
            )
            chunk_dim_map = {
                DimensionNames.Time: 1,
                DimensionNames.Channel: 1,
                DimensionNames.SpatialZ: nplanes_per_chunk,
                DimensionNames.SpatialY: image_data.shape[ydimindex],
                DimensionNames.SpatialX: image_data.shape[xdimindex],
            }
            chunks = [
                dict(
                    chunks=OmeZarrWriter._build_chunk_dims(
                        chunk_dim_map=chunk_dim_map, dimension_order=dimension_order
                    ),
                    compressor=default_compressor,
                )
            ]
        else:
            chunks = [
                dict(
                    chunks=chunk_dims,
                    compressor=default_compressor,
                )
            ]

        lasty = image_data.shape[ydimindex]
        lastx = image_data.shape[xdimindex]
        # TODO scaler might want to use different method for segmentations than raw
        # TODO allow custom scaler or pre-computed multiresolution levels
        if scale_num_levels > 1:
            # TODO As of this writing, this Scaler is not the most general
            # implementation (it does things by xy plane) but it's code already
            # written that also works with dask, so it's a good starting point.
            scaler = Scaler()
            scaler.method = "nearest"
            scaler.max_layer = scale_num_levels - 1
            scaler.downscale = scale_factor if scale_factor is not None else 2
            for _ in range(scale_num_levels - 1):
                scale_dim_map[DimensionNames.SpatialY] *= scaler.downscale
                scale_dim_map[DimensionNames.SpatialX] *= scaler.downscale
                transforms.append(
                    [
                        {
                            "type": "scale",
                            "scale": [scale_dim_map[d] for d in dimension_order],
                        }
                    ]
                )

                if chunk_dims is None:
                    lasty = int(math.ceil(lasty / scaler.downscale))
                    lastx = int(math.ceil(lastx / scaler.downscale))
                    chunk_dim_map = {
                        DimensionNames.Time: 1,
                        DimensionNames.Channel: 1,
                    }
                    plane_size = lasty * lastx * image_data.itemsize
                    nplanes_per_chunk = int(math.ceil(target_chunk_size / plane_size))
                    nplanes_per_chunk = (
                        min(nplanes_per_chunk, image_data.shape[zdimindex])
                        if zdimindex > -1
                        else 1
                    )

                    chunk_dim_map[DimensionNames.SpatialZ] = nplanes_per_chunk
                    chunk_dim_map[DimensionNames.SpatialY] = lasty
                    chunk_dim_map[DimensionNames.SpatialX] = lastx

                    chunks.append(
                        dict(
                            chunks=OmeZarrWriter._build_chunk_dims(
                                chunk_dim_map=chunk_dim_map,
                                dimension_order=dimension_order,
                            ),
                            compressor=default_compressor,
                        )
                    )
                else:
                    rescaley = int(math.ceil(chunk_dims[ydimindex] / scaler.downscale))
                    rescalex = int(math.ceil(chunk_dims[xdimindex] / scaler.downscale))
                    chunk_dims = tuple(list(chunk_dims[:-2]) + [rescaley, rescalex])

                    chunks.append(
                        dict(
                            chunks=chunk_dims,
                            compressor=default_compressor,
                        )
                    )

        else:
            scaler = None

        # try to construct per-image metadata
        if channel_minmax is None:
            channel_minmax = [
                (0.0, 1.0)
                for i in range(image_data.shape[cdimindex] if cdimindex > -1 else 1)
            ]
        ome_json = OmeZarrWriter.build_ome(
            image_data.shape[zdimindex] if zdimindex > -1 else 1,
            image_name,
            channel_names=channel_names,  # type: ignore
            channel_colors=channel_colors,  # type: ignore
            # This can be slow if computed here.
            channel_minmax=channel_minmax,
        )
        # TODO user supplies units?
        dim_to_axis = {
            DimensionNames.Time: {"name": "t", "type": "time", "unit": "millisecond"},
            DimensionNames.Channel: {"name": "c", "type": "channel"},
            DimensionNames.SpatialZ: {
                "name": "z",
                "type": "space",
                "unit": "micrometer",
            },
            DimensionNames.SpatialY: {
                "name": "y",
                "type": "space",
                "unit": "micrometer",
            },
            DimensionNames.SpatialX: {
                "name": "x",
                "type": "space",
                "unit": "micrometer",
            },
        }

        axes = [dim_to_axis[d] for d in dimension_order]

        # TODO image name must be unique within this root group
        group = self.root_group  # .create_group(image_name, overwrite=True)
        group.attrs["omero"] = ome_json

        write_image(
            image=image_data,
            group=group,
            scaler=scaler,
            axes=axes,
            # For each resolution, we have a List of transformation Dicts (not
            # validated). Each list of dicts are added to each datasets in order.
            coordinate_transformations=transforms,
            # Options to be passed on to the storage backend. A list would need to
            # match the number of datasets in a multiresolution pyramid. One can
            # provide different chunk size for each level of a pyramid using this
            # option.
            storage_options=chunks,
        )


def hex_to_int(hex_color: str):
    """Convert hexadecimal color strings to integer values."""
    return int(hex_color.lstrip("#"), 16)


def im_to_ome_zarr(
    img_path: Path,
    output_zarr_path: Path,
    scale_factor: float = 3.0,
    min_tile_size: int = 1024,
    overwrite=False,
    min_max_quantiles: Optional[Tuple[float, float]] = None,
    reverse_channels: bool = False,
):
    """
    Convert an image to OME-Zarr format.

    Parameters
    ----------
    img_path: Path
        Path to the image file.
    output_zarr_path: Path
        Path to the output OME-Zarr file.
    scale_factor: float
        Scale factor for the pyramid levels.
    min_tile_size: int
        Minimum tile size for the pyramid levels.
    overwrite: bool
        Overwrite the output path if it already exists.
    min_max_quantiles: Tuple[float, float]
        Quantiles for min-max normalization.
    """
    p = Path(output_zarr_path)
    if p.exists() and p.is_dir() and overwrite:
        shutil.rmtree(p)
    elif p.exists() and p.is_dir() and not overwrite:
        raise Exception(
            f"Output path {output_zarr_path} already exists. Set overwrite=True to remove it."
        )

    img = AICSImage(img_path)

    if img.dims["C"][0] == 5:
        # Define channel colors
        channel_colors_hex = [
            "#00FFFF",  # Cyan
            "#FF00FF",  # Magenta
            "#FF0000",  # Red
            "#00FF00",  # Green
            "#808080",  # Grey
        ]
    elif img.dims["C"][0] == 3:
        channel_colors_hex = [
            "#00FFFF",  # Cyan
            "#FF00FF",  # Magenta
            "#808080",  # Grey
        ]
    else:
        raise ValueError("Unsupported number of channels.")

    channel_colors = [hex_to_int(color) for color in channel_colors_hex]

    # Calculate the number of levels for the pyramid
    biggest_dim = max(img.shape[-2:])
    scale_num_levels = (
        int(np.ceil(np.log(biggest_dim / min_tile_size) / np.log(scale_factor))) + 1
    )

    if min_max_quantiles:
        im_xr = img.xarray_dask_data
        # This is run on the maximum projection to avoid overloading the memory
        quantiles_per_channel = (
            im_xr.squeeze()
            .max("Z")
            .quantile(min_max_quantiles, dim=["Y", "X"])
            .compute()
        )
        channel_minmax: List[Tuple[float, float]] = [
            tuple(row) for row in quantiles_per_channel.compute().values.T
        ]
    else:
        channel_minmax = [(0, 2**16 - 1) for _ in range(img.dims["C"][0])]

    if reverse_channels:
        channel_colors = channel_colors[::-1]
        img_data = img.dask_data[
            :, ::-1, ...
        ]  # reverse the channels assuming the order is TCZYX
        channel_minmax = channel_minmax[::-1]
    else:
        img_data = img.dask_data

    # Write the image data to OME-Zarr
    writer = OmeZarrWriter(output_zarr_path)
    with ProgressBar():
        writer.write_image(
            image_data=img_data,
            image_name=img_path.stem,
            physical_pixel_sizes=img.physical_pixel_sizes,
            channel_names=img.channel_names,
            channel_colors=channel_colors,
            channel_minmax=channel_minmax,
            scale_num_levels=scale_num_levels,
            scale_factor=scale_factor,
            dimension_order="TCZYX",
            # make it so all channels are in one chunk
            chunk_dims=(
                1,
                img.dims["C"][0],
                1,
                min_tile_size,
                min_tile_size,
            ),
        )


# if __name__ == "__main__":
#     import typer

#     typer.run(im_to_ome_zarr)
# %%
# root_folder = Path(
#     "~/Desktop/Link to YC/90br_CRC_ortho_2023Sep/FF-CRC-to-analyze"
# ).expanduser()
# files = sorted(root_folder.rglob("*.nd2"), key=lambda x: x.stat().st_size)
# im = AICSImage(list(files)[-1])
# im_arr = im.xarray_dask_data
# # %%
# im_arr.squeeze().max("Z").quantile([0.05, 0.999], dim=["Y", "X"]).compute()
# %%

"""
Usage Example:
root_folder = Path(
    "~/Desktop/Link to YC/90br_CRC_ortho_2023Sep/FF-CRC-to-analyze"
).expanduser()
files = sorted(root_folder.rglob("*.nd2"), key=lambda x: x.stat().st_size)
im_to_ome_zarr(
    list(files)[-1],
    "/gstore/scratch/u/lubecke/image_2.ome.zarr",
    overwrite=True,
    min_tile_size=3000,
    scale_factor=2.0,
    min_max_quantiles=(0.05, 0.999),
)
"""
