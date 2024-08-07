# Script that will align WS image from stitching with Xenium
""" "
example:
python scripts/register_xenium.py /gnet/is1/p01/shares/ctg-microscopy/EL/perturbview_output/eTK146A/intermediate_outputs/20240225_204545_612_stitched.zarr /gnet/is1/p01/shares/ctg-microscopy/EL/perturbview_output/eTK146A/intermediate_outputs/pheno.zarr -o /gnet/is1/p01/shares/ctg-microscopy/EL/perturbview_output/eTK146A/intermediate_outputs/20240225_204545_612_transforms.json
"""

import argparse
import json
from pathlib import Path

import dask.array as da
import itk
import numpy as np
import SimpleITK as sitk
from aicsimageio import AICSImage
from wsireg import WsiReg2D
from wsireg.parameter_maps.preprocessing import ImagePreproParams
from wsireg.reg_images.loader import reg_image_loader
from wsireg.utils.reg_utils import _prepare_reg_models


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "stitched_img_path", required=True, help="Path to ZARR output of stitching"
    )
    parser.add_argument(
        "xenium_img_path",
        required=True,
        help="Path to Xenium morphology image. We use the maximum projection.",
    )
    parser.add_argument(
        "-o", "--output_path", required=True, help="Output name of stitched image."
    )
    args = parser.parse_args()

    return {
        "src_img": args.stitched_img_path,
        "fix_img": args.xenium_img_path,
        "out_path": args.output_path,
    }


# Here we assume (from the current Xenium output) that we only have one channel - DAPI
def prepare_xenium_image(img_path, reg_channel=0):
    imobj = AICSImage(img_path)
    imarr = imobj.get_image_dask_data("YX", C=reg_channel, Z=0, T=0)

    hi, lo = da.percentile(imarr.ravel(), [2, 98])
    narr = da.clip((imarr.astype(np.float32) - lo) / (hi - lo), 0, 1)
    narr = (narr * 32768).astype(np.int16)

    imdict = dict(
        image_fp=narr,
        image_res=imobj.physical_pixel_sizes.X,
        channel_names=["DAPI"],
        channel_colors=["red"],
        preprocessing={
            "image_type": "FL",
            "ch_indices": [0],
            "as_uint8": False,
            "contrast_enhance": False,
        },
    )

    return imdict


# This will load a single image. Both registration channel and umperpix are hardcoded currently
# rotation angle is best specified ahead of time to improve convergence.
def prepare_ISS_image(
    img_path, umperpix=0.65, rot_angle=-90, reg_channel=0, color="blue"
):
    if str(img_path).endswith(".zarr") and not str(img_path).endswith(".ome.zarr"):
        imarr = da.from_zarr(img_path)
    else:
        imarr = AICSImage(img_path).get_image_dask_data("YX", C=reg_channel, Z=0, T=0)

    hi, lo = da.percentile(imarr.ravel(), [2, 98])
    narr = da.clip((imarr.astype(np.float32) - lo) / (hi - lo), 0, 1)
    narr = (narr * 32768).astype(np.int16)

    imdict = dict(
        image_fp=narr,
        image_res=umperpix,
        channel_names=["DAPI"],
        channel_colors=[color],
        preprocessing={
            "image_type": "FL",
            "ch_indices": [0],
            "as_uint8": False,
            "contrast_enhance": False,
            "rot_cc": rot_angle,
        },
    )

    return imdict


# image conversion
def itk_image_to_sitk_image(image):
    origin = tuple(image.GetOrigin())
    spacing = tuple(image.GetSpacing())
    direction = itk.GetArrayFromMatrix(image.GetDirection()).flatten()
    image = sitk.GetImageFromArray(
        itk.GetArrayFromImage(image),
        isVector=image.GetNumberOfComponentsPerPixel() > 1,
    )
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    return image


def main(
    src_image: Path,
    fixed_image: Path,
    output_path: Path,
    reg_params=["rigid", "nl"],
    reg_channel=0,
    debug=True,
):
    # Parse images
    if Path(fixed_image).suffix == ".zarr":
        tgt_img = prepare_ISS_image(
            fixed_image, color="red", rot_angle=0, reg_channel=reg_channel
        )
        src_img = prepare_ISS_image(src_image, rot_angle=0, reg_channel=reg_channel)
    else:
        tgt_img = prepare_xenium_image(fixed_image, reg_channel=reg_channel)
        src_img = prepare_ISS_image(src_image, reg_channel=reg_channel)

    # Configure elastix
    reg_graph = WsiReg2D()

    reg_graph.add_modality("ISS", **src_img)
    reg_graph.add_modality("Xenium", **tgt_img)

    reg_graph.add_reg_path(
        "ISS",
        "Xenium",
        thru_modality=None,
        reg_params=reg_params,
    )

    reg_graph.cache_images = False
    reg_params = _prepare_reg_models(reg_graph.reg_graph_edges[0]["params"])

    # Some parameters that worked for Xenium-ISS. Making this visible for the power users who know elastix.
    # Don't really want to handle the generic case for this use case.
    for pmap in reg_params:
        pmap["UseMultiThreadingForMetrics"] = ["false"]
        pmap["MaximumNumberOfIterations"] = [
            "5000"
        ]  # This was key for this alignment task. Unfortunately this means it takes 1 hour to align instead of 10mins...
        pmap["MaximumNumberOfSamplingAttempts"] = ["1000"]

    src_img = reg_image_loader(
        src_img["image_fp"],
        src_img["image_res"],
        preprocessing=ImagePreproParams(**src_img["preprocessing"]),
    )
    src_img.read_reg_image()
    src_img.reg_image_sitk_to_itk(cast_to_float32=False)

    tgt_img = reg_image_loader(
        tgt_img["image_fp"],
        tgt_img["image_res"],
        preprocessing=ImagePreproParams(**tgt_img["preprocessing"]),
    )
    tgt_img.read_reg_image()
    tgt_img.reg_image_sitk_to_itk(cast_to_float32=False)

    # Registration object creation
    selx = itk.ElastixRegistrationMethod.New()
    selx.SetMovingImage(src_img.reg_image)
    selx.SetFixedImage(tgt_img.reg_image)

    parameter_object_registration = itk.ParameterObject.New()
    for idx, pmap in enumerate(reg_params):
        if idx == 0:
            if "WriteResultImage" not in pmap:
                pmap["WriteResultImage"] = ["false"]
            if "AutomaticTransformInitialization" not in pmap:
                pmap["AutomaticTransformInitialization"] = ["true"]  # true

            parameter_object_registration.AddParameterMap(pmap)
        else:
            pmap["WriteResultImage"] = ["false"]
            pmap["AutomaticTransformInitialization"] = ["false"]

            parameter_object_registration.AddParameterMap(pmap)

    # Set registration parameters
    selx.SetParameterObject(parameter_object_registration)

    # Debugging
    selx.SetLogToConsole(False)  # Important to stop crashing (for some reason?)
    selx.SetNumberOfThreads(4)
    p = Path(output_path).parent
    p.mkdir(parents=True, exist_ok=True)
    selx.SetOutputDirectory(str(p))

    # This is the registration loop.
    selx.UpdateLargestPossibleRegion()

    # Read out the registration transforms.
    result_transform_parameters = selx.GetTransformParameterObject()

    tform_list = []
    for idx in range(result_transform_parameters.GetNumberOfParameterMaps()):
        tform = {}
        for k, v in result_transform_parameters.GetParameterMap(idx).items():
            tform[k] = v
        tform_list.append(tform)

    # Save both pre-registration and registration transforms into a json dictionary.
    tform_composite = {
        "pre_reg_transforms": src_img.pre_reg_transforms,
        "reg_transforms": tform_list,
    }

    with open(output_path, "w") as f:
        json.dump(tform_composite, f)

    return


if __name__ == "__main__":
    args = parse_args()
    main(args["src_img"], args["fix_img"], args["out_path"])
