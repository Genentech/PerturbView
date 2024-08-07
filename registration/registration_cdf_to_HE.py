import argparse
import json
import os

import itk
import numpy as np
import pandas as pd
import SimpleITK as sitk
from wsireg.parameter_maps.preprocessing import ImagePreproParams
from wsireg.reg_images.loader import reg_image_loader

# sys.path.append('/gne/data/t3imagedata/conrad_store/modules/')
# from wsireg_mod import WsiReg2D
# from wsireg_mod.img_utils import *


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


# Load linked image arrays into memory for registration.
def load_image(pdict):
    img_file = pdict["image_fp"]
    imarr = np.load(img_file)
    pdict["image_fp"] = imarr.astype(np.int16)

    if "image_mask" in pdict:
        pdict["image_mask"] = np.load(pdict["image_mask"]).astype(np.uint16)

    return pdict


# Handle CLI arguments.
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("mv_img", help="Moving image parameters (json dict)")
    parser.add_argument("fx_img", help="Fixed image parameters (json dict)")
    parser.add_argument("transform", help="Registration parameter list")
    parser.add_argument("-o", "--output_dir", help="Output directory")

    args = parser.parse_args()

    src_img_name = args.mv_img
    tgt_img_name = args.fx_img
    transform = args.transform

    # Load moving image dict
    with open(src_img_name, "r") as f:
        src_img_dict = json.load(f)
    src_img_dict = load_image(src_img_dict)

    # Fixed image dict
    with open(tgt_img_name, "r") as f:
        tgt_img_dict = json.load(f)
    tgt_img_dict = load_image(tgt_img_dict)

    # Registration dictionary
    with open(transform, "r") as f:
        reg_params = json.load(f)

    if pd.isnull(args.output_dir):
        print("No output directory specified, dumping to current directory.")
        odir = os.getcwd()
    else:
        odir = args.output_dir

    # Return parsed arg dict.
    return {
        "src_img_dict": src_img_dict,
        "tgt_img_dict": tgt_img_dict,
        "output_dir": odir + r"/",
        "reg_params": reg_params,
    }


def main():
    print("Parsing arguments...")
    adict = parse_args()
    if pd.isnull(adict):
        return

    os.makedirs(adict["output_dir"], exist_ok=True)

    src = adict["src_img_dict"]
    tgt = adict["tgt_img_dict"]

    # Load images into ITK images.
    print("Loading images into memory...")
    src_img = reg_image_loader(
        src["image_fp"],
        src["image_res"],
        preprocessing=ImagePreproParams(**src["preprocessing"]),
    )
    src_img.read_reg_image()
    src_img.reg_image_sitk_to_itk(cast_to_float32=False)

    tgt_img = reg_image_loader(
        tgt["image_fp"],
        tgt["image_res"],
        preprocessing=ImagePreproParams(**tgt["preprocessing"]),
    )
    tgt_img.read_reg_image()
    tgt_img.reg_image_sitk_to_itk(cast_to_float32=False)

    # Registration object creation
    print("Registering images...")
    selx = itk.ElastixRegistrationMethod.New()
    selx.SetMovingImage(src_img.reg_image)
    selx.SetFixedImage(tgt_img.reg_image)

    # Load image masks (if they exist)
    if "image_mask" in src:
        # src['preprocessing']['as_uint8'] = True
        src_mask = reg_image_loader(
            src["image_mask"],
            src["image_res"],
            preprocessing=None,  # ImagePreproParams(**src['preprocessing']),
        )
        src_mask.read_reg_image()
        src_mask.reg_image_sitk_to_itk(cast_to_float32=False)

        selx.SetMovingMask(src_mask.reg_image)

    if "image_mask" in tgt:
        # tgt['preprocessing']['as_uint8'] = True
        tgt_mask = reg_image_loader(
            tgt["image_mask"],
            tgt["image_res"],
            preprocessing=None,  # ImagePreproParams(**tgt['preprocessing']),
        )
        tgt_mask.read_reg_image()
        tgt_mask.reg_image_sitk_to_itk(cast_to_float32=False)

        selx.SetFixedMask(tgt_mask.reg_image)

    # Get default WSIReg parameters. Note that this is setup to do a single alignment.
    # Therefore we always assume len(reg_graph.reg_graph_edges) = 1.
    # Load parameters into selx. This is just a dictionary that can be modified to our liking.
    parameter_object_registration = itk.ParameterObject.New()
    for idx, pmap in enumerate(adict["reg_params"]):
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
    selx.SetOutputDirectory(adict["output_dir"])
    selx.SetLogFileName("selx_log.txt")
    selx.LogToFileOn()

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

    print("Saving transformations...")
    with open(adict["output_dir"] + "transforms.json", "w") as f:
        json.dump(tform_composite, f)
    print("Done!")
    return


if __name__ == "__main__":
    main()
