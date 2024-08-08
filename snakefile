from pathlib import Path
import json

# Define constants and input parameters
script_dir = "scripts"  # Assuming scripts are directly callable

sub_dir = "eTK146B"
sample_dir = Path(f"/gstore/data/ctgbioinfo/kudot3/eTK146/{sub_dir}/ISS")
pheno_dir = Path(f"/gstore/data/ctgbioinfo/kudot3/eTK146/{sub_dir}/pheno")
grnas = "/gstore/data/ctgbioinfo/kudot3/eTK128/gRNAs_mouseMinimal1.csv"
output_dir = Path(
    "/gnet/is1/p01/shares/ctg-microscopy/EL/perturbview_output/"
).joinpath(sub_dir)
intermediate_output_dir = output_dir.joinpath("intermediate_outputs")
pheno_img = intermediate_output_dir.joinpath("pheno.zarr")
temp_dir = "/gstore/scratch/u/lubecke"
pixel_size = 0.65

# Ensure output directories exist
intermediate_output_dir.mkdir(parents=True, exist_ok=True)
intermediate_output_dir.joinpath("segmented").mkdir(parents=True, exist_ok=True)

# Gets subdirectories for each image set
samples = [
    next(d.glob("*")).parent
    for d in sample_dir.glob("*")
    if d.is_dir() and not d.name.startswith("DAPI")
]

print(f"Processing: {samples}")

print("\n" * 3)

image_path = [
    d for d in pheno_dir.glob("*") if d.suffix in [".tif", ".tiff", ".nd2", ".ome.tiff"]
]

images = [d.stem for d in image_path]

print(f"Segmenting: {images}")


rule all:
    input:
        #stitch_images
        expand(
            str(intermediate_output_dir.joinpath("{sample.name}_stitched.zarr")),
            sample=samples,
        ),
        # get_registration 
        expand(
            str(intermediate_output_dir.joinpath("{sample.name}_transforms.json")),
            sample=samples,
        ),
        # register_images
        expand(
            str(intermediate_output_dir.joinpath("{sample.name}_registered.zarr")),
            sample=samples,
        ),
        # call_peaks
        expand(
            str(intermediate_output_dir.joinpath("{sample.name}_peaks.geojson")),
            sample=samples,
        ),
        # track particles
        str(intermediate_output_dir.joinpath("particles.geojson")),
        # # # make points image
        str(intermediate_output_dir.joinpath("barcodes.zarr")),
        # segment_phenotype
        expand(
            str(
                intermediate_output_dir.joinpath(
                    "segmented", "{image_name}_segmented.ome.tiff"
                )
            ),
            image_name=[d.stem for d in image_path],
        ),
        # stitch_pheno
        str(intermediate_output_dir.joinpath("pheno_segmented.zarr")),
        # get_cell_hulls
        str(intermediate_output_dir.joinpath("nuclei.parquet")),
        # bg_subtract_pheno
        str(intermediate_output_dir.joinpath("pheno_bg.zarr")),


rule segment_phenotype:
    input:
        lambda wildcards: str(next(pheno_dir.glob(f"{wildcards.image}.*"), None)),
    output:
        intermediate_output_dir.joinpath("segmented", "{image}_segmented.ome.tiff"),
    threads: 14
    shell:
        "python utils/segment_nuclei.py {input} 0 {output} --dimensionality 2D --z-slice auto"  # --zoom 2.0"  # --flip-x --flip-y"


rule stitch_pheno:
    input:
        dir_path=pheno_dir,
    output:
        im=directory(str(pheno_img)),
        stitch_file=intermediate_output_dir.joinpath("pheno.zarr_stitching_results.csv"),
    threads: 4
    shell:
        (
            "python scripts/stitch_images.py {input.dir_path} -f 0.5 -o {output.im} -x True -y True"
        )


rule stitch_segment:
    input:
        im_dir=intermediate_output_dir.joinpath("segmented"),
        stitch_file=intermediate_output_dir.joinpath("pheno.zarr_stitching_results.csv"),
    output:
        directory(str(intermediate_output_dir.joinpath("pheno_segmented.zarr"))),
    threads: 4
    shell:
        (
            "python utils/stitch_utils.py {input.im_dir} {input.stitch_file} {output} --flip-x --flip-y"
        )


rule stitch_images:
    input:
        dir_path=sample_dir.joinpath("{sample}"),
    threads: 4
    output:
        directory(str(intermediate_output_dir) + "/{sample}_stitched.zarr"),
    shell:
        """
        python scripts/stitch_images.py {input.dir_path} -f 0.5 -o {output} -x True -y True
        """


rule get_registration:
    input:
        reg_im=pheno_img,
        stitched="{output_dir}/intermediate_outputs/{sample}_stitched.zarr",
    threads: 7
    output:
        transforms="{output_dir}/intermediate_outputs/{sample}_transforms.json",
    shell:
        (
            "python scripts/register_xenium.py {input.stitched}"
            + " {input.reg_im} "
            + "-o {output.transforms}"
        )


rule register_images:
    input:
        stitched="{output_dir}/intermediate_outputs/{sample}_stitched.zarr",
        transforms="{output_dir}/intermediate_outputs/{sample}_transforms.json",
    output:
        registered=directory(
            "{output_dir}/intermediate_outputs/{sample}_registered.zarr"
        ),
    shell:
        (
            "python utils/register_dask_im.py "
            "{input.stitched} "
            "{output.registered} "
            "{input.transforms} "
            "--pixel-size {pixel_size} "
        )


rule call_peaks:
    input:
        registered="{output_dir}/intermediate_outputs/{sample}_registered.zarr",
    output:
        peaks="{output_dir}/intermediate_outputs/{sample}_peaks.geojson",
    threads: 12
    shell:
        # "python scripts/call_bases.py {input.registered} -o {output.peaks} -t {temp_dir} -p spotiflow -s 1.6 -r 2 -d 5 --xenium False"
        "python scripts/call_bases.py {input.registered} -o {output.peaks} -t {temp_dir} -p spotiflow -r 2 -d 5 --no-xenium"



# Channel map JSON creation
channel_map = output_dir / "channel_map.json"
with open(channel_map, "w") as f:
    json.dump({"ch_1": "G", "ch_2": "T", "ch_3": "A", "ch_4": "C", "0": "N"}, f)


rule track_particles:
    input:
        peaks=expand(
            str(intermediate_output_dir.joinpath("{sample.name}_peaks.geojson")),
            sample=samples,
        ),
    output:
        particles="{output_dir}/intermediate_outputs/particles.geojson",
    params:
        peaks=lambda wildcars: " ".join(
            expand(
                str(intermediate_output_dir.joinpath("{sample.name}_peaks.geojson")),
                sample=samples,
            )
        ),
    shell:
        "python {script_dir}/track_particles.py {params.peaks} "
        "{grnas} "
        "--output_path {output.particles} "
        "--search_radius 30 "
        "--channel_map {channel_map}"


rule make_points_image:
    input:
        particles="{intermediate_output_dir}/particles.geojson",
        registered=expand(
            str(intermediate_output_dir.joinpath("{sample.name}_registered.zarr")),
            sample=samples,
        ),
    output:
        points_image=directory("{intermediate_output_dir}/barcodes.zarr"),
    shell:
        "python {script_dir}/points_to_image.py {input.particles} {output.points_image} {input.registered} "


# Extremely slow (~4hrs for 1 image)
rule get_nuclei_masks:
    input:
        segmented=str(intermediate_output_dir.joinpath("pheno_segmented.zarr")),
    output:
        hulls=str(intermediate_output_dir.joinpath("nuclei.parquet")),
    threads: 12
    shell:
        "python {script_dir}/cell_masks_to_hulls.py {input.segmented} {output.hulls}"


rule bg_subtract_pheno:
    input:
        pheno=intermediate_output_dir.joinpath("pheno.zarr"),
    output:
        pheno_bg=directory(intermediate_output_dir.joinpath("pheno_bg.zarr")),
    threads: 12
    shell:
        "python utils/bg_subtract.py {input.pheno} {output.pheno_bg} 10"
