# This script will take a stitched image and detect guide RNA peaks using the LoG filter
import argparse
import json

import geopandas
import numpy as np
import pandas as pd
import shapely

from utils import track_utils


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Adjusted to accept a list of GeoJSON files
    parser.add_argument(
        "peaks", nargs="+", help="Paths to GeoJSONs containing GeoPandas dataframe"
    )
    parser.add_argument("codebook_path", help="Path to CSV defining sgRNA guidebook")
    parser.add_argument("-o", "--output_path", help="Output GeoJSON name")
    parser.add_argument(
        "-r",
        "--search_radius",
        required=True,
        help="Search radius for combining particles (in units of geometry)",
    )
    parser.add_argument(
        "-c",
        "--channel_map",
        required=True,
        help="JSON dictionary that defines which base maps to which channel",
    )
    parser.add_argument(
        "--codebook_name_col",
        help="Name column in codebook",
        default="",
        required=False,
    )
    args = parser.parse_args()

    return {
        "peaks": args.peaks,
        "codebook": pd.read_csv(args.codebook_path, index_col=0),
        "output_path": args.output_path,
        "search_radius": float(args.search_radius),
        "channel_map": json.load(open(args.channel_map, "r")),
    }


def main():
    adict = parse_args()

    # Geometries
    all_peaks = pd.concat(
        [geopandas.read_file(x, driver="GeoJSON") for x in adict["peaks"]]
    )

    # Convert time to integers
    all_peaks["time"] = all_peaks["time"].astype("str")

    timestamps = np.unique(all_peaks["time"])
    timeorder = np.argsort([np.datetime64(x) for x in timestamps])
    tdict = {y: x for x, y in zip(timeorder, timestamps)}

    all_peaks["time"] = all_peaks["time"].apply(lambda x: tdict[x])

    # Tracking
    particles = track_utils.track_particles(
        all_peaks,
        adict["search_radius"],
    )

    # Barcoding
    particles = track_utils.barcode_particles(particles, adict["channel_map"])
    particles, call_rate = track_utils.barcode_to_perturbation(
        particles, adict["codebook"]
    )

    print(f"Call rate: {call_rate.shape[0]/particles.shape[0]}")

    # GeoJSON
    particle_gd = geopandas.GeoSeries(
        data=[shapely.Point([x.x, x.y]) for _, x in particles.iterrows()],
        index=particles.index,
    )

    called_peaks = particles.drop(["x", "y"], axis=1)
    called_peaks = geopandas.GeoDataFrame(
        data=called_peaks,
        geometry=particle_gd,
    )
    barcode_length = particles["barcode"].str.len().max()

    codebook = adict["codebook"].set_index(
        adict["codebook"].spacer.str[:barcode_length]
    )
    # Determine the appropriate column name for identifiers using match statement
    match codebook.columns:
        case columns if "gene_symbol" in columns:
            name_col = "gene_symbol"
        case columns if "ID" in columns:
            name_col = "ID"
        case _:
            name_col = ""  # Default case if no match is found

    if adict.get("codebook_name_col", ""):
        name_col = adict["codebook_name_col"]

    # Apply the perturbation labels based on the matched barcode
    if name_col:
        called_peaks["perturbation"] = called_peaks["matched_barcode"].map(
            lambda x: str(codebook.at[x, name_col])
        )
    called_peaks.to_file(adict["output_path"], driver="GeoJSON")

    return


if __name__ == "__main__":
    main()
