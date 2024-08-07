import sys
from pathlib import Path

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from aicsimageio import AICSImage
from sklearn.neighbors import KDTree


def integrate_xenium(peaks, xenium_dir):
    rdr = AICSImage(Path(xenium_dir, "morphology_mip.ome.tif"))

    peaks = peaks.set_geometry(
        [
            shapely.affinity.scale(
                x,
                xfact=rdr.physical_pixel_sizes.X,
                yfact=rdr.physical_pixel_sizes.Y,
                origin=(0, 0, 0),
            )
            for x in peaks["geometry"]
        ]
    )

    cells = pd.read_csv(Path(xenium_dir, "cells.csv.gz"))
    cells = cells.set_index("cell_id")

    test = pd.read_csv(Path(xenium_dir, "transcripts.csv.gz"))
    cxg = (
        test.groupby("cell_id")["feature_name"]
        .value_counts()
        .unstack()
        .fillna(0)
        .astype(np.uint16)
    )
    cxg = cxg.loc[cxg.index != "UNASSIGNED", :]

    adata = ad.AnnData(cxg, obs=cells.loc[cxg.index, :])
    adata.obsm["spatial"] = np.asarray(adata.obs.loc[:, ["x_centroid", "y_centroid"]])

    gpts = np.array(
        [(x.coords.xy[0][0], x.coords.xy[1][0]) for x in peaks.geometry]
    ).squeeze()

    tree = KDTree(gpts)
    z = tree.query(adata.obsm["spatial"], k=1, return_distance=False)
    for f in ["matched_barcode", "hamming_distance"]:
        adata.obs[f] = peaks.loc[z.squeeze(), f].values

    return adata


if __name__ == "__main__":
    peaks_file = sys.argv[1]
    xenium_dir = sys.argv[2]
    output_file = sys.argv[3]

    peaks = gpd.read_file(peaks_file, driver="GeoJSON")
    adata = integrate_xenium(peaks, xenium_dir)
    adata.write_h5ad(output_file)
