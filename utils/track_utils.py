import re

import dask.array as da
import numpy as np
import pandas as pd
import trackpy as tp
from scipy.spatial.distance import cdist

tp.linking.Linker.MAX_SUB_NET_SIZE_ADAPTIVE = 15


def _channel_crosstalk_matrix(a):
    """From Scry.
    TODO:Replace this
    Estimate and correct differences in channel intensity and spectral overlap among sequencing
    channels. For each channel, find points where the largest signal is from that channel. Use the
    median of these points to define new basis vectors. Describe with linear transformation w.

    so that w * x_array = y, where y is the corrected data.

    :param a: raw data to transform (read + t, c)
    :return: The inverse matrix, w
    """
    max_indices = a.argmax(axis=1)  # max indices per channel
    if isinstance(a, da.Array):
        max_indices = max_indices.compute()
        median_array = np.array(
            [
                da.median(a.vindex[np.where(max_indices == i)], axis=0)
                for i in range(a.shape[1])
            ]
        ).T
    else:
        median_array = np.array(
            [np.median(a[max_indices == i], axis=0) for i in range(a.shape[1])]
        ).T
    median_array = median_array / median_array.sum(axis=0)
    w = np.linalg.inv(median_array)
    only_inverse = True

    if only_inverse:
        return w
    y = w.dot(a.T).T.astype(int)
    return y, w


# Converts peaks at each timepoint into particles across time
# peaks_df: GeoPandas with 'time'=temporal index,'ch_{ix}'=intensity in ch{ix}
# search_radius : radius around particle position to link peaks together. In units of the supplied geometry
# min_lifespan tracks the minimum number of timepoints a particle must exist in
# peak_int_threshold is the peak intensity threshold below which we throw away a peak
def track_particles(peaks_df, search_radius, peak_int_threshold=0.0, min_lifespan=5):
    # Preprocess
    ccols = [x for x in peaks_df.columns if re.search("ch_([1-9]+)", x)]
    for c in ccols:
        peaks_df[c] = peaks_df[c].astype(np.float32)
    peaks_df["max_int"] = peaks_df.loc[:, ccols].max(axis=1).astype(np.float32)

    peaks_df["x"] = peaks_df.geometry.apply(lambda x: x.coords.xy[0][0])
    peaks_df["y"] = peaks_df.geometry.apply(lambda x: x.coords.xy[1][0])

    filt_peaks = peaks_df.loc[peaks_df.max_int > peak_int_threshold, :]

    # Track particles
    tracked_particles = tp.link(
        filt_peaks,
        search_radius,
        adaptive_step=0.90,
        adaptive_stop=5,
        t_column="time",
        memory=1,
        link_strategy="numba",
    )

    # Filter particles
    frequent_particles = tracked_particles["particle"].value_counts()
    frequent_particles = frequent_particles[frequent_particles >= min_lifespan].index
    frequent_particle_tracks = tracked_particles.loc[
        tracked_particles.particle.isin(frequent_particles), :
    ]

    # Remove channel crosstalk (neighboring channels have spectral bleed)
    crosstalk_matrix = _channel_crosstalk_matrix(
        frequent_particle_tracks.loc[:, ccols].values
    )
    compensated_data = crosstalk_matrix.dot(
        frequent_particle_tracks.loc[:, ccols].values.T
    ).T
    compensated_particle_tracks = frequent_particle_tracks.copy()
    compensated_particle_tracks.loc[:, ccols] = compensated_data

    # Add some info for each peak.
    compensated_particle_tracks["max_channel"] = compensated_particle_tracks.loc[
        :, ccols
    ].idxmax(axis=1)
    compensated_particle_tracks["max_intensity"] = compensated_particle_tracks.apply(
        lambda x: x[x.max_channel], axis=1
    )

    return compensated_particle_tracks


def barcode_particles(compensated_particle_tracks, channel_dict):
    # Reconstruct particles
    barcode_length = compensated_particle_tracks.groupby("particle").size().max()
    particle_df = pd.DataFrame(
        index=compensated_particle_tracks.particle.value_counts().index,
        columns=[
            "x",
            "y",
            "barcode",
        ]
        + [f"i_{ix}" for ix in range(6)],
    )

    pgroups = compensated_particle_tracks.groupby("particle")

    # Coordinates
    particle_df.loc[:, "x"] = pgroups.x.mean()
    particle_df.loc[:, "y"] = pgroups.y.mean()

    # Channel intensities per particle
    max_int = (
        compensated_particle_tracks.groupby(["time", "particle"])["max_intensity"]
        .first()
        .unstack()
        .T
    )
    for c in max_int:
        particle_df.loc[:, f"i_{c}"] = max_int[c]

    # Construct text barcode.
    channel_dict["0"] = "N"
    bcodes = (
        compensated_particle_tracks.groupby(["time", "particle"])["max_channel"]
        .first()
        .unstack()
    )
    bcodes = bcodes.fillna("0")
    bcodes = bcodes.apply(lambda x: "".join([channel_dict[z] for z in x]))

    particle_df.loc[:, "barcode"] = bcodes

    return particle_df


# Given a codebook that maps genes/perturbations to barcodes, does 1 bit error correction and returns detected + matched barcodes,
# along with the Hamming distance to the nearest valid code.
def barcode_to_perturbation(particle_df, ref_codebook):
    # Process reference codebook
    barcode_length = particle_df["barcode"].str.len().max()
    reference_barcodes = ref_codebook.spacer.str[:barcode_length]
    reference_barcode_arrays = np.array(
        [[ord(char) for char in barcode] for barcode in reference_barcodes]
    )

    # Call barcodes
    observed_barcode_arrays = np.array(
        [[ord(char) for char in barcode] for barcode in particle_df["barcode"]]
    )

    hamming_distances = cdist(
        observed_barcode_arrays, reference_barcode_arrays, metric="hamming"
    )
    min_hamming_distances = hamming_distances.min(axis=1) * len(reference_barcodes[0])
    closest_matches = hamming_distances.argmin(axis=1)
    matched_barcodes = reference_barcodes.values[closest_matches]

    particle_df["matched_barcode"] = matched_barcodes
    particle_df["hamming_distance"] = min_hamming_distances

    return particle_df, particle_df.query("hamming_distance < 2")
