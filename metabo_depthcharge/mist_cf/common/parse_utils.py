# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""parse_utils.py"""

from itertools import groupby
from pathlib import Path

import numpy as np
from tqdm import tqdm


def max_thresh_spec(spec, max_peaks=100, inten_thresh=0.003):
    spec_masses, spec_intens = spec[:, 0], spec[:, 1]
    new_sort_order = np.argsort(spec_intens)[::-1]
    new_sort_order = new_sort_order[:max_peaks]

    spec_masses = spec_masses[new_sort_order]
    spec_intens = spec_intens[new_sort_order]

    spec_mask = spec_intens > inten_thresh
    spec_masses = spec_masses[spec_mask]
    spec_intens = spec_intens[spec_mask]
    out_ar = np.vstack([spec_masses, spec_intens]).transpose(1, 0)
    return out_ar


def process_spec_file_unbinned(spec_name, data_dir, precision=4, return_meta=False):
    spec_file = data_dir / "spec_files" / f"{spec_name}.ms"
    meta, tuples = parse_spectra(spec_file)
    parent_mass = float(meta.get("precursor_mz", meta["parentmass"]))

    fused_tuples = [x for _, x in tuples if x.size > 0]
    if len(fused_tuples) == 0:
        return (spec_name, None)
    merged_spec = merge_spec_tuples(fused_tuples, parent_mass, precision=precision)

    if merged_spec.size == 0:
        return (spec_name, None)

    if return_meta:
        return (meta, merged_spec)
    else:
        return (spec_name, merged_spec)


def merge_spec_tuples(spec_list, parent_mass, precision=4):
    mz_ind_to_inten = {}
    mz_ind_to_mz = {}
    for i in spec_list:
        for tup in i:
            mz, inten = tup
            mz_ind = np.round(mz, precision)
            cur_inten = mz_ind_to_inten.get(mz_ind)
            if cur_inten is None or inten > cur_inten:
                mz_ind_to_inten[mz_ind] = inten
                mz_ind_to_mz[mz_ind] = mz

    new_tuples = []
    for k, mz in mz_ind_to_mz.items():
        new_tuples.append(np.array([mz, mz_ind_to_inten[k]]))
    merged_spec = np.vstack(new_tuples)
    merged_spec = merged_spec[merged_spec[:, 0] <= (parent_mass + 1)]
    if merged_spec.size != 0:
        merged_spec[:, 1] = merged_spec[:, 1] / merged_spec[:, 1].max()
    return merged_spec


def parse_spectra(spectra_file):
    lines = [i.strip() for i in open(spectra_file).readlines()]

    group_num = 0
    metadata = {}
    spectras = []
    my_iterator = groupby(
        lines, lambda line: line.startswith(">") or line.startswith("#")
    )

    for _index, (_start_line, lines) in enumerate(my_iterator):
        group_lines = list(lines)
        subject_lines = list(next(my_iterator)[1])
        if group_num > 0:
            spectra_header = group_lines[0].split(">")[1]
            peak_data = [
                [float(x) for x in peak.split()[:2]]
                for peak in subject_lines
                if peak.strip()
            ]
            if len(peak_data):
                peak_data = np.vstack(peak_data)
                spectras.append((spectra_header, peak_data))
        else:
            entries = {}
            for i in group_lines:
                if " " not in i:
                    continue
                elif i.startswith("#INSTRUMENT TYPE"):
                    key = "#INSTRUMENT TYPE"
                    val = i.split(key)[1].strip()
                    entries[key[1:]] = val
                else:
                    start, end = i.split(" ", 1)
                    start = start[1:]
                    while start in entries:
                        start = f"{start}'"
                    entries[start] = end

            metadata.update(entries)
        group_num += 1

    metadata["_FILE_PATH"] = spectra_file
    metadata["_FILE"] = Path(spectra_file).stem
    return metadata, spectras


def parse_spectra_mgf(mgf_file, max_num=None):
    def key(x):
        return x.strip() == "BEGIN IONS"

    parsed_spectra = []
    with open(mgf_file) as fp:
        for is_header, group in tqdm(groupby(fp, key)):
            if is_header:
                continue

            meta = {}
            spectra = []
            cur_spectra_name = "spec"
            cur_spectra = []
            group = list(group)
            for line in group:
                line = line.strip()
                if not line or line == "END IONS" or line == "BEGIN IONS":
                    pass
                elif "=" in line:
                    k, v = (i.strip() for i in line.split("=", 1))
                    meta[k] = v
                else:
                    mz, intens = line.split()
                    cur_spectra.append((float(mz), float(intens)))

            if len(cur_spectra) > 0:
                cur_spectra = np.vstack(cur_spectra)
                spectra.append((cur_spectra_name, cur_spectra))
                parsed_spectra.append((meta, spectra))

            if max_num is not None and len(parsed_spectra) > max_num:
                break
        return parsed_spectra
