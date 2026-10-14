"""BP+LSD decoders for sinter, with the BP and LSD settings of the paper's run.

The paper's two_block_measurement.py decodes with ldpc's BpLsdDecoder on the
full circuit-level detector error model (decompose_errors=False): min-sum BP
(scaling 0.9, serial schedule, 30 iterations) followed by localized-statistics
decoding (lsd_cs, order 4).  That is the decoder "bplsd_full" here.  At the
sizes of this campaign it costs seconds per shot, so the campaign decodes each
CSS sector on its own ("bplsd_X", "bplsd_Z"), with the same BP and LSD
settings:

  * the X experiment's observables (reported bit, X logicals) are X-type and
    are flipped only by the Z part of a fault, which is seen only by X-type
    detectors (detector coordinate 2 in {0: X checks, 2: Gauss laws});
  * the Z experiment uses the Z-type detectors (coordinate 2 in {1: Z checks,
    3: flux checks}).

Restricting a mechanism to one sector keeps its effect on that sector's
detectors and observables; mechanisms with the same restricted effect are
merged, with probability q = q1 (1 - q2) + q2 (1 - q1) (independent XOR).
This is standard independent X/Z decoding: it ignores the correlation that a
Y fault induces between the sectors.  All faults are still sampled, so the
noise is never made weaker; only the decoder uses less information.
decoder_check.py compares both decoders on identical shots.

Shots with no detection event in the sector are predicted as "no flip" without
calling the decoder (BP returns the zero correction for a zero syndrome).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import sinter
import stim

BP_LSD_SETTINGS = dict(max_iter=30, bp_method="minimum_sum", ms_scaling_factor=0.9,
                       schedule="serial", lsd_method="lsd_cs", lsd_order=4)


def dem_mechanisms(dem: stim.DetectorErrorModel):
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets, obs = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
        yield dets, obs, inst.args_copy()[0]


def sector_matrices(dem: stim.DetectorErrorModel, sectors=None):
    """(H, L, priors, keep).  sectors=None: full model, no merging.

    keep: indices of the detectors used (all of them for the full model).
    """
    nd, no = dem.num_detectors, dem.num_observables
    if sectors is None:
        keep = np.arange(nd)
        cols = [(tuple(d), tuple(o), p) for d, o, p in dem_mechanisms(dem)]
    else:
        coords = dem.get_detector_coordinates()
        keep = np.array(sorted(d for d, c in coords.items() if len(c) > 2 and int(c[2]) in sectors),
                        dtype=np.int64)
        remap = {int(d): i for i, d in enumerate(keep)}
        merged = {}
        for dets, obs, p in dem_mechanisms(dem):
            dd = []
            for d in dets:
                j = remap.get(d)
                if j is not None:
                    dd.append(j)
            dd, oo = tuple(sorted(dd)), tuple(sorted(obs))
            if not dd:
                continue            # undetectable in this sector: the decoder cannot act on it
            key = (dd, oo)
            q = merged.get(key, 0.0)
            merged[key] = q * (1 - p) + p * (1 - q)
        cols = [(d, o, p) for (d, o), p in merged.items()]
    r, c, lr, lc, pri = [], [], [], [], []
    for j, (dets, obs, p) in enumerate(cols):
        for d in dets:
            r.append(d); c.append(j)
        for o in obs:
            lr.append(o); lc.append(j)
        pri.append(p)
    ne = len(cols)
    H = sp.csr_matrix((np.ones(len(r), np.uint8), (r, c)), shape=(len(keep), ne))
    L = sp.csr_matrix((np.ones(len(lr), np.uint8), (lr, lc)), shape=(no, ne))
    H.data &= 1; L.data &= 1
    H.eliminate_zeros(); L.eliminate_zeros()
    return H, L, np.array(pri), keep


class _CompiledBpLsd(sinter.CompiledDecoder):
    def __init__(self, dec, L, keep, nd, no):
        self.dec, self.L, self.keep, self.nd, self.no = dec, L.tocsr(), keep, nd, no

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        dets = np.unpackbits(bit_packed_detection_event_data, axis=1, bitorder="little")[:, :self.nd]
        dets = dets[:, self.keep]
        out = np.zeros((dets.shape[0], self.no), np.uint8)
        for i in np.flatnonzero(dets.any(axis=1)):
            corr = self.dec.decode(np.ascontiguousarray(dets[i]))
            out[i] = (self.L @ corr) % 2
        return np.packbits(out, axis=1, bitorder="little")


class BpLsd(sinter.Decoder):
    """sectors: None (full model, the paper's decoder), {0, 2} (X) or {1, 3} (Z)."""

    def __init__(self, sectors=None, **settings):
        self.sectors = None if sectors is None else frozenset(sectors)
        self.settings = dict(BP_LSD_SETTINGS, **settings)

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        from ldpc import BpLsdDecoder
        H, L, pri, keep = sector_matrices(dem, self.sectors)
        dec = BpLsdDecoder(H, error_channel=list(pri), **self.settings)
        return _CompiledBpLsd(dec, L, keep, dem.num_detectors, dem.num_observables)

    def decode_via_files(self, *, num_shots, num_dets, num_obs, dem_path, dets_b8_in_path,
                         obs_predictions_b8_out_path, tmp_dir):
        dem = stim.DetectorErrorModel.from_file(dem_path)
        comp = self.compile_decoder_for_dem(dem=dem)
        data = stim.read_shot_data_file(path=dets_b8_in_path, format="b8",
                                        num_detectors=num_dets, bit_packed=True)
        pred = comp.decode_shots_bit_packed(bit_packed_detection_event_data=data)
        stim.write_shot_data_file(data=pred, path=obs_predictions_b8_out_path, format="b8",
                                  num_observables=num_obs)


def custom_decoders():
    return {"bplsd_X": BpLsd({0, 2}), "bplsd_Z": BpLsd({1, 3}), "bplsd_full": BpLsd(None)}
