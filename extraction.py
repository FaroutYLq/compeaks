import numpy as np
import strax
import cutax
from straxen import pre_apply_function
from straxen.plugins.peak_processing import PeakBasics
import simwrap

export, __all__ = strax.exporter()

SUPPORTED_SIGNALS = ["KrS1A", "KrS1B", "ArS1", "sim_KrS1A", "sim_KrS1B"]
PEAK_BASICS_DTYPE = [
    (("Start time of the peak (ns since unix epoch)", "time"), np.int64),
    (("End time of the peak (ns since unix epoch)", "endtime"), np.int64),
    (
        ("Weighted center time of the peak (ns since unix epoch)", "center_time"),
        np.int64,
    ),
    (("Peak integral in PE", "area"), np.float32),
    (("Number of PMTs contributing to the peak", "n_channels"), np.int16),
    (("PMT number which contributes the most PE", "max_pmt"), np.int16),
    (
        ("Area of signal in the largest-contributing PMT (PE)", "max_pmt_area"),
        np.float32,
    ),
    (("Total number of saturated channels", "n_saturated_channels"), np.int16),
    (
        ("Width (in ns) of the central 50% area of the peak", "range_50p_area"),
        np.float32,
    ),
    (
        ("Width (in ns) of the central 90% area of the peak", "range_90p_area"),
        np.float32,
    ),
    (
        (
            "Fraction of area seen by the top array (NaN for peaks with non-positive area)",
            "area_fraction_top",
        ),
        np.float32,
    ),
    (("Length of the peak waveform in samples", "length"), np.int32),
    (("Time resolution of the peak waveform in ns", "dt"), np.int16),
    (("Time between 10% and 50% area quantiles [ns]", "rise_time"), np.float32),
    (("Hits within tight range of mean", "tight_coincidence"), np.int16),
    (("PMT channel within tight range of mean", "tight_coincidence_channel"), np.int16),
    (("Classification of the peak(let)", "type"), np.int8),
]
EVENT_MERGED_DTYPE = [
    ("time", "<i8"),
    ("endtime", "<i8"),
    ("cut_fiducial_volume", "?"),
    ("cut_ambience", "?"),
    ("cut_ar37_kshell_area", "?"),
    ("cut_cs2_area_fraction_top", "?"),
    ("cut_daq_veto", "?"),
    ("cut_interaction_exists", "?"),
    ("cut_main_is_valid_triggering_peak", "?"),
    ("cut_run_boundaries", "?"),
    ("cut_s1_area_fraction_top", "?"),
    ("cut_s1_max_pmt", "?"),
    ("cut_s1_naive_bayes", "?"),
    ("cut_s1_pattern_bottom", "?"),
    ("cut_s1_pattern_top", "?"),
    ("cut_s1_single_scatter", "?"),
    ("cut_s1_tightcoin_3fold", "?"),
    ("cut_s1_width", "?"),
    ("cut_s2_naive_bayes", "?"),
    ("cut_s2_pattern", "?"),
    ("cut_s2_recon_pos_diff", "?"),
    ("cut_s2_single_scatter", "?"),
    ("cut_s2_width", "?"),
    ("cut_position_shadow", "?"),
    ("cut_time_shadow", "?"),
    ("cut_time_veto", "?"),
    ("cut_shadow", "?"),
    ("cuts_ar37_kshell_s1s2", "?"),
    ("cs1", "<f4"),
    ("cs1_wo_timecorr", "<f4"),
    ("cs2_wo_elifecorr", "<f4"),
    ("cs2_wo_timecorr", "<f4"),
    ("cs2_area_fraction_top", "<f4"),
    ("cs2_bottom", "<f4"),
    ("cs2", "<f4"),
    ("alt_cs1", "<f4"),
    ("alt_cs1_wo_timecorr", "<f4"),
    ("alt_cs2_wo_elifecorr", "<f4"),
    ("alt_cs2_wo_timecorr", "<f4"),
    ("alt_cs2_area_fraction_top", "<f4"),
    ("alt_cs2_bottom", "<f4"),
    ("alt_cs2", "<f4"),
    ("e_light", "<f4"),
    ("e_charge", "<f4"),
    ("e_ces", "<f4"),
    ("n_peaks", "<i4"),
    ("drift_time", "<f4"),
    ("event_number", "<i8"),
    ("s1_index", "<i4"),
    ("alt_s1_index", "<i4"),
    ("s1_time", "<i8"),
    ("alt_s1_time", "<i8"),
    ("s1_center_time", "<i8"),
    ("alt_s1_center_time", "<i8"),
    ("s1_endtime", "<i8"),
    ("alt_s1_endtime", "<i8"),
    ("s1_area", "<f4"),
    ("alt_s1_area", "<f4"),
    ("s1_n_channels", "<i2"),
    ("alt_s1_n_channels", "<i2"),
    ("s1_n_hits", "<i2"),
    ("alt_s1_n_hits", "<i2"),
    ("s1_n_competing", "<i4"),
    ("alt_s1_n_competing", "<i4"),
    ("s1_max_pmt", "<i2"),
    ("alt_s1_max_pmt", "<i2"),
    ("s1_max_pmt_area", "<f4"),
    ("alt_s1_max_pmt_area", "<f4"),
    ("s1_range_50p_area", "<f4"),
    ("alt_s1_range_50p_area", "<f4"),
    ("s1_range_90p_area", "<f4"),
    ("alt_s1_range_90p_area", "<f4"),
    ("s1_rise_time", "<f4"),
    ("alt_s1_rise_time", "<f4"),
    ("s1_area_fraction_top", "<f4"),
    ("alt_s1_area_fraction_top", "<f4"),
    ("s1_tight_coincidence", "<i2"),
    ("alt_s1_tight_coincidence", "<i2"),
    ("s1_n_saturated_channels", "<i2"),
    ("alt_s1_n_saturated_channels", "<i2"),
    ("alt_s1_interaction_drift_time", "<f4"),
    ("alt_s1_delay", "<i4"),
    ("s2_index", "<i4"),
    ("alt_s2_index", "<i4"),
    ("s2_time", "<i8"),
    ("alt_s2_time", "<i8"),
    ("s2_center_time", "<i8"),
    ("alt_s2_center_time", "<i8"),
    ("s2_endtime", "<i8"),
    ("alt_s2_endtime", "<i8"),
    ("s2_area", "<f4"),
    ("alt_s2_area", "<f4"),
    ("s2_n_channels", "<i2"),
    ("alt_s2_n_channels", "<i2"),
    ("s2_n_hits", "<i2"),
    ("alt_s2_n_hits", "<i2"),
    ("s2_n_competing", "<i4"),
    ("alt_s2_n_competing", "<i4"),
    ("s2_max_pmt", "<i2"),
    ("alt_s2_max_pmt", "<i2"),
    ("s2_max_pmt_area", "<f4"),
    ("alt_s2_max_pmt_area", "<f4"),
    ("s2_range_50p_area", "<f4"),
    ("alt_s2_range_50p_area", "<f4"),
    ("s2_range_90p_area", "<f4"),
    ("alt_s2_range_90p_area", "<f4"),
    ("s2_rise_time", "<f4"),
    ("alt_s2_rise_time", "<f4"),
    ("s2_area_fraction_top", "<f4"),
    ("alt_s2_area_fraction_top", "<f4"),
    ("s2_tight_coincidence", "<i2"),
    ("alt_s2_tight_coincidence", "<i2"),
    ("s2_n_saturated_channels", "<i2"),
    ("alt_s2_n_saturated_channels", "<i2"),
    ("alt_s2_interaction_drift_time", "<f4"),
    ("alt_s2_delay", "<i4"),
    ("s2_x", "<f4"),
    ("s2_y", "<f4"),
    ("alt_s2_x", "<f4"),
    ("alt_s2_y", "<f4"),
    ("area_before_main_s2", "<f4"),
    ("large_s2_before_main_s2", "<f4"),
    ("s2_x_cnn", "<f4"),
    ("s2_y_cnn", "<f4"),
    ("alt_s2_x_cnn", "<f4"),
    ("alt_s2_y_cnn", "<f4"),
    ("s2_x_gcn", "<f4"),
    ("s2_y_gcn", "<f4"),
    ("alt_s2_x_gcn", "<f4"),
    ("alt_s2_y_gcn", "<f4"),
    ("s2_x_mlp", "<f4"),
    ("s2_y_mlp", "<f4"),
    ("alt_s2_x_mlp", "<f4"),
    ("alt_s2_y_mlp", "<f4"),
    ("x", "<f4"),
    ("alt_s1_x_fdc", "<f4"),
    ("alt_s2_x_fdc", "<f4"),
    ("y", "<f4"),
    ("alt_s1_y_fdc", "<f4"),
    ("alt_s2_y_fdc", "<f4"),
    ("r", "<f4"),
    ("alt_s1_r_fdc", "<f4"),
    ("alt_s2_r_fdc", "<f4"),
    ("z", "<f4"),
    ("alt_s1_z", "<f4"),
    ("alt_s2_z", "<f4"),
    ("r_naive", "<f4"),
    ("alt_s1_r_naive", "<f4"),
    ("alt_s2_r_naive", "<f4"),
    ("z_naive", "<f4"),
    ("alt_s1_z_naive", "<f4"),
    ("alt_s2_z_naive", "<f4"),
    ("r_field_distortion_correction", "<f4"),
    ("alt_s1_r_field_distortion_correction", "<f4"),
    ("alt_s2_r_field_distortion_correction", "<f4"),
    ("z_field_distortion_correction", "<f4"),
    ("alt_s1_z_field_distortion_correction", "<f4"),
    ("alt_s2_z_field_distortion_correction", "<f4"),
    ("alt_s1_theta", "<f4"),
    ("alt_s2_theta", "<f4"),
    ("theta", "<f4"),
]
EVENT_DOUBLE_MERGED_DTYPE = [
    ("time", "<i8"),
    ("endtime", "<i8"),
    ("cut_Kr_DoubleS1_SingleS2", "?"),
    ("cut_fiducial_volume", "?"),
    ("cs1_a", "<f4"),
    ("cs1_a_wo_timecorr", "<f4"),
    ("cs2_a_wo_elifecorr", "<f4"),
    ("cs2_a_wo_timecorr", "<f4"),
    ("cs2_a_area_fraction_top", "<f4"),
    ("cs2_a_bottom", "<f4"),
    ("cs2_a", "<f4"),
    ("cs1_b", "<f4"),
    ("cs1_b_wo_timecorr", "<f4"),
    ("cs2_b_wo_elifecorr", "<f4"),
    ("cs2_b_wo_timecorr", "<f4"),
    ("cs2_b_area_fraction_top", "<f4"),
    ("cs2_b_bottom", "<f4"),
    ("cs2_b", "<f4"),
    ("e_light", "<f4"),
    ("e_charge", "<f4"),
    ("e_ces", "<f4"),
    ("n_peaks", "<i4"),
    ("drift_time", "<f4"),
    ("event_number", "<i8"),
    ("s1_a_index", "<i4"),
    ("s1_b_index", "<i4"),
    ("s1_a_time", "<i8"),
    ("s1_b_time", "<i8"),
    ("s1_a_center_time", "<i8"),
    ("s1_b_center_time", "<i8"),
    ("s1_a_endtime", "<i8"),
    ("s1_b_endtime", "<i8"),
    ("s1_a_area", "<f4"),
    ("s1_b_area", "<f4"),
    ("s1_a_n_channels", "<i2"),
    ("s1_b_n_channels", "<i2"),
    ("s1_a_n_hits", "<i2"),
    ("s1_b_n_hits", "<i2"),
    ("s1_a_n_competing", "<i4"),
    ("s1_b_n_competing", "<i4"),
    ("s1_a_max_pmt", "<i2"),
    ("s1_b_max_pmt", "<i2"),
    ("s1_a_max_pmt_area", "<f4"),
    ("s1_b_max_pmt_area", "<f4"),
    ("s1_a_range_50p_area", "<f4"),
    ("s1_b_range_50p_area", "<f4"),
    ("s1_a_range_90p_area", "<f4"),
    ("s1_b_range_90p_area", "<f4"),
    ("s1_a_rise_time", "<f4"),
    ("s1_b_rise_time", "<f4"),
    ("s1_a_area_fraction_top", "<f4"),
    ("s1_b_area_fraction_top", "<f4"),
    ("s1_a_tight_coincidence", "<i2"),
    ("s1_b_tight_coincidence", "<i2"),
    ("s1_a_n_saturated_channels", "<i2"),
    ("s1_b_n_saturated_channels", "<i2"),
    ("s1_b_interaction_drift_time", "<f4"),
    ("ds_s1_dt", "<i4"),
    ("s2_a_index", "<i4"),
    ("s2_b_index", "<i4"),
    ("s2_a_time", "<i8"),
    ("s2_b_time", "<i8"),
    ("s2_a_center_time", "<i8"),
    ("s2_b_center_time", "<i8"),
    ("s2_a_endtime", "<i8"),
    ("s2_b_endtime", "<i8"),
    ("s2_a_area", "<f4"),
    ("s2_b_area", "<f4"),
    ("s2_a_n_channels", "<i2"),
    ("s2_b_n_channels", "<i2"),
    ("s2_a_n_hits", "<i2"),
    ("s2_b_n_hits", "<i2"),
    ("s2_a_n_competing", "<i4"),
    ("s2_b_n_competing", "<i4"),
    ("s2_a_max_pmt", "<i2"),
    ("s2_b_max_pmt", "<i2"),
    ("s2_a_max_pmt_area", "<f4"),
    ("s2_b_max_pmt_area", "<f4"),
    ("s2_a_range_50p_area", "<f4"),
    ("s2_b_range_50p_area", "<f4"),
    ("s2_a_range_90p_area", "<f4"),
    ("s2_b_range_90p_area", "<f4"),
    ("s2_a_rise_time", "<f4"),
    ("s2_b_rise_time", "<f4"),
    ("s2_a_area_fraction_top", "<f4"),
    ("s2_b_area_fraction_top", "<f4"),
    ("s2_a_tight_coincidence", "<i2"),
    ("s2_b_tight_coincidence", "<i2"),
    ("s2_a_n_saturated_channels", "<i2"),
    ("s2_b_n_saturated_channels", "<i2"),
    ("s2_b_interaction_drift_time", "<f4"),
    ("ds_s2_dt", "<i4"),
    ("s2_a_x", "<f4"),
    ("s2_a_y", "<f4"),
    ("s2_b_x", "<f4"),
    ("s2_b_y", "<f4"),
    ("area_before_main_s2", "<f4"),
    ("large_s2_before_main_s2", "<f4"),
    ("s2_a_x_cnn", "<f4"),
    ("s2_a_y_cnn", "<f4"),
    ("s2_b_x_cnn", "<f4"),
    ("s2_b_y_cnn", "<f4"),
    ("s2_a_x_gcn", "<f4"),
    ("s2_a_y_gcn", "<f4"),
    ("s2_b_x_gcn", "<f4"),
    ("s2_b_y_gcn", "<f4"),
    ("s2_a_x_mlp", "<f4"),
    ("s2_a_y_mlp", "<f4"),
    ("s2_b_x_mlp", "<f4"),
    ("s2_b_y_mlp", "<f4"),
    ("x", "<f4"),
    ("s1_b_x_fdc", "<f4"),
    ("s2_b_x_fdc", "<f4"),
    ("y", "<f4"),
    ("s1_b_y_fdc", "<f4"),
    ("s2_b_y_fdc", "<f4"),
    ("r", "<f4"),
    ("s1_b_r_fdc", "<f4"),
    ("s2_b_r_fdc", "<f4"),
    ("z", "<f4"),
    ("s1_b_z", "<f4"),
    ("s2_b_z", "<f4"),
    ("r_naive", "<f4"),
    ("s1_b_r_naive", "<f4"),
    ("s2_b_r_naive", "<f4"),
    ("z_naive", "<f4"),
    ("s1_b_z_naive", "<f4"),
    ("s2_b_z_naive", "<f4"),
    ("r_field_distortion_correction", "<f4"),
    ("s1_b_r_field_distortion_correction", "<f4"),
    ("s2_b_r_field_distortion_correction", "<f4"),
    ("z_field_distortion_correction", "<f4"),
    ("s1_b_z_field_distortion_correction", "<f4"),
    ("s2_b_z_field_distortion_correction", "<f4"),
    ("s1_b_theta", "<f4"),
    ("s2_b_theta", "<f4"),
    ("theta", "<f4"),
    ("s1_b_distinct_channels", "<i4"),
]
SIM_EVENT_DOUBLE_MERGED_DTYPE = [
    ("time", "<i8"),
    ("endtime", "<i8"),
    ("cut_Kr_DoubleS1_SingleS2", "?"),
    ("cut_fiducial_volume", "?"),
    ("cs1_a", "<f4"),
    ("cs1_a_wo_timecorr", "<f4"),
    ("cs2_a_wo_elifecorr", "<f4"),
    ("cs2_a_wo_timecorr", "<f4"),
    ("cs2_a_area_fraction_top", "<f4"),
    ("cs2_a_bottom", "<f4"),
    ("cs2_a", "<f4"),
    ("cs1_b", "<f4"),
    ("cs1_b_wo_timecorr", "<f4"),
    ("cs2_b_wo_elifecorr", "<f4"),
    ("cs2_b_wo_timecorr", "<f4"),
    ("cs2_b_area_fraction_top", "<f4"),
    ("cs2_b_bottom", "<f4"),
    ("cs2_b", "<f4"),
    ("e_light", "<f4"),
    ("e_charge", "<f4"),
    ("e_ces", "<f4"),
    ("n_peaks", "<i4"),
    ("drift_time", "<f4"),
    ("event_number", "<i8"),
    ("s1_a_index", "<i4"),
    ("s1_b_index", "<i4"),
    ("s1_a_time", "<i8"),
    ("s1_b_time", "<i8"),
    ("s1_a_center_time", "<i8"),
    ("s1_b_center_time", "<i8"),
    ("s1_a_endtime", "<i8"),
    ("s1_b_endtime", "<i8"),
    ("s1_a_area", "<f4"),
    ("s1_b_area", "<f4"),
    ("s1_a_n_channels", "<i2"),
    ("s1_b_n_channels", "<i2"),
    ("s1_a_n_hits", "<i2"),
    ("s1_b_n_hits", "<i2"),
    ("s1_a_n_competing", "<i4"),
    ("s1_b_n_competing", "<i4"),
    ("s1_a_max_pmt", "<i2"),
    ("s1_b_max_pmt", "<i2"),
    ("s1_a_max_pmt_area", "<f4"),
    ("s1_b_max_pmt_area", "<f4"),
    ("s1_a_range_50p_area", "<f4"),
    ("s1_b_range_50p_area", "<f4"),
    ("s1_a_range_90p_area", "<f4"),
    ("s1_b_range_90p_area", "<f4"),
    ("s1_a_rise_time", "<f4"),
    ("s1_b_rise_time", "<f4"),
    ("s1_a_area_fraction_top", "<f4"),
    ("s1_b_area_fraction_top", "<f4"),
    ("s1_a_tight_coincidence", "<i2"),
    ("s1_b_tight_coincidence", "<i2"),
    ("s1_a_n_saturated_channels", "<i2"),
    ("s1_b_n_saturated_channels", "<i2"),
    ("s1_b_interaction_drift_time", "<f4"),
    ("ds_s1_dt", "<i4"),
    ("s2_a_index", "<i4"),
    ("s2_b_index", "<i4"),
    ("s2_a_time", "<i8"),
    ("s2_b_time", "<i8"),
    ("s2_a_center_time", "<i8"),
    ("s2_b_center_time", "<i8"),
    ("s2_a_endtime", "<i8"),
    ("s2_b_endtime", "<i8"),
    ("s2_a_area", "<f4"),
    ("s2_b_area", "<f4"),
    ("s2_a_n_channels", "<i2"),
    ("s2_b_n_channels", "<i2"),
    ("s2_a_n_hits", "<i2"),
    ("s2_b_n_hits", "<i2"),
    ("s2_a_n_competing", "<i4"),
    ("s2_b_n_competing", "<i4"),
    ("s2_a_max_pmt", "<i2"),
    ("s2_b_max_pmt", "<i2"),
    ("s2_a_max_pmt_area", "<f4"),
    ("s2_b_max_pmt_area", "<f4"),
    ("s2_a_range_50p_area", "<f4"),
    ("s2_b_range_50p_area", "<f4"),
    ("s2_a_range_90p_area", "<f4"),
    ("s2_b_range_90p_area", "<f4"),
    ("s2_a_rise_time", "<f4"),
    ("s2_b_rise_time", "<f4"),
    ("s2_a_area_fraction_top", "<f4"),
    ("s2_b_area_fraction_top", "<f4"),
    ("s2_a_tight_coincidence", "<i2"),
    ("s2_b_tight_coincidence", "<i2"),
    ("s2_a_n_saturated_channels", "<i2"),
    ("s2_b_n_saturated_channels", "<i2"),
    ("s2_b_interaction_drift_time", "<f4"),
    ("ds_s2_dt", "<i4"),
    ("s2_a_x", "<f4"),
    ("s2_a_y", "<f4"),
    ("s2_b_x", "<f4"),
    ("s2_b_y", "<f4"),
    ("area_before_main_s2", "<f4"),
    ("large_s2_before_main_s2", "<f4"),
    ("s2_a_x_cnn", "<f4"),
    ("s2_a_y_cnn", "<f4"),
    ("s2_b_x_cnn", "<f4"),
    ("s2_b_y_cnn", "<f4"),
    ("s2_a_x_gcn", "<f4"),
    ("s2_a_y_gcn", "<f4"),
    ("s2_b_x_gcn", "<f4"),
    ("s2_b_y_gcn", "<f4"),
    ("s2_a_x_mlp", "<f4"),
    ("s2_a_y_mlp", "<f4"),
    ("s2_b_x_mlp", "<f4"),
    ("s2_b_y_mlp", "<f4"),
    ("x", "<f4"),
    ("s1_b_x_fdc", "<f4"),
    ("s2_b_x_fdc", "<f4"),
    ("y", "<f4"),
    ("s1_b_y_fdc", "<f4"),
    ("s2_b_y_fdc", "<f4"),
    ("r", "<f4"),
    ("s1_b_r_fdc", "<f4"),
    ("s2_b_r_fdc", "<f4"),
    ("z", "<f4"),
    ("s1_b_z", "<f4"),
    ("s2_b_z", "<f4"),
    ("r_naive", "<f4"),
    ("s1_b_r_naive", "<f4"),
    ("s2_b_r_naive", "<f4"),
    ("z_naive", "<f4"),
    ("s1_b_z_naive", "<f4"),
    ("s2_b_z_naive", "<f4"),
    ("r_field_distortion_correction", "<f4"),
    ("s1_b_r_field_distortion_correction", "<f4"),
    ("s2_b_r_field_distortion_correction", "<f4"),
    ("z_field_distortion_correction", "<f4"),
    ("s1_b_z_field_distortion_correction", "<f4"),
    ("s2_b_z_field_distortion_correction", "<f4"),
    ("s1_b_theta", "<f4"),
    ("s2_b_theta", "<f4"),
    ("theta", "<f4"),
    ("s1_b_distinct_channels", "<i4"),
]

AR_AVAILABLE = np.array(
    [
        "034160",
        "033781",
        "033492",
        "033492",
        "033582",
        "033823",
        "033841",
        "034145",
        "033555",
        "033573",
        "034211",
        "034076",
        "033995",
        "034163",
        "033540",
        "034157",
        "033802",
        "033781",
        "034301",
        "034013",
        "033959",
        "033995",
        "034235",
        "033790",
        "033488",
        "033564",
        "034274",
        "034142",
        "034280",
        "033475",
        "034250",
        "034214",
        "034262",
        "034148",
        "034301",
        "034121",
        "034292",
        "034097",
        "033519",
        "034028",
        "033841",
        "033501",
        "034070",
        "033591",
        "033745",
        "034250",
        "033579",
        "033796",
        "033826",
        "034016",
    ]
)
KR_AVAILABLE = np.array(
    [
        "018223",
        "018834",
        "030532",
        "030430",
        "030403",
        "023392",
        "030406",
        "018902",
        "018913",
        "025633",
        "033226",
        "023555",
        "018767",
        "029509",
        "018614",
        "031903",
        "018253",
        "018568",
        "028701",
        "027016",
        "018653",
        "018929",
        "028665",
        "018777",
        "025633",
        "021731",
        "018630",
        "030505",
        "019188",
        "018844",
        "018617",
        "018722",
        "018503",
        "018578",
        "019240",
        "021725",
        "030355",
        "028656",
        "018485",
        "023479",
        "018759",
        "033256",
        "030484",
        "024345",
        "021530",
        "023395",
        "030448",
        "027039",
        "026419",
        "018364",
    ]
)
SOURCE_RUNS = {"KrS1A": KR_AVAILABLE, "KrS1B": KR_AVAILABLE, "ArS1": AR_AVAILABLE}


@export
class EventsDoubleMerged(strax.MergeOnlyPlugin):
    """Merge event_info_double and cuts."""

    depends_on = (
        "event_info_double",
        "cut_Kr_DoubleS1_SingleS2",
        "cut_fiducial_volume",
    )
    provides = "events_merged"
    __version__ = "0.0.0"
    # save_when = strax.SaveWhen.NEVER


@export
class EventsMerged(strax.MergeOnlyPlugin):
    """Merge event_info and cuts."""

    depends_on = ("event_info", "cuts_ar37_kshell_s1s2", "cut_fiducial_volume")
    provides = "events_merged"
    __version__ = "0.0.0"
    # save_when = strax.SaveWhen.NEVER


@export
@strax.takes_config(
    strax.Option(
        "signal_type",
        type=str,
        default="ArS1",
        help="signal type that can be one of KrS1A, KrS1B, ArS1",
    ),
    strax.Option(
        "fv_cut",
        type=bool,
        default=True,
        help="whether to apply fiducial volume cut or not",
    ),
)
class EventsOfInterest(strax.Plugin):
    """Extract events of interest."""

    depends_on = "events_merged"
    provides = "events_oi"
    __version__ = "0.0.0"
    # save_when = strax.SaveWhen.NEVER

    def infer_dtype(self):
        assert (
            self.config["signal_type"] in SUPPORTED_SIGNALS
        ), "Please try signal type in the supported list: %s" % (SUPPORTED_SIGNALS)
        if (self.config["signal_type"] == "KrS1A") or (
            self.config["signal_type"] == "KrS1B"
        ):
            return EVENT_DOUBLE_MERGED_DTYPE
        elif (self.config["signal_type"] == "sim_KrS1A") or (
            self.config["signal_type"] == "sim_KrS1B"
        ):
            return SIM_EVENT_DOUBLE_MERGED_DTYPE
        else:
            return EVENT_MERGED_DTYPE

    def cut_name(self):
        if self.config["signal_type"] == "ArS1":
            cut = "cuts_ar37_kshell_s1s2"
        # elif self.config['signal_type'] == 'RnS1':
        # cut = 'cuts_rn220'
        elif self.config["signal_type"] == "KrS1A" or "KrS1B":
            cut = "cut_Kr_DoubleS1_SingleS2"
        return cut

    def compute(self, events):
        if self.config["fv_cut"]:
            events_oi = events[events["cut_fiducial_volume"]]
            events_oi = events_oi[events_oi[self.cut_name()]]
        else:
            events_oi = events[events[self.cut_name()]]
        return events_oi


@export
@strax.takes_config(
    strax.Option(
        "signal_type",
        type=str,
        help="signal type that can be one of KrS1A, KrS1B, ArS1",
    ),
    strax.Option("n_tpc_pmts", type=int, help="Number of TPC PMTs"),
)
class PeaksOfInterest(strax.Plugin):
    """Extract peak-level data from events level data."""

    depends_on = ("events_oi", "peaks")
    provides = "peaks_oi"
    __version__ = "0.0.0"
    # save_when = strax.SaveWhen.NEVER

    def infer_dtype(self):
        peaklets_type = strax.peak_dtype(n_channels=self.config["n_tpc_pmts"])
        return peaklets_type

    def locate_peaks(self, events_oi):
        if self.config["signal_type"][-2:] == "S1":
            selection = np.stack(
                (events_oi["s1_time"], events_oi["s1_endtime"]), axis=1
            )
        elif self.config["signal_type"][-2:] == "S2":
            selection = np.stack(
                (events_oi["s2_time"], events_oi["s2_endtime"]), axis=1
            )
        elif self.config["signal_type"][-3:] == "S1A":
            selection = np.stack(
                (events_oi["s1_a_time"], events_oi["s1_a_endtime"]), axis=1
            )
        elif self.config["signal_type"][-3:] == "S1B":
            selection = np.stack(
                (events_oi["s1_b_time"], events_oi["s1_b_endtime"]), axis=1
            )
        return selection

    def compute(self, events, peaks):
        events_oi = events
        intervals = self.locate_peaks(events_oi)
        peaks_oi = np.zeros(len(intervals), dtype=self.infer_dtype())
        for i, t in enumerate(intervals):
            peak = peaks[(peaks["time"] >= t[0]) & (strax.endtime(peaks) <= t[1])]
            peaks_oi[i] = peak
        return peaks_oi


@export
class PeakBasicsOfInterest(PeakBasics):
    """Compute peak basics information based on peaks_oi."""

    __version__ = "0.0.0"
    parallel = True
    provides = "peak_basics_oi"
    depends_on = ("peaks_oi",)
    dtype = PEAK_BASICS_DTYPE
    # save_when = strax.SaveWhen.NEVER

    def compute(self, events):  # named events, but actually are peaks_oi as we desired
        # same as original plugin https://github.com/XENONnT/straxen/blob/86a8a55f3d79d361181196b21ee7ae96e2af2fc4/straxen/plugins/peak_processing.py#L73
        p = events
        r = np.zeros(len(p), self.dtype)
        for q in "time length dt area type".split():
            r[q] = p[q]
        r["endtime"] = p["time"] + p["dt"] * p["length"]
        r["n_channels"] = (p["area_per_channel"] > 0).sum(axis=1)
        r["range_50p_area"] = p["width"][:, 5]
        r["range_90p_area"] = p["width"][:, 9]
        r["max_pmt"] = np.argmax(p["area_per_channel"], axis=1)
        r["max_pmt_area"] = np.max(p["area_per_channel"], axis=1)
        r["tight_coincidence"] = p["tight_coincidence"]
        r["n_saturated_channels"] = p["n_saturated_channels"]

        n_top = self.config["n_top_pmts"]
        area_top = p["area_per_channel"][:, :n_top].sum(axis=1)
        # Recalculate to prevent numerical inaccuracy #442
        area_total = p["area_per_channel"].sum(axis=1)
        # Negative-area peaks get NaN AFT
        m = p["area"] > 0
        r["area_fraction_top"][m] = area_top[m] / area_total[m]
        r["area_fraction_top"][~m] = float("nan")
        r["rise_time"] = -p["area_decile_from_midpoint"][:, 1]

        if self.config["check_peak_sum_area_rtol"] is not None:
            PeakBasics.check_area(
                area_total, p, self.config["check_peak_sum_area_rtol"]
            )
        # Negative or zero-area peaks have centertime at startime
        r["center_time"] = p["time"]
        r["center_time"][m] += PeakBasics.compute_center_times(events[m])
        return r


@export
class PeakExtra(strax.MergeOnlyPlugin):
    """Event level information with peak waveforms."""

    depends_on = ("events_oi", "peaks_oi", "peak_basics_oi")
    provides = "peak_extra"
    __version__ = "0.0.0"
    save_when = strax.SaveWhen.ALWAYS


def get_context(
    signal_type, version="xenonnt_v8", output_folder="/project2/lgrandi/yuanlq/xenonnt/"
):
    """wrapper around context to get peaks of a certain source. Now supporting KrS1A, KrS1B, Ar37.

    Args:
        signal_type (str): signal type that can be one of KrS1A, KrS1B, ArS1
        version (str, optional): cutax version to load events. Defaults to 'xenonnt_v8'.
        output_folder (str, optional): strax data output folder. Defaults to '/project2/lgrandi/yuanlq/xenonnt/'.
    """
    assert (
        signal_type in SUPPORTED_SIGNALS
    ), "Please try signal type in the supported list: %s" % (SUPPORTED_SIGNALS)

    # init context
    context_function = getattr(cutax, version)
    cutax_context = context_function(output_folder=output_folder)

    # whether need to depend on event_info_double
    cutax_context.register_all(cutax.cut_lists.kr83m)
    cutax_context.register_all(cutax.cut_lists.rn220)
    cutax_context.register_all(cutax.cut_lists.ar37)
    cutax_context.register_all(cutax.cut_lists.basic)
    if (signal_type[-1] == "A") or (signal_type[-1] == "B"):
        cutax_context.register(EventsDoubleMerged)
    else:
        cutax_context.register(EventsMerged)
    cutax_context.register(EventsOfInterest)
    cutax_context.register(PeaksOfInterest)
    cutax_context.register(PeakBasicsOfInterest)
    cutax_context.register(PeakExtra)

    return cutax_context


def get_peaks(
    runs,
    signal_type,
    version="xenonnt_v8",
    output_folder="/project2/lgrandi/yuanlq/xenonnt/",
    **kargs
):
    """wrapper around get_array to get peaks of a certain source. Now supporting KrS1A, KrS1B, Ar37.

    Args:
        runs (str or 1darray): runs to extract certain signal. Assumed all type of runs if provided array.
        signal_type (str): signal type that can be one of KrS1A, KrS1B, ArS1
        version (str, optional): cutax version to load events. Defaults to 'xenonnt_v8'.
        output_folder (str, optional): strax data output folder. Defaults to '/project2/lgrandi/yuanlq/xenonnt/'.
    """
    cutax_context = get_context(
        signal_type=signal_type, version=version, output_folder=output_folder
    )
    result = cutax_context.get_array(
        runs, "peaks_oi", config=dict(signal_type=signal_type, **kargs)
    )
    return result


def get_peak_basics(
    runs,
    signal_type,
    version="xenonnt_v8",
    output_folder="/project2/lgrandi/yuanlq/xenonnt/",
    **kargs
):
    """wrapper around get_array to get peak_basics of a certain source. Now supporting KrS1A, KrS1B, Ar37.

    Args:
        runs (str or 1darray): runs to extract certain signal. Assumed all type of runs if provided array.
        signal_type (str): signal type that can be one of KrS1A, KrS1B, ArS1
        version (str, optional): cutax version to load events. Defaults to 'xenonnt_v8'.
        output_folder (str, optional): strax data output folder. Defaults to '/project2/lgrandi/yuanlq/xenonnt/'.
    """
    cutax_context = get_context(
        signal_type=signal_type, version=version, output_folder=output_folder
    )
    result = cutax_context.get_array(
        runs, "peak_basics_oi", config=dict(signal_type=signal_type, **kargs)
    )
    return result


def get_peak_extra_from_events(
    signal_type,
    runs=False,
    version="xenonnt_v8",
    output_folder="/project2/lgrandi/yuanlq/xenonnt/",
    interaction_type=11,
    energy=41.5,
    N=10000,
    fv_cut=True,
    **kargs
):
    """wrapper around get_array to get peak_extra of a certain source from events. Now supporting KrS1A, KrS1B, Ar37.

    Args:
        signal_type (str): signal type that can be one of KrS1A, KrS1B, ArS1.
        runs (str or 1darray, optional): runs to extract certain signal. Assumed all type of runs if provided array. If not specified, will use runs in SOURCE_RUNS
        version (str, optional): cutax version to load events. Defaults to 'xenonnt_v8'.
        output_folder (str, optional): strax data output folder. Defaults to '/project2/lgrandi/yuanlq/xenonnt/'.
        interaction_type (int): Following the NEST type of intereaction. Only used in simulation case.
                energy (float or list): energy deposit in unit of keV. Only used in simulation case.
                N (int, optional): simulation number. Defaults to 10000. Only used in simulation case.
        fv_cut (bool, optional): whether to include fiducial volume cut.
    """
    if signal_type[:6] == "sim_Kr":
        cutax_context = simwrap.get_sim_context(
            interaction_type=interaction_type,
            energy=energy,
            N=N,
            version=version,
            **kargs
        )
        cutax_context.register(EventsDoubleMerged)
        cutax_context.register(EventsOfInterest)
        cutax_context.register(PeaksOfInterest)
        cutax_context.register(PeakBasicsOfInterest)
        cutax_context.register(PeakExtra)
    else:
        cutax_context = get_context(
            signal_type=signal_type, version=version, output_folder=output_folder
        )
    if not (type(runs) == bool and runs == False):
        result = cutax_context.get_array(
            runs,
            "peak_extra",
            config=dict(signal_type=signal_type, fv_cut=fv_cut, **kargs),
        )
    else:
        result = cutax_context.get_array(
            SOURCE_RUNS[signal_type],
            "peak_extra",
            config=dict(signal_type=signal_type, fv_cut=fv_cut, **kargs),
        )

    return result
