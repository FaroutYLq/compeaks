import numpy as np
import strax
import cutax
from straxen import pre_apply_function
from straxen.plugins.peak_processing import PeakBasics
export, __all__ = strax.exporter()

SUPPORTED_SIGNALS = ['KrS1A', 'KrS1B', 'ArS1']
PEAK_BASICS_DTYPE = [(('Start time of the peak (ns since unix epoch)', 'time'), numpy.int64),(('End time of the peak (ns since unix epoch)', 'endtime'), numpy.int64),(('Weighted center time of the peak (ns since unix epoch)', 'center_time'),numpy.int64),(('Peak integral in PE', 'area'), numpy.float32),(('Number of PMTs contributing to the peak', 'n_channels'), numpy.int16),(('PMT number which contributes the most PE', 'max_pmt'), numpy.int16),(('Area of signal in the largest-contributing PMT (PE)', 'max_pmt_area'),numpy.float32),(('Total number of saturated channels', 'n_saturated_channels'), numpy.int16),(('Width (in ns) of the central 50% area of the peak', 'range_50p_area'),numpy.float32),(('Width (in ns) of the central 90% area of the peak', 'range_90p_area'),numpy.float32),(('Fraction of area seen by the top array (NaN for peaks with non-positive area)','area_fraction_top'),numpy.float32),(('Length of the peak waveform in samples', 'length'), numpy.int32),(('Time resolution of the peak waveform in ns', 'dt'), numpy.int16),(('Time between 10% and 50% area quantiles [ns]', 'rise_time'),numpy.float32),(('Hits within tight range of mean', 'tight_coincidence'), numpy.int16),(('PMT channel within tight range of mean', 'tight_coincidence_channel'),numpy.int16),(('Classification of the peak(let)', 'type'), numpy.int8)]
EVENT_MERGED_DTYPE = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Fiducial volume cut based on z versus r.', 'cut_fiducial_volume'), '?'), (('S1 and S2 area selection for Ar37 decay at 2.82 keV.', 'cut_ar37_kshell_area'), '?'), (('cS2 area fraction top(AFT) cut constraining the spread of cS2AFT.', 'cut_cs2_area_fraction_top'), '?'), (('S1 max pmt cut based on s1_max_pmt_area vs s1_area', 'cut_s1_max_pmt'), '?'), (('S2 reconstruction position difference cut based on 99% upper limit(MLP,GCN,CNN)', 'cut_s2_recon_pos_diff'), '?'), (('Single scatter cut based on s2_area versus alt_s2_area.', 'cut_s2_single_scatter'), '?'), (('S2 Width cut with special treatment near the wires', 'cut_s2_width'), '?'), (('Accumulated boolean over all cuts.', 'cuts_ar37_kshell_s1s2'), '?'), (('Corrected area of main S1 [PE]', 'cs1'), '<f4'), (('Corrected area of main S2 before elife correction (s2 xy correction + SEG/EE correction applied) [PE]', 'cs2_wo_elifecorr'), '<f4'), (('Corrected area of main S2 before SEG/EE and elife corrections(s2 xy correction applied) [PE]', 'cs2_wo_timecorr'), '<f4'), (('Fraction of area seen by the top PMT array for corrected main S2', 'cs2_area_fraction_top'), '<f4'), (('Corrected area of main S2 in the bottom PMT array [PE]', 'cs2_bottom'), '<f4'), (('Corrected area of main S2 [PE]', 'cs2'), '<f4'), (('Corrected area of alternate S1 [PE]', 'alt_cs1'), '<f4'), (('Corrected area of alternate S2 before elife correction (s2 xy correction + SEG/EE correction applied) [PE]', 'alt_cs2_wo_elifecorr'), '<f4'), (('Corrected area of alternate S2 before SEG/EE and elife corrections(s2 xy correction applied) [PE]', 'alt_cs2_wo_timecorr'), '<f4'), (('Fraction of area seen by the top PMT array for corrected alternate S2', 'alt_cs2_area_fraction_top'), '<f4'), (('Corrected area of alternate S2 in the bottom PMT array [PE]', 'alt_cs2_bottom'), '<f4'), (('Corrected area of alternate S2 [PE]', 'alt_cs2'), '<f4'), (('Energy in light signal [keVee]', 'e_light'), '<f4'), (('Energy in charge signal [keVee]', 'e_charge'), '<f4'), (('Energy estimate [keVee]', 'e_ces'), '<f4'), (('Number of peaks in the event', 'n_peaks'), '<i4'), (('Drift time between main S1 and S2 in ns', 'drift_time'), '<f4'), (('Event number in this dataset', 'event_number'), '<i8'), (('Main S1 peak index in event', 's1_index'), '<i4'), (('Alternate S1 peak index in event', 'alt_s1_index'), '<i4'), (('Main S1 start time since unix epoch [ns]', 's1_time'), '<i8'), (('Alternate S1 start time since unix epoch [ns]', 'alt_s1_time'), '<i8'), (('Main S1 weighted center time since unix epoch [ns]', 's1_center_time'), '<i8'), (('Alternate S1 weighted center time since unix epoch [ns]', 'alt_s1_center_time'), '<i8'), (('Main S1 end time since unix epoch [ns]', 's1_endtime'), '<i8'), (('Alternate S1 end time since unix epoch [ns]', 'alt_s1_endtime'), '<i8'), (('Main S1 area, uncorrected [PE]', 's1_area'), '<f4'), (('Alternate S1 area, uncorrected [PE]', 'alt_s1_area'), '<f4'), (('Main S1 count of contributing PMTs', 's1_n_channels'), '<i2'), (('Alternate S1 count of contributing PMTs', 'alt_s1_n_channels'), '<i2'), (('Main S1 number of competing peaks', 's1_n_competing'), '<i4'), (('Alternate S1 number of competing peaks', 'alt_s1_n_competing'), '<i4'), (('Main S1 PMT number which contributes the most PE', 's1_max_pmt'), '<i2'), (('Alternate S1 PMT number which contributes the most PE', 'alt_s1_max_pmt'), '<i2'), (('Main S1 area in the largest-contributing PMT (PE)', 's1_max_pmt_area'), '<f4'), (('Alternate S1 area in the largest-contributing PMT (PE)', 'alt_s1_max_pmt_area'), '<f4'), (('Main S1 width, 50% area [ns]', 's1_range_50p_area'), '<f4'), (('Alternate S1 width, 50% area [ns]', 'alt_s1_range_50p_area'), '<f4'), (('Main S1 width, 90% area [ns]', 's1_range_90p_area'), '<f4'), (('Alternate S1 width, 90% area [ns]', 'alt_s1_range_90p_area'), '<f4'), (('Main S1 time between 10% and 50% area quantiles [ns]', 's1_rise_time'), '<f4'), (('Alternate S1 time between 10% and 50% area quantiles [ns]', 'alt_s1_rise_time'), '<f4'), (('Main S1 fraction of area seen by the top PMT array', 's1_area_fraction_top'), '<f4'), (('Alternate S1 fraction of area seen by the top PMT array', 'alt_s1_area_fraction_top'), '<f4'), (('Main S1 Channel within tight range of mean', 's1_tight_coincidence'), '<i2'), (('Alternate S1 Channel within tight range of mean', 'alt_s1_tight_coincidence'), '<i2'), (('Main S1 Total number of saturated channels', 's1_n_saturated_channels'), '<i2'), (('Alternate S1 Total number of saturated channels', 'alt_s1_n_saturated_channels'), '<i2'), (('Drift time using alternate S1 [ns]', 'alt_s1_interaction_drift_time'), '<f4'), (('Time between main and alternate S1 [ns]', 'alt_s1_delay'), '<i4'), (('Main S2 peak index in event', 's2_index'), '<i4'), (('Alternate S2 peak index in event', 'alt_s2_index'), '<i4'), (('Main S2 start time since unix epoch [ns]', 's2_time'), '<i8'), (('Alternate S2 start time since unix epoch [ns]', 'alt_s2_time'), '<i8'), (('Main S2 weighted center time since unix epoch [ns]', 's2_center_time'), '<i8'), (('Alternate S2 weighted center time since unix epoch [ns]', 'alt_s2_center_time'), '<i8'), (('Main S2 end time since unix epoch [ns]', 's2_endtime'), '<i8'), (('Alternate S2 end time since unix epoch [ns]', 'alt_s2_endtime'), '<i8'), (('Main S2 area, uncorrected [PE]', 's2_area'), '<f4'), (('Alternate S2 area, uncorrected [PE]', 'alt_s2_area'), '<f4'), (('Main S2 count of contributing PMTs', 's2_n_channels'), '<i2'), (('Alternate S2 count of contributing PMTs', 'alt_s2_n_channels'), '<i2'), (('Main S2 number of competing peaks', 's2_n_competing'), '<i4'), (('Alternate S2 number of competing peaks', 'alt_s2_n_competing'), '<i4'), (('Main S2 PMT number which contributes the most PE', 's2_max_pmt'), '<i2'), (('Alternate S2 PMT number which contributes the most PE', 'alt_s2_max_pmt'), '<i2'), (('Main S2 area in the largest-contributing PMT (PE)', 's2_max_pmt_area'), '<f4'), (('Alternate S2 area in the largest-contributing PMT (PE)', 'alt_s2_max_pmt_area'), '<f4'), (('Main S2 width, 50% area [ns]', 's2_range_50p_area'), '<f4'), (('Alternate S2 width, 50% area [ns]', 'alt_s2_range_50p_area'), '<f4'), (('Main S2 width, 90% area [ns]', 's2_range_90p_area'), '<f4'), (('Alternate S2 width, 90% area [ns]', 'alt_s2_range_90p_area'), '<f4'), (('Main S2 time between 10% and 50% area quantiles [ns]', 's2_rise_time'), '<f4'), (('Alternate S2 time between 10% and 50% area quantiles [ns]', 'alt_s2_rise_time'), '<f4'), (('Main S2 fraction of area seen by the top PMT array', 's2_area_fraction_top'), '<f4'), (('Alternate S2 fraction of area seen by the top PMT array', 'alt_s2_area_fraction_top'), '<f4'), (('Main S2 Channel within tight range of mean', 's2_tight_coincidence'), '<i2'), (('Alternate S2 Channel within tight range of mean', 'alt_s2_tight_coincidence'), '<i2'), (('Main S2 Total number of saturated channels', 's2_n_saturated_channels'), '<i2'), (('Alternate S2 Total number of saturated channels', 'alt_s2_n_saturated_channels'), '<i2'), (('Drift time using alternate S2 [ns]', 'alt_s2_interaction_drift_time'), '<f4'), (('Time between main and alternate S2 [ns]', 'alt_s2_delay'), '<i4'), (('Main S2 reconstructed X position, uncorrected [cm]', 's2_x'), '<f4'), (('Main S2 reconstructed Y position, uncorrected [cm]', 's2_y'), '<f4'), (('Alternate S2 reconstructed X position, uncorrected [cm]', 'alt_s2_x'), '<f4'), (('Alternate S2 reconstructed Y position, uncorrected [cm]', 'alt_s2_y'), '<f4'), (('Sum of areas before Main S2 [PE]', 'area_before_main_s2'), '<f4'), (('The largest S2 before the Main S2 [PE]', 'large_s2_before_main_s2'), '<f4'), (('Main S2 cnn-reconstructed X position, uncorrected [cm]', 's2_x_cnn'), '<f4'), (('Main S2 cnn-reconstructed Y position, uncorrected [cm]', 's2_y_cnn'), '<f4'), (('Alternate S2 cnn-reconstructed X position, uncorrected [cm]', 'alt_s2_x_cnn'), '<f4'), (('Alternate S2 cnn-reconstructed Y position, uncorrected [cm]', 'alt_s2_y_cnn'), '<f4'), (('Main S2 gcn-reconstructed X position, uncorrected [cm]', 's2_x_gcn'), '<f4'), (('Main S2 gcn-reconstructed Y position, uncorrected [cm]', 's2_y_gcn'), '<f4'), (('Alternate S2 gcn-reconstructed X position, uncorrected [cm]', 'alt_s2_x_gcn'), '<f4'), (('Alternate S2 gcn-reconstructed Y position, uncorrected [cm]', 'alt_s2_y_gcn'), '<f4'), (('Main S2 mlp-reconstructed X position, uncorrected [cm]', 's2_x_mlp'), '<f4'), (('Main S2 mlp-reconstructed Y position, uncorrected [cm]', 's2_y_mlp'), '<f4'), (('Alternate S2 mlp-reconstructed X position, uncorrected [cm]', 'alt_s2_x_mlp'), '<f4'), (('Alternate S2 mlp-reconstructed Y position, uncorrected [cm]', 'alt_s2_y_mlp'), '<f4'), (('Interaction x-position, field-distortion corrected (cm)', 'x'), '<f4'), (('Interaction y-position, field-distortion corrected (cm)', 'y'), '<f4'), (('Interaction z-position, using mean drift velocity only (cm)', 'z'), '<f4'), (('Interaction radial position, field-distortion corrected (cm)', 'r'), '<f4'), (('Interaction z-position using mean drift velocity only (cm)', 'z_naive'), '<f4'), (('Interaction r-position using observed S2 positions directly (cm)', 'r_naive'), '<f4'), (('Correction added to r_naive for field distortion (cm)', 'r_field_distortion_correction'), '<f4'), (('Correction added to z_naive for field distortion (cm)', 'z_field_distortion_correction'), '<f4'), (('Interaction angular position (radians)', 'theta'), '<f4')]
EVENT_DOUBLE_MERGED_DTYPE = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Kr Double S1 Single S2 event topology cut', 'cut_Kr_DoubleS1_SingleS2'), '?'), (('Fiducial volume cut based on z versus r.', 'cut_fiducial_volume'), '?'), (('Corrected area of main S1 [PE]', 'cs1_a'), '<f4'), (('Corrected area of main S2 before elife correction (s2 xy correction + SEG/EE correction applied) [PE]', 'cs2_a_wo_elifecorr'), '<f4'), (('Corrected area of main S2 before SEG/EE and elife corrections(s2 xy correction applied) [PE]', 'cs2_wo_timecorr'), '<f4'), (('Fraction of area seen by the top PMT array for corrected main S2', 'cs2_a_area_fraction_top'), '<f4'), (('Corrected area of main S2 in the bottom PMT array [PE]', 'cs2_a_bottom'), '<f4'), (('Corrected area of main S2 [PE]', 'cs2_a'), '<f4'), (('Corrected area of alternate S1 [PE]', 'cs1_b'), '<f4'), (('Corrected area of alternate S2 before elife correction (s2 xy correction + SEG/EE correction applied) [PE]', 'cs2_b_wo_elifecorr'), '<f4'), (('Corrected area of alternate S2 before SEG/EE and elife corrections(s2 xy correction applied) [PE]', 'alt_cs2_wo_timecorr'), '<f4'), (('Fraction of area seen by the top PMT array for corrected alternate S2', 'cs2_b_area_fraction_top'), '<f4'), (('Corrected area of alternate S2 in the bottom PMT array [PE]', 'cs2_b_bottom'), '<f4'), (('Corrected area of alternate S2 [PE]', 'cs2_b'), '<f4'), (('Energy in light signal [keVee]', 'e_light'), '<f4'), (('Energy in charge signal [keVee]', 'e_charge'), '<f4'), (('Energy estimate [keVee]', 'e_ces'), '<f4'), (('Number of peaks in the event', 'n_peaks'), '<i4'), (('Drift time between main S1 and S2 in ns', 'drift_time'), '<f4'), (('Event number in this dataset', 'event_number'), '<i8'), (('Main S1 peak index in event', 's1_a_index'), '<i4'), (('Alternate S1 peak index in event', 's1_b_index'), '<i4'), (('Main S1 start time since unix epoch [ns]', 's1_a_time'), '<i8'), (('Alternate S1 start time since unix epoch [ns]', 's1_b_time'), '<i8'), (('Main S1 weighted center time since unix epoch [ns]', 's1_a_center_time'), '<i8'), (('Alternate S1 weighted center time since unix epoch [ns]', 's1_b_center_time'), '<i8'), (('Main S1 end time since unix epoch [ns]', 's1_a_endtime'), '<i8'), (('Alternate S1 end time since unix epoch [ns]', 's1_b_endtime'), '<i8'), (('Main S1 area, uncorrected [PE]', 's1_a_area'), '<f4'), (('Alternate S1 area, uncorrected [PE]', 's1_b_area'), '<f4'), (('Main S1 count of contributing PMTs', 's1_a_n_channels'), '<i2'), (('Alternate S1 count of contributing PMTs', 's1_b_n_channels'), '<i2'), (('Main S1 number of competing peaks', 's1_a_n_competing'), '<i4'), (('Alternate S1 number of competing peaks', 's1_b_n_competing'), '<i4'), (('Main S1 PMT number which contributes the most PE', 's1_a_max_pmt'), '<i2'), (('Alternate S1 PMT number which contributes the most PE', 's1_b_max_pmt'), '<i2'), (('Main S1 area in the largest-contributing PMT (PE)', 's1_a_max_pmt_area'), '<f4'), (('Alternate S1 area in the largest-contributing PMT (PE)', 's1_b_max_pmt_area'), '<f4'), (('Main S1 width, 50% area [ns]', 's1_a_range_50p_area'), '<f4'), (('Alternate S1 width, 50% area [ns]', 's1_b_range_50p_area'), '<f4'), (('Main S1 width, 90% area [ns]', 's1_a_range_90p_area'), '<f4'), (('Alternate S1 width, 90% area [ns]', 's1_b_range_90p_area'), '<f4'), (('Main S1 time between 10% and 50% area quantiles [ns]', 's1_a_rise_time'), '<f4'), (('Alternate S1 time between 10% and 50% area quantiles [ns]', 's1_b_rise_time'), '<f4'), (('Main S1 fraction of area seen by the top PMT array', 's1_a_area_fraction_top'), '<f4'), (('Alternate S1 fraction of area seen by the top PMT array', 's1_b_area_fraction_top'), '<f4'), (('Main S1 Channel within tight range of mean', 's1_a_tight_coincidence'), '<i2'), (('Alternate S1 Channel within tight range of mean', 's1_b_tight_coincidence'), '<i2'), (('Main S1 Total number of saturated channels', 's1_a_n_saturated_channels'), '<i2'), (('Alternate S1 Total number of saturated channels', 's1_b_n_saturated_channels'), '<i2'), (('Drift time using alternate S1 [ns]', 's1_b_interaction_drift_time'), '<f4'), (('Time between main and alternate S1 [ns]', 'ds_s1_dt'), '<i4'), (('Main S2 peak index in event', 's2_a_index'), '<i4'), (('Alternate S2 peak index in event', 's2_b_index'), '<i4'), (('Main S2 start time since unix epoch [ns]', 's2_a_time'), '<i8'), (('Alternate S2 start time since unix epoch [ns]', 's2_b_time'), '<i8'), (('Main S2 weighted center time since unix epoch [ns]', 's2_a_center_time'), '<i8'), (('Alternate S2 weighted center time since unix epoch [ns]', 's2_b_center_time'), '<i8'), (('Main S2 end time since unix epoch [ns]', 's2_a_endtime'), '<i8'), (('Alternate S2 end time since unix epoch [ns]', 's2_b_endtime'), '<i8'), (('Main S2 area, uncorrected [PE]', 's2_a_area'), '<f4'), (('Alternate S2 area, uncorrected [PE]', 's2_b_area'), '<f4'), (('Main S2 count of contributing PMTs', 's2_a_n_channels'), '<i2'), (('Alternate S2 count of contributing PMTs', 's2_b_n_channels'), '<i2'), (('Main S2 number of competing peaks', 's2_a_n_competing'), '<i4'), (('Alternate S2 number of competing peaks', 's2_b_n_competing'), '<i4'), (('Main S2 PMT number which contributes the most PE', 's2_a_max_pmt'), '<i2'), (('Alternate S2 PMT number which contributes the most PE', 's2_b_max_pmt'), '<i2'), (('Main S2 area in the largest-contributing PMT (PE)', 's2_a_max_pmt_area'), '<f4'), (('Alternate S2 area in the largest-contributing PMT (PE)', 's2_b_max_pmt_area'), '<f4'), (('Main S2 width, 50% area [ns]', 's2_a_range_50p_area'), '<f4'), (('Alternate S2 width, 50% area [ns]', 's2_b_range_50p_area'), '<f4'), (('Main S2 width, 90% area [ns]', 's2_a_range_90p_area'), '<f4'), (('Alternate S2 width, 90% area [ns]', 's2_b_range_90p_area'), '<f4'), (('Main S2 time between 10% and 50% area quantiles [ns]', 's2_a_rise_time'), '<f4'), (('Alternate S2 time between 10% and 50% area quantiles [ns]', 's2_b_rise_time'), '<f4'), (('Main S2 fraction of area seen by the top PMT array', 's2_a_area_fraction_top'), '<f4'), (('Alternate S2 fraction of area seen by the top PMT array', 's2_b_area_fraction_top'), '<f4'), (('Main S2 Channel within tight range of mean', 's2_a_tight_coincidence'), '<i2'), (('Alternate S2 Channel within tight range of mean', 's2_b_tight_coincidence'), '<i2'), (('Main S2 Total number of saturated channels', 's2_a_n_saturated_channels'), '<i2'), (('Alternate S2 Total number of saturated channels', 's2_b_n_saturated_channels'), '<i2'), (('Drift time using alternate S2 [ns]', 's2_b_interaction_drift_time'), '<f4'), (('Time between main and alternate S2 [ns]', 'ds_s2_dt'), '<i4'), (('Main S2 reconstructed X position, uncorrected [cm]', 's2_a_x'), '<f4'), (('Main S2 reconstructed Y position, uncorrected [cm]', 's2_a_y'), '<f4'), (('Alternate S2 reconstructed X position, uncorrected [cm]', 's2_b_x'), '<f4'), (('Alternate S2 reconstructed Y position, uncorrected [cm]', 's2_b_y'), '<f4'), (('Sum of areas before Main S2 [PE]', 'area_before_main_s2'), '<f4'), (('The largest S2 before the Main S2 [PE]', 'large_s2_before_main_s2'), '<f4'), (('Main S2 cnn-reconstructed X position, uncorrected [cm]', 's2_a_x_cnn'), '<f4'), (('Main S2 cnn-reconstructed Y position, uncorrected [cm]', 's2_a_y_cnn'), '<f4'), (('Alternate S2 cnn-reconstructed X position, uncorrected [cm]', 's2_b_x_cnn'), '<f4'), (('Alternate S2 cnn-reconstructed Y position, uncorrected [cm]', 's2_b_y_cnn'), '<f4'), (('Main S2 gcn-reconstructed X position, uncorrected [cm]', 's2_a_x_gcn'), '<f4'), (('Main S2 gcn-reconstructed Y position, uncorrected [cm]', 's2_a_y_gcn'), '<f4'), (('Alternate S2 gcn-reconstructed X position, uncorrected [cm]', 's2_b_x_gcn'), '<f4'), (('Alternate S2 gcn-reconstructed Y position, uncorrected [cm]', 's2_b_y_gcn'), '<f4'), (('Main S2 mlp-reconstructed X position, uncorrected [cm]', 's2_a_x_mlp'), '<f4'), (('Main S2 mlp-reconstructed Y position, uncorrected [cm]', 's2_a_y_mlp'), '<f4'), (('Alternate S2 mlp-reconstructed X position, uncorrected [cm]', 's2_b_x_mlp'), '<f4'), (('Alternate S2 mlp-reconstructed Y position, uncorrected [cm]', 's2_b_y_mlp'), '<f4'), (('Interaction x-position, field-distortion corrected (cm)', 'x'), '<f4'), (('Interaction y-position, field-distortion corrected (cm)', 'y'), '<f4'), (('Interaction z-position, using mean drift velocity only (cm)', 'z'), '<f4'), (('Interaction radial position, field-distortion corrected (cm)', 'r'), '<f4'), (('Interaction z-position using mean drift velocity only (cm)', 'z_naive'), '<f4'), (('Interaction r-position using observed S2 positions directly (cm)', 'r_naive'), '<f4'), (('Correction added to r_naive for field distortion (cm)', 'r_field_distortion_correction'), '<f4'), (('Correction added to z_naive for field distortion (cm)', 'z_field_distortion_correction'), '<f4'), (('Interaction angular position (radians)', 'theta'), '<f4'), (('Number of PMTs contributing to the secondary S1 that do not contribute to the main S1', 's1_b_distinct_channels'), '<i4')]


@export
class EventsDoubleMerged(strax.MergeOnlyPlugin):
    """Merge event_info_double and cuts.
    """
    depends_on = ('event_info_double', 
                  'cut_Kr_DoubleS1_SingleS2', 
                  'cut_fiducial_volume')
    provides = 'events_merged'
    __version__ = '0.0.0'
    save_when = strax.SaveWhen.NEVER


@export
class EventsMerged(strax.MergeOnlyPlugin):
    """Merge event_info and cuts.
    """
    depends_on = ('event_info', 
                  'cuts_ar37_kshell_s1s2', 
                  'cut_fiducial_volume')
    provides = 'events_merged'
    __version__ = '0.0.0'
    save_when = strax.SaveWhen.NEVER


@export
@strax.takes_config(
    strax.Option('signal_type', type=str, default='ArS1',
                 help='signal type that can be one of KrS1A, KrS1B, ArS1'),
    strax.Option('fv_cut', type=bool, default=True,
                 help='whether to apply fiducial volume cut or not'),
)
class EventsOfInterest(strax.Plugin):
    """Extract events of interest.
    """
    depends_on = ('events_merged')
    provides = 'events_oi'
    __version__ = '0.0.0'
    save_when = strax.SaveWhen.NEVER

    def infer_dtype(self):
        assert self.config['signal_type'] in SUPPORTED_SIGNALS, "Please try signal type in the supported list: %s"%(SUPPORTED_SIGNALS)
        if self.config['signal_type'][:-1] == 'A' or 'B':
            return EVENT_DOUBLE_MERGED_DTYPE 
        else:
            return EVENT_MERTED_DTYPE

    def cut_name(self):
        if self.config['signal_type'] == 'ArS1':
            cut = 'cuts_ar37_kshell_s1s2'
        #elif self.config['signal_type'] == 'RnS1':
            #cut = 'cuts_rn220'
        elif self.config['signal_type'] == 'KrS1A' or 'KrS1B':
            cut = 'cut_Kr_DoubleS1_SingleS2'
        return cut

    def compute(self, events):
        if self.config['fv_cut']:
            events_oi = events[events['cut_fiducial_volume']]
        events_oi = events_oi[events_oi[self.cut_name()]]
        return events_oi


@export
@strax.takes_config(
    strax.Option('signal_type', type=str,
                 help='signal type that can be one of KrS1A, KrS1B, ArS1'),
    strax.Option('n_tpc_pmts', type=int,
                 help='Number of TPC PMTs'),
)
class PeaksOfInterest(strax.Plugin):
    """Extract peak-level data from events level data.
    """
    depends_on = ('events_oi', 'peaks')
    provides = 'peaks_oi'
    __version__ = '0.0.0'
    save_when = strax.SaveWhen.NEVER


    def infer_dtype(self):
        peaklets_type=strax.peak_dtype(n_channels=self.config['n_tpc_pmts'])
        return peaklets_type

    def locate_peaks(self, events_oi):
        if self.config['signal_type'][-2:] == 'S1':
            selection = np.stack((events_oi['s1_time'], events_oi['s1_endtime']), axis=1)
        elif self.config['signal_type'][-2:] == 'S2':
            selection = np.stack((events_oi['s2_time'], events_oi['s2_endtime']), axis=1)
        elif self.config['signal_type'][-3:] == 'S1A':
            selection = np.stack((events_oi['s1_a_time'], events_oi['s1_a_endtime']), axis=1)
        elif self.config['signal_type'][-3:] == 'S1B':
            selection = np.stack((events_oi['s1_b_time'], events_oi['s1_b_endtime']), axis=1)           
        return selection

    def compute(self, events, peaks):
        events_oi = events
        intervals = self.locate_peaks(events_oi)
        peaks_oi = np.zeros(len(intervals), dtype=self.infer_dtype())
        for i, t in enumerate(intervals):
            peak = peaks[(peaks['time']>=t[0]) & (strax.endtime(peaks)<=t[1])]
            peaks_oi[i] = peak
        return peaks_oi            
        

@export
class PeakBasicsOfInterest(PeakBasics):
    """Compute peak basics information based on peaks_oi.
    """
    __version__ = '0.0.0'
    parallel = True
    provides = ('peak_basics_oi')
    depends_on = ('peaks_oi',)
    dtype = PEAK_BASICS_DTYPE
    save_when = strax.SaveWhen.NEVER

    def compute(self, events): # named events, but actually are peaks_oi as we desired
        # same as original plugin https://github.com/XENONnT/straxen/blob/86a8a55f3d79d361181196b21ee7ae96e2af2fc4/straxen/plugins/peak_processing.py#L73
        p = events
        r = np.zeros(len(p), self.dtype)
        for q in 'time length dt area type'.split():
            r[q] = p[q]
        r['endtime'] = p['time'] + p['dt'] * p['length']
        r['n_channels'] = (p['area_per_channel'] > 0).sum(axis=1)
        r['range_50p_area'] = p['width'][:, 5]
        r['range_90p_area'] = p['width'][:, 9]
        r['max_pmt'] = np.argmax(p['area_per_channel'], axis=1)
        r['max_pmt_area'] = np.max(p['area_per_channel'], axis=1)
        r['tight_coincidence'] = p['tight_coincidence']
        r['n_saturated_channels'] = p['n_saturated_channels']

        n_top = self.config['n_top_pmts']
        area_top = p['area_per_channel'][:, :n_top].sum(axis=1)
        # Recalculate to prevent numerical inaccuracy #442
        area_total = p['area_per_channel'].sum(axis=1)
        # Negative-area peaks get NaN AFT
        m = p['area'] > 0
        r['area_fraction_top'][m] = area_top[m]/area_total[m]
        r['area_fraction_top'][~m] = float('nan')
        r['rise_time'] = -p['area_decile_from_midpoint'][:, 1]

        if self.config['check_peak_sum_area_rtol'] is not None:
            PeakBasics.check_area(area_total, p, self.config['check_peak_sum_area_rtol'])
        # Negative or zero-area peaks have centertime at startime
        r['center_time'] = p['time']
        r['center_time'][m] += PeakBasics.compute_center_times(events[m])
        return r
    

@export
class PeakExtra(strax.MergeOnlyPlugin):
    """Event level information with peak waveforms.
    """
    depends_on = ('events_oi', 
                  'peaks_oi', 
                  'peak_basics_oi')
    provides = 'peak_extra'
    __version__ = '0.0.0'


def get_context(signal_type, version='xenonnt_v6', output_folder='/project2/lgrandi/yuanlq/xenonnt/'):
    """wrapper around context to get peaks of a certain source. Now supporting KrS1A, KrS1B, Ar37.

    Args:
        signal_type (str): signal type that can be one of KrS1A, KrS1B, ArS1
        version (str, optional): cutax version to load events. Defaults to 'xenonnt_v6'.
        output_folder (str, optional): strax data output folder. Defaults to '/project2/lgrandi/yuanlq/xenonnt/'.
    """
    assert signal_type in SUPPORTED_SIGNALS, "Please try signal type in the supported list: %s"%(SUPPORTED_SIGNALS)

    # init context
    context_function = getattr(cutax, version)
    cutax_context = context_function(output_folder=output_folder)

    # whether need to depend on event_info_double 
    cutax_context.register_all(cutax.cut_lists.kr83m)
    cutax_context.register_all(cutax.cut_lists.rn220)
    cutax_context.register_all(cutax.cut_lists.ar37)
    cutax_context.register_all(cutax.cut_lists.basic)
    if signal_type[:-1] == 'A' or 'B': 
        cutax_context.register(EventsDoubleMerged)
    else:
        cutax_context.register(EventsMerged)
    cutax_context.register(EventsOfInterest)
    cutax_context.register(PeaksOfInterest)
    cutax_context.register(PeakBasicsOfInterest)
    cutax_context.register(PeakExtra)

    return cutax_context


def get_peaks(runs, signal_type, version='xenonnt_v6', output_folder='/project2/lgrandi/yuanlq/xenonnt/', **kargs):
    """wrapper around get_array to get peaks of a certain source. Now supporting KrS1A, KrS1B, Ar37.

    Args:
        runs (str or 1darray): runs to extract certain signal. Assumed all type of runs if provided array.
        signal_type (str): signal type that can be one of KrS1A, KrS1B, ArS1
        version (str, optional): cutax version to load events. Defaults to 'xenonnt_v6'.
        output_folder (str, optional): strax data output folder. Defaults to '/project2/lgrandi/yuanlq/xenonnt/'.
    """
    cutax_context = get_context(signal_type=signal_type, version=version, output_folder=output_folder)
    result = cutax_context.get_array(runs, 'peaks_oi', config=dict(signal_type=signal_type, **kargs))
    return result


def get_peak_basics(runs, signal_type, version='xenonnt_v6', output_folder='/project2/lgrandi/yuanlq/xenonnt/', **kargs):
    """wrapper around get_array to get peak_basics of a certain source. Now supporting KrS1A, KrS1B, Ar37.

    Args:
        runs (str or 1darray): runs to extract certain signal. Assumed all type of runs if provided array.
        signal_type (str): signal type that can be one of KrS1A, KrS1B, ArS1
        version (str, optional): cutax version to load events. Defaults to 'xenonnt_v6'.
        output_folder (str, optional): strax data output folder. Defaults to '/project2/lgrandi/yuanlq/xenonnt/'.
    """
    cutax_context = get_context(signal_type=signal_type, version=version, output_folder=output_folder)
    result = cutax_context.get_array(runs, 'peak_basics_oi', config=dict(signal_type=signal_type, **kargs))
    return result


def get_peak_extra(runs, signal_type, version='xenonnt_v6', output_folder='/project2/lgrandi/yuanlq/xenonnt/', **kargs):
    """wrapper around get_array to get peak_extra of a certain source. Now supporting KrS1A, KrS1B, Ar37.

    Args:
        runs (str or 1darray): runs to extract certain signal. Assumed all type of runs if provided array.
        signal_type (str): signal type that can be one of KrS1A, KrS1B, ArS1
        version (str, optional): cutax version to load events. Defaults to 'xenonnt_v6'.
        output_folder (str, optional): strax data output folder. Defaults to '/project2/lgrandi/yuanlq/xenonnt/'.
    """
    cutax_context = get_context(signal_type=signal_type, version=version, output_folder=output_folder)
    result = cutax_context.get_array(runs, 'peak_extra', config=dict(signal_type=signal_type, **kargs))
    return result