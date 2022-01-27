import numpy as np
import strax
from straxen import pre_apply_function
from straxen.plugins.peak_processing import PeakBasics
export, __all__ = strax.exporter()

SUPPORTED_SIGNALS = ['KrS1A', 'KrS1B', 'ArS1', 'RnS1']


@export
class EventsDoubleOfInterest(strax.MergeOnlyPlugin):
    """Extract event info double data from cutax cuts.
    """
    depends_on = ('event_info_double', 
                  'cut_Kr_DoubleS1_SingleS2', 
                  'cut_fiducial_volume')
    provides = 'events_oi'
    __version__ = '0.0.0'


@export
class EventsOfInterest(strax.MergeOnlyPlugin):
    """Extract event info data from cutax cuts.
    """
    depends_on = ('event_info', 
                  'cuts_ar37_kshell_s1s2', 
                  'cut_fiducial_volume', 
                  'cuts_rn220')
    provides = 'events_oi'
    __version__ = '0.0.0'


@export
@strax.takes_config(
    strax.Option('signal_type', type=str,
                 help='signal type that can be one of KrS1A, KrS1B, ArS1, RnS1'),
    strax.Option('n_tpc_pmts', type=int,
                 help='Number of TPC PMTs'),
    strax.Option('fv_cut', type=bool, default =True,
                 help='whether to apply fiducial volume cut or not'),
)
class PeaksOfInterest(strax.Plugin):
    """Extract peak-level data from events level data.
    """
    depends_on = ('events_oi', 'peaks')
    provides = 'peaks_oi'
    __version__ = '0.0.0'


    def infer_dtype(self):
        peaklets_type=strax.peak_dtype(n_channels=self.config['n_tpc_pmts'])
        return peaklets_type

    def cut_name(self):
        assert self.config['signal_type'] in SUPPORTED_SIGNALS, "Please try signal type in the supported list: %s"%(SUPPORTED_SIGNALS)
        if self.config['signal_type'] == 'ArS1':
            cut = 'cuts_ar37_kshell_s1s2'
        elif self.config['signal_type'] == 'RnS1':
            cut = 'cuts_rn220'
        elif self.config['signal_type'] == 'KrS1A' or 'KrS1B':
            cut = 'cut_Kr_DoubleS1_SingleS2'
        return cut

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
        if self.config['fv_cut']:
            events_oi = events[events['cut_fiducial_volume']]
        events_oi = events_oi[events_oi[self.cut_name()]]
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
    dtype = [
        (('Start time of the peak (ns since unix epoch)',
          'time'), np.int64),
        (('End time of the peak (ns since unix epoch)',
          'endtime'), np.int64),
        (('Weighted center time of the peak (ns since unix epoch)',
          'center_time'), np.int64),
        (('Peak integral in PE',
            'area'), np.float32),
        (('Number of PMTs contributing to the peak',
            'n_channels'), np.int16),
        (('PMT number which contributes the most PE',
            'max_pmt'), np.int16),
        (('Area of signal in the largest-contributing PMT (PE)',
            'max_pmt_area'), np.float32),
        (('Total number of saturated channels',
          'n_saturated_channels'), np.int16),
        (('Width (in ns) of the central 50% area of the peak',
            'range_50p_area'), np.float32),
        (('Width (in ns) of the central 90% area of the peak',
            'range_90p_area'), np.float32),
        (('Fraction of area seen by the top array '
          '(NaN for peaks with non-positive area)',
            'area_fraction_top'), np.float32),
        (('Length of the peak waveform in samples',
          'length'), np.int32),
        (('Time resolution of the peak waveform in ns',
          'dt'), np.int16),
        (('Time between 10% and 50% area quantiles [ns]',
          'rise_time'), np.float32),
        (('Hits within tight range of mean',
          'tight_coincidence'), np.int16),
        (('PMT channel within tight range of mean',
          'tight_coincidence_channel'), np.int16),
        (('Classification of the peak(let)',
          'type'), np.int8)
    ]

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
    

