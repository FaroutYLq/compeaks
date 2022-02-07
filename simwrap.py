import wfsim
import numpy as np
import nestpy
import straxen
import sys
import gzip
import pickle
from multihist import Hist1d
import matplotlib.pyplot as plt
import nestpy
import pandas as pd
import pema

nc = nestpy.NESTcalc(nestpy.DetectorExample_XENON10())
density = 2.94
driftfield= 18.3
NSUMWVSAMPLES = 200
NWIDTHS = 11
INT_NAN = -99999
FAX_CONFIG_DEFAULT={
        's1_model_type': 'nest+optical_propagation',
        's1_pattern_map': '/dali/lgrandi/xenonnt/simulations/optphot/mc_v4.1.0/S1_1.69_0.99_0.99_0.99_0.99_10000_100_30/XENONnT_S1_xyz_patterns_LCE_corrected_QEs_MCv4.1.0_wires.pkl',
        's1_time_spline': 'XENONnT_s1_proponly_va43fa9b_wires_20200625.json.gz',
        'enable_noise': True,
        'enable_electron_afterpulses': True,
        'enable_pmt_afterpulses': True,
    }


def instruction(interaction_type, energy, N_events=1):  
    """Generate simulation instruction.
    nr=0, wimp=1, b8=2, dd=3, ambe=4, cf=5, ion=6, gammaray=7,
    beta=8, ch3t=9, c14=10, kr83m=11, nonetype=12

    Args:
        interaction_type (int): Following the NEST type of intereaction.
        energy (float or list): energy deposit in unit of keV
        N_events (int, optional): simulation number. Defaults to 1.
    """
    if (type(energy) != float) and (type(energy) != int):
        assert len(energy)==2, 'You must input a single energy or array like [low, high] for uniform dist'

    instr = np.zeros(N_events , wfsim.instruction_dtype)
    #instr['x'] = evt['x']
    #instr['y'] = evt['y']
    instr['z'] = np.random.uniform(-134,-13) # Fiducial volume z
    instr['type'] = 1
    instr['recoil'] = interaction_type
    instr['local_field'] = driftfield

    for i in range(0, N_events):
        if (type(energy) == float) or (type(energy) == int):
            e = energy
        else:
            e = np.random.uniform(low=energy[0], high=energy[1])
        yields = nc.GetYields(nestpy.INTERACTION_TYPE(interaction_type), e, density, driftfield)
        cur_q = nc.GetQuanta(yields)
        instr['time'][i] = (i+1) * int(1e6)
        instr['amp'][i] = cur_q.photons
        instr['n_excitons'][i] = cur_q.excitons
        instr['e_dep'][i] = e
    
    return instr  


def get_sim_context(interaction_type, energy, N=10000, **kargs):
    """Generate simulation context.
    nr=0, wimp=1, b8=2, dd=3, ambe=4, cf=5, ion=6, gammaray=7,
    beta=8, ch3t=9, c14=10, kr83m=11, nonetype=12

    Args:
        interaction_type (int): Following the NEST type of intereaction.
        energy (float or list): energy deposit in unit of keV
        N (int, optional): simulation number. Defaults to 1.
    """
    fax_instr = []
    N_events = 1
    for j in range(N):
        temp = instruction(interaction_type, energy, N_events)
        temp['time'] = temp['time'] + j * N_events * int(1e6)
        fax_instr.append(temp)
    fax_instr = np.concatenate(fax_instr)
    if (type(energy) == float) or (type(energy) == int):
        file_name = '/dali/lgrandi/yuanlq/s1_wf_comparison/wfsim_config/int%s_e%s_%s.csv'%(interaction_type, int(energy),int(energy))
    else:
        file_name = '/dali/lgrandi/yuanlq/s1_wf_comparison/wfsim_config/int%s_e%s_%s.csv'%(interaction_type, int(energy[0]),int(energy[1]))
    pd.DataFrame(fax_instr).to_csv(file_name, index=False)

    stwf = straxen.contexts.xenonnt_simulation(
        cmt_run_id_sim = '034000',
        output_folder='/dali/lgrandi/yuanlq/s1_wf_comparison/wfsim_data',
        fax_config='fax_config_nt_sr0_v0.json',)

    config_dict = FAX_CONFIG_DEFAULT.update(kargs)
    stwf.set_config(dict(fax_config_overide=config_dict))

    stwf.set_config(
        dict(fax_file=file_name,
            right_raw_extension=20000,
            event_rate=1000,
            chunk_size=1,
            nchunk=10,))

    stwf.register_all(pema.match_plugins)

    return stwf


def sim_peak_extra(peaks, peak_basics, truth, match):
    """generate simulation peak_extra by putting useful info in peaks, peak_baiscs, truth and pema match together.

    Args:
        peaks (ndarray): straxen peaks
        peak_basics (ndarray): straxen peak_basics
        truth (ndarray): wfsim truth
        match (ndarray): pema match_acceptance_extended

    Returns:
        (ndarray): peak extra for simulation (not exactly same dtype as is in regular peak_extra)
    """
    dtypes = [
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
          'type'), np.int8),
        (('Waveform data in PE/sample (not PE/ns!)',
          'data'), np.float32, NSUMWVSAMPLES),
        (('Peak widths: time between nth and 5th area decile [ns]',
          'area_decile_from_midpoint'), np.float32, NWIDTHS),
        (('x coordinates', 'x'), np.float32),
        (('y coordinates', 'y'),np.float32),
        (('z coordinates', 'z'), np.float32),
    ]
    
    peak_indecies = match['matched_to']
    matched_mask = peak_indecies != INT_NAN

    truth = truth[matched_mask] # in case of missed
    peaks = peaks[peak_indecies[matched_mask]]
    peak_basics = peak_basics[peak_indecies[matched_mask]]
    
    peak_extra = np.zeros(len(peaks), dtype=dtypes)
    
    for i in range(len(dtypes)):
        field = dtypes[i][0][1]
        if field == 'data' or field == 'area_decile_from_midpoint':
            peak_extra[field] = peaks[field]
        elif field == 'x' or field == 'y' or field == 'z':
            peak_extra[field] = truth[field]
        else:
            peak_extra[field] = peak_basics[field]
    
    return peak_extra


def get_sim_peak_extra(runid, interaction_type, energy, N=10000, straxen_config={}, **kargs):
    """Get peak_extra for simulation.
    nr=0, wimp=1, b8=2, dd=3, ambe=4, cf=5, ion=6, gammaray=7,
    beta=8, ch3t=9, c14=10, kr83m=11, nonetype=12

    Args:
        runid (str): runid in wfsim
        interaction_type (int): Following the NEST type of intereaction.
        energy (float or list): energy deposit in unit of keV
        N (int, optional): simulation number. Defaults to 1.
        **kargs: overide fax config.

    Returns:
        (ndarray): peak extra for simulation (not exactly same dtype as is in regular peak_extra)
    """
    st = get_sim_context(interaction_type=interaction_type, energy=energy, N=N, **kargs)
    peaks = st.get_array(runid, 'peaks', config=straxen_config)
    peak_basics = st.get_array(runid, 'peak_basics', config=straxen_config)
    truth = st.get_array(runid, 'truth', config=straxen_config)
    match = st.get_array(runid, 'match_acceptance_extended', config=straxen_config)
    peak_extra = sim_peak_extra(peaks, peak_basics, truth, match)

    return peak_extra


