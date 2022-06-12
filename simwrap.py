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
import datetime
import generator

downloader = straxen.MongoDownloader()
nc = nestpy.NESTcalc(nestpy.DetectorExample_XENON10())
density = 2.94
driftfield= 22.92 # not really used. we are using spatial dependent map
NSUMWVSAMPLES = 200
NWIDTHS = 11
INT_NAN = -99999
FIELD_FILE="fieldmap_2D_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz"
FAX_CONFIG_DEFAULT={
        's1_time_spline': '/project2/lgrandi/yuanlq/shared/s1_optical/XENONnT_s1_proponly_pc_reflection_optPhot_perPMT_S1_local_20220510.json.gz',
        'enable_noise': True,
        'enable_electron_afterpulses': True,
        'enable_pmt_afterpulses': False,  
        'override_s1_photon_time_field': -1
    }
FIELD_MAP = straxen.InterpolatingMap(
                straxen.get_resource(downloader.download_single(FIELD_FILE),
                fmt="json.gz"),
                method="RegularGridInterpolator")


def generate_vertex(r_range=(0,64), 
                    z_range=(-142, -6), size=1):
    """Generate x,y,z position. Copied from https://github.com/XENONnT/analysiscode/blob/master/wfsim/sample_generation/generators.py

    Args:
        r_range (tuple, optional): radial coordinate. Defaults to (0,64).
        z_range (tuple, optional): depth range. Defaults to (-142, -6).
        size (int, optional): _description_. Defaults to 1.

    Returns:
        x, y, z (float): x, y, z coordinate of the simulated vertex
    """
    phi = np.random.uniform(size=size)*2*np.pi
    r = r_range[1]*np.sqrt(np.random.uniform( (r_range[0]/r_range[1])**2, 1, size=size) )
    z = np.random.uniform(z_range[0],z_range[1],size=size)
    x=(r*np.cos(phi))
    y=(r*np.sin(phi))

    return x[0],y[0],z[0]


def single_s1_instruction(interaction_type, energy, N_events=1):
    """Generate simulation instruction for single S1 peak simulation.
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
    x, y, z = generate_vertex() # Fiducial volume considered
    instr['x'] = x
    instr['y'] = y
    instr['z'] = z
    instr['type'] = 1
    instr['recoil'] = interaction_type
    instr['local_field'] = FIELD_MAP([np.sqrt(x**2 + y**2),z])[0]

    for i in range(0, N_events):
        if (type(energy) == float) or (type(energy) == int):
            e = energy
        else:
            e = np.random.uniform(low=energy[0], high=energy[1])
        yields = nc.GetYields(nestpy.INTERACTION_TYPE(interaction_type), e, density, instr['local_field'])
        cur_q = nc.GetQuanta(yields)
        instr['time'][i] = (i+1) * int(1e6)
        instr['amp'][i] = cur_q.photons
        instr['n_excitons'][i] = cur_q.excitons
        instr['e_dep'][i] = e
    
    return instr  


def instruction(interaction_type, energy, N=10000):
	"""Build instruction, depending on interaction type and energy.

	Args:
		interaction_type (int): Following the NEST type of intereaction.
		energy (float or list): energy deposit in unit of keV
		N_events (int, optional): simulation number. Defaults to 10000.
	"""
	# If not Krypton, only simulate one peaks.
	if interaction_type != 11:
		fax_instr = []
		N_events = 1
		for j in range(N):
			temp = single_s1_instruction(interaction_type, energy, N_events)
			temp['time'] = temp['time'] + j * N_events * int(1e6)
			fax_instr.append(temp)
		fax_instr = np.concatenate(fax_instr)
	
	# If Krypton, simulate the whole event using generators.
	else:
		fax_instr = generator.generator_Kr83m(
						n_tot=N, 
						recoil=11,
						rate = 30.0, # in Hz, hardcoded for Kr
						fmap=FIELD_MAP,
						nc=nc, #nest calculator object
						r_range = (0, 64),
						z_range = (-142, -6))
	
	energy = 41 # hardcoded for Krypton
	file_name = instr_file_name(fax_instr, interaction_type, energy)
	return file_name


def instr_file_name(fax_instr, interaction_type, energy):
	"""Generate file name for instruction and save it.

	Args:
		fax_instr (array): Fax instruction.
		interaction_type (int): Following the NEST type of intereaction.
		energy (float or list): energy deposit in unit of keV
	"""
	now_str = datetime.datetime.now().strftime('%Y%m%d%H%M')
	if (type(energy) == float) or (type(energy) == int):
		file_name = '/dali/lgrandi/yuanlq/s1_wf_comparison/wfsim_config/int%s_e%s_%s_%s.csv'%(interaction_type, int(energy),int(energy),now_str)
	else:
		file_name = '/dali/lgrandi/yuanlq/s1_wf_comparison/wfsim_config/int%s_e%s_%s_%s.csv'%(interaction_type, int(energy[0]),int(energy[1]), now_str)
	pd.DataFrame(fax_instr).to_csv(file_name, index=False)
	print('Instruction file at: %s'%(file_name))

	return file_name


def get_sim_context(interaction_type, energy, N=10000, **kargs):
    """Generate simulation context. Assumed single peak simulation, which might be unphysical for Kryptons.
    nr=0, wimp=1, b8=2, dd=3, ambe=4, cf=5, ion=6, gammaray=7,
    beta=8, ch3t=9, c14=10, kr83m=11, nonetype=12

    Args:
        interaction_type (int): Following the NEST type of intereaction.
        energy (float or list): energy deposit in unit of keV
        N (int, optional): simulation number. Defaults to 1.
    """
    # generate and save instruction
    file_name = instruction(interaction_type, energy, N)
    

    stwf = straxen.contexts.xenonnt_simulation(
        cmt_run_id_sim = '034000',
        output_folder='/dali/lgrandi/yuanlq/s1_wf_comparison/wfsim_data',
        fax_config='fax_config_nt_sr0_v1.json',)

    config_dict = FAX_CONFIG_DEFAULT
    config_dict.update(**kargs)
    print('FAX config override:')
    print(config_dict)
    stwf.set_config(dict(fax_config_override=config_dict))

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
        (('Integral per channel [PE]',
          'area_per_channel'), np.float32, 494),
        (('Length of the peak waveform in samples',
          'length'), np.int32),
        (('Time resolution of the peak waveform in ns',
          'dt'), np.int16),
        (('Time between 10% and 50% area quantiles [ns]',
          'rise_time'), np.float32),
        (('Hits within tight range of mean',
          'tight_coincidence'), np.int16),
        (('Number of hits contributing at least one sample to the peak',
          'n_hits'), np.int32),
        (('PMT channel within tight range of mean',
          'tight_coincidence_channel'), np.int16),
        (('Classification of the peak(let)',
          'type'), np.int8),
        (('Waveform data in PE/sample (not PE/ns!)',
          'data'), np.float32, NSUMWVSAMPLES),
        (('Peak widths: time between nth and 5th area decile [ns]',
          'area_decile_from_midpoint'), np.float32, NWIDTHS),
        (('Peak widths in range of central area fraction [ns]',
          'width'), np.float32, NWIDTHS),
        (('Largest gap between hits inside peak [ns]',
          'max_gap'), np.int32),
        (('Maximum interior goodness of split',
          'max_goodness_of_split'), np.float32),
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
        if (field == 'data') or (field == 'area_decile_from_midpoint') or (
            field=='n_hits') or (field=='area_per_channel') or (
            field=='width') or (field=='max_gap') or (field=='max_goodness_of_split'):
            peak_extra[field] = peaks[field]
        elif field == 'x' or field == 'y' or field == 'z':
            peak_extra[field] = truth[field]
        else:
            peak_extra[field] = peak_basics[field]
    
    return peak_extra


def get_sim_peak_extra(runid, interaction_type, energy, N=100000, straxen_config={}, **kargs):
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
    truth = st.get_array(runid, 'truth', config=straxen_config)
    peaks = st.get_array(runid, 'peaks', config=straxen_config)
    peak_basics = st.get_array(runid, 'peak_basics', config=straxen_config)
    match = st.get_array(runid, 'match_acceptance_extended', config=straxen_config)
    peak_extra = sim_peak_extra(peaks, peak_basics, truth, match)

    return peak_extra


