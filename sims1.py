import wfsim
import numpy as np
import nestpy
import straxen
import sys
import gzip
import pickle
from multihist import Hist1d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# Mean AFTs around certain z positions from Ar37 S1 data
ZSLIACES = np.array([-128.20647559, -116.14423757, -104.08199954,  -92.01976151, -79.95752349,  
                     -67.89528546,  -55.83304744,  -43.77080941, -31.70857139,  -19.64633336])
AFTS = np.array([0.11675146, 0.14507483, 0.17712556,0.21277627, 0.2502666,
                 0.29347506, 0.33834386, 0.3823964, 0.43201056, 0.47643015])


def get_config(config_file='fax_config_nt_sr0_v0.json'):
    """Get wfsim configuration file.

    Args:
        config_file (str, optional): Configuration file name. Defaults to 'fax_config_nt_sr0_v0.json'.
    """
    config = config = straxen.get_resource(config_file, config_file.split('.')[-1])
    return config


def scintillation(interaction_type, e_dep, config_file='fax_config_nt_sr0_v0.json'):
    """Scintillation time delay from NEST. NEST interaction type:
    nr=0, wimp=1, b8=2, dd=3, ambe=4, cf=5, ion=6, gammaray=7,
    beta=8, ch3t=9, c14=10, kr83m=11, nonetype=12

    Args:
        interaction_type (int): Following the NEST type of intereaction.
        e_dep (float): energy deposit in unit of keV
        config_file (str, optional): File as wfsim config. Defaults to 'fax_config_nt_sr0_v0.json'.

    Returns:
        scint_t, scint_y [1darray]: scintillation delay as a function af time
    """
    config = get_config(config_file=config_file)
    nestpy_calc = nestpy.NESTcalc(nestpy.DetectorExample_XENON10())
    local_field = config['drift_field']
    interaction = nestpy.INTERACTION_TYPE(interaction_type)
    nucleus_A = 131.293
    nucleus_Z = 54.
    lxe_density = 2.862

    hist = Hist1d([], bins=np.linspace(-100, 1000, 1101),)
    for i in range(10000):
        energy_deposit = e_dep
        
        y = nestpy_calc.GetYields(interaction,
                                energy_deposit,
                                lxe_density,
                                local_field,
                                nucleus_A,
                                nucleus_Z,
                                )
        q = nestpy_calc.GetQuanta(y, lxe_density)
        scint_time = nestpy_calc.GetPhotonTimes(
            nestpy.INTERACTION_TYPE(interaction_type),
            q.photons,
            q.excitons,
            local_field,
            energy_deposit,)
        
        scint_time = np.clip(scint_time, 0, config.get('maximum_recombination_time', 10000))
        
        hist.add(scint_time)

    scint_t = hist.bin_centers
    scint_t = np.around(scint_t)+1
    scint_t = scint_t[:-1]
    scint_y = hist.histogram
    scint_y = scint_y[:-1]/scint_y[:-1].sum() 

    return scint_t, scint_y


def interp_aft(z_position):
    """Interpolate AFT based on z position.

    Args:
        z_position (float): z position.
    """
    f_aft = interp1d(ZSLIACES, AFTS)
    return f_aft(z_position)


def propagation(z_position, spline_file='XENONnT_s1_proponly_va43fa9b_wires_20200625.json.gz'):
    """Function gettting times from s1 timing splines. Weighted based on aft.

    Args:
        z_position (float): The Z positions of all s1 photon.
        spline_file: pointer to s1 optical propagation splines from resources.

    Returns:
        prop_t, prop_y (1darray): photon propagation as a function af time
    """
    aft = interp_aft(z_position)
    prop_t, prop_y_top = optical_propagation(top=True,  z_position=z_position, spline_file=spline_file)
    prop_t, prop_y_bot = optical_propagation(top=False, z_position=z_position, spline_file=spline_file)

    prop_y = prop_y_top*aft + prop_y_bot*(1-aft)
    prop_y = prop_y/prop_y.sum()
    return prop_t, prop_y


def optical_propagation(top, z_position, 
                        spline_file='XENONnT_s1_proponly_va43fa9b_wires_20200625.json.gz'):
    """Function gettting times from s1 timing splines in a certain array.
    
    Args:
        top (bool): whether we are in top array or not.
        z_position (float): The Z positions of all s1 photon.
        spline_file: pointer to s1 optical propagation splines from resources.

    Returns:
        prop_t, prop_y (1darray): photon propagation as a function af time in the specified array.
    """
    spline = wfsim.load_resource.make_map(spline_file)
    z_positions = z_position * np.ones(1000000)

    prop_time = np.zeros(1000000)
    z_rand = np.array([z_positions, np.random.rand(len(prop_time))]).T

    if top:
        prop_time = spline(z_rand, map_name='top')
    else:
        prop_time = spline(z_rand, map_name='bottom')

    prop_y, prop_t_bins = np.histogram(prop_time, bins=np.linspace(-10,500,511))
    prop_t = (prop_t_bins[1:]+prop_t_bins[:-1])/2

    return prop_t, prop_y


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def transit(config_file='fax_config_nt_sr0_v0.json'):
    """transit time spread in PMT as a function of time.

    Args:
        config_file (str, optional): Configuration file name. Defaults to 'fax_config_nt_sr0_v0.json'.
    """
    config = get_config(config_file=config_file)
    tts_t = np.arange(100)
    tts_y = gaussian(tts_t, config['pmt_transit_time_mean'], config['pmt_transit_time_spread']/2.355)
    tts_y = tts_y/tts_y.sum()
    return tts_t, tts_y


def spe(config_file='fax_config_nt_sr0_v0.json'):
    """SPE pulse shape.

    Args:
        config_file (str, optional): Configuration file name. Defaults to 'fax_config_nt_sr0_v0.json'.
    """
    config = get_config(config_file=config_file)
    pe_pulse_ys = np.array(config['pe_pulse_ys'])
    pe_pulse_ts = np.array(config['pe_pulse_ts'])
    pe_pulse_ys = pe_pulse_ys/pe_pulse_ys.sum()

    return pe_pulse_ts, pe_pulse_ys


def sims1(z_position, interaction_type, e_dep,
          spline_file='XENONnT_s1_proponly_va43fa9b_wires_20200625.json.gz', 
          config_file='fax_config_nt_sr0_v0.json'):
    """Do as what wfsim does, give the S1 pulse shape of certain type interaction at certain depth.

    Args:
        z_position (float): The Z positions of all s1 photon.
        interaction_type (int): Following the NEST type of intereaction.
        e_dep (float): energy deposit in unit of keV
        spline_file: pointer to s1 optical propagation splines from resources.
        config_file (str, optional): Configuration file name. Defaults to 'fax_config_nt_sr0_v0.json'.
    """
    prop_t, prop_y = propagation(z_position=z_position, spline_file=spline_file)
    scint_t, scint_y = scintillation(interaction_type=interaction_type, e_dep=e_dep, 
                                     config_file=config_file)
    tts_t, tts_y = transit(config_file=config_file)
    spe_t, spe_y = spe(config_file=config_file)

    result = np.convolve(prop_y, scint_y)
    result = np.convolve(result, spe_y)
    result = np.convolve(result, tts_y)

    return result


def get_s1_templates(interaction_type, e_dep, z_positions=ZSLIACES,
                     spline_file='XENONnT_s1_proponly_va43fa9b_wires_20200625.json.gz', 
                     config_file='fax_config_nt_sr0_v0.json',
                     plot=True):
    """Get s1 templates based on wfsim from different depth slices.

    Args:
        interaction_type ([type]): [description]
        e_dep ([type]): [description]
        z_positions ([type], optional): [description]. Defaults to ZSLIACES.
        spline_file (str, optional): [description]. Defaults to 'XENONnT_s1_proponly_va43fa9b_wires_20200625.json.gz'.
        config_file (str, optional): [description]. Defaults to 'fax_config_nt_sr0_v0.json'.
        plot (bool, optional): [description]. Defaults to True.
    """

    import matplotlib as mpl
    if plot:
        plt.figure(dpi=200)
        colors_er = plt.get_cmap('jet', 10*len(z_positions)+1)

    sim_wfs = np.zeros((len(z_positions),800))

    for e in range(len(z_positions)):
        y1 = sims1(z_position=z_positions[e], interaction_type=interaction_type, e_dep=e_dep)
        if plot:
            plt.plot(y1, c=colors_er(1+e*10) , linewidth=1, alpha=0.3)
        sim_wfs[e] = y1[100:900]

    if plot:
        norm = mpl.colors.Normalize(vmin=z_positions[0], vmax=z_positions[-1])
        sm1 = plt.cm.ScalarMappable(cmap=colors_er, norm=norm)
        sm1.set_array([])
        cb1 = plt.colorbar(sm1)
        cb1.set_label('depth [cm]')
        plt.xlabel('time [ns]')
        plt.title('WFSIM S1 at different positions [E=%skeV, NESTtype%s]'%(e_dep, interaction_type))
        plt.xlim(100,500)
        plt.show()
    
    return sim_wfs