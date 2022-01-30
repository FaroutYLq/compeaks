import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats, interpolate, optimize
import numba


def align_area_range(peaklets, percent=20, sample_per_wf=110, align_at=20, dt=10):
    """Align the waveform of peak level data at a certain point of area range.
    Args:
        peaklets (ndarray): Peak level data. 
        percent (int/float, optional): How many percent area range you want to find the time. Defaults to 20.
        sample_per_wf (int, optional): Number of samples in the aligned waveform. Defaults to 110.
        align_at (int, optional): The output waveforms will be aligned at this index. Defaults to 20.
        dt (int, optional): Assumed time length for each sample in the waveform. Defaults to 10 ns.
    Returns:
        (2darray): 2d array of waveforms of the aligned peaks.
    """    
    peaklets = peaklets[peaklets['dt']==dt]
    area_percent_time = area_percent_times(peaklets, percent)
    aligned_wfs = align_peaks_at_times(peaklets, area_percent_time, sample_per_wf, align_at)

    return aligned_wfs


def align_peaks_at_times(peaklets, align_time, sample_per_wf=110, align_at=20):
    """Align the waveform of peak level data at a certain point. We assign trivial 0s outside the 
    alignment range.
    Args:
        peaklets (ndarray): Peak level data. Assumed sharing dt.
        align_time (ndarray): 1d array containing the time point to align
        sample_per_wf (int, optional): Number of samples in the aligned waveform. Defaults to 110.
        align_at (int, optional): The output waveforms will be aligned at this index. Defaults to 20.
    Returns:
        (2darray): 2d array of waveforms of the aligned peaks.
    """    
    aligned_wfs = np.zeros((len(peaklets),sample_per_wf))
    dt = peaklets[0]['dt']
    
    assert len(np.unique(peaklets['dt'])==1)
    
    for i,p in tqdm(enumerate(peaklets)):
        # Find the closest sample to the alignment time
        area_percent_sample_i = int(np.around(align_time[i]/dt))

        start_sample_i = max(area_percent_sample_i-align_at, 0) # put the align point at 20th sample
        end_sample_i = min(area_percent_sample_i+(sample_per_wf-align_at-1),len(p['data']))

        aligned_wfs[i][align_at-(area_percent_sample_i-start_sample_i):
                       align_at+(end_sample_i-area_percent_sample_i)] = p['data'][start_sample_i:end_sample_i]

        # Normalization
        aligned_wfs[i] = aligned_wfs[i] / aligned_wfs[i].sum()
    
    return aligned_wfs


def area_percent_times(peaklets, percent=20):
    """Return time stamps in each peaklets for the point to align.
    Args:
        peaklets (ndarray): Peak level data. 
        percent (int/float, optional): How many percent area range you want to find the time. Defaults to 20.
    Returns:
        (ndarray): 1d array of time in unit of ns in each peaklets for the point to align.
    """    
    # Manually find the 50% area point by computing CDF ourselves.
    midpoint_times = peaklets['dt']*np.argmin(abs(np.cumsum(peaklets['data'],axis=1) - 0.5*peaklets['area'][:,np.newaxis]),axis=1)

    percent_index = percent // 10 # We only have percent area decile defined every ten.
    # Based on area decile from mid point.
    area_percent_times = midpoint_times + peaklets['area_decile_from_midpoint'][:,percent_index]


def delayed_sum(peaks, samples_delayed=4):
    """Delay each event by samples_delayed and overlap it with the original waveform
    parameters.
    Args:
        peaks (ndarray): Peak level data. 
        samples_delayed (int, optional): [description]. Defaults to 4.
    Returns:
        (type): 2d array of overlapped waveforms.
    """    
    peaks_length = len(peaks[0]['data'])
    summed_waveforms = np.zeros((len(peaks),peaks_length+samples_delayed))
    
    for i in range(len(peaks)):
        summed_waveforms[i,:peaks_length] = peaks[i]['data'] 
        summed_waveforms[i,samples_delayed:] -= peaks[i]['data']
    
    return summed_waveforms


def interp_summed_waveforms(summed_waveforms, x_new=np.arange(1000), dt=10):
    """To do constant fraction discriminator, we need to find the null point,
    which would be a challenge for low resolution waveforms. Then we need to interpolate
    to get 'higher resolution' to find null points.
    Args:
        summed_waveforms (ndarray): 2d array of delay-summed waveforms from peaks
        x_new (ndarray, optional): New time coordinates after interpolation. Defaults to np.arange(1000).
        dt (int, optional): Time length of each sample in unit of ns in the waveform. Defaults to 10.
    Returns:
        (2darray): 2d array of interpolated waveforms.
    """    
    length = len(summed_waveforms[0])
    x_old = dt*np.arange(length)
    interp_waveforms = np.zeros((summed_waveforms.shape[0], len(x_new)))
    
    for i in range(summed_waveforms.shape[0]):
        f = interpolate.interp1d(x_old, summed_waveforms[i])
        
        interp_waveforms[i] = f(x_new)
    
    return interp_waveforms


def find_null_point(interp_waveforms, dt=10):
    """Find the index closest to null points in the aligned waveforms.
    Args:
        interp_waveforms (ndarray): 2d array of interpolated waveforms.
        dt (int, optional): Time length of each sample in unit of ns in the waveform. Defaults to 10.
    Returns:
        (1darray): 1d array of the index closest to null points in the aligned waveforms.
    """
    to_align = []
    for i in range(interp_waveforms.shape[0]):
        waveform = interp_waveforms[i]
        mini = np.argmin(waveform)
        try:
            positive_ind = np.where(waveform[:mini]>0)[0].max()
            positive_ind = round(positive_ind/dt)
            to_align.append(positive_ind)
        except:
            print('Cannot find null point')
    return np.array(to_align)


def align_cfd(peaks, samples_delayed=4, x_new=np.arange(1000), dt=10):
    """Align waveforms based on the constant fraction discriminator. 
    Args:
        peaks (ndarray): Peak level data. 
        samples_delayed (int, optional): The delay of numebr of sample in CFD. Defaults to 4.
        x_new ([type], ndarray): 1d array containing the time stamp of waveforms. Defaults to np.arange(1000).
        dt (int, optional): Assumed time length for each sample in the waveform. Defaults to 10 ns.
    Returns:
        (2darray): 2d array of waveforms of the aligned peaks.
    """
    peaks = peaks[peaks['dt']==dt]
    dt = peaks[0]['dt']
    waveforms_d = delayed_sum(peaks=peaks, samples_delayed=4)
    interp_waveforms = interp_summed_waveforms(summed_waveforms=waveforms_d, x_new=np.arange(1000), dt=10)
    align_ind = find_align_point(interp_waveforms=interp_waveforms, dt=10)
    
    aligned_wfs = np.zeros((len(peaks),110))
    for i,p in enumerate(peaks):
        start_sample_i = max(align_ind[i]-30, 0)
        end_sample_i = min(align_ind[i]+79,len(p['data']))
        
        # fix null point from cfd at index 30
        aligned_wfs[i][30-(align_ind[i]-start_sample_i):
                   30+(end_sample_i-align_ind[i])] = p['data'][start_sample_i:end_sample_i]
    
        # Normalization
        aligned_wfs[i] = aligned_wfs[i] / aligned_wfs[i].sum()
    
    return aligned_wfs


def overlay_wfs(average_wf, individual_wfs, strings='', xlim=(10,60), ylim=(-0.01,0.15)):
    """Put the average waveforms and individual waveforms together to compare. Plot the
    overlayed waveforms.
    Args:
        average_wf (1darray): One vector of average waveform from some alignment techniques.
        individual_wfs (2darray): 2d arrays with each row (axis0) to be an individual waveform.
        strings (str, optional): Description you want to add to the plot title. Defaults to ''.
        xlim (tuple, optional): Plot x range (x_left, x_right). Default to (10,60).
        ylim (tuple, optional): Plot y range (y_bot, x_top). Default to (-0.01,0.15).
    """   

    # assumed all normalized and aligned
    plt.figure(dpi=200)
    for wf in individual_wfs:
        plt.plot(wf/wf.sum(), alpha=0.01, color='k')
    plt.plot(average_wf/average_wf.sum(),color = 'r')
    plt.xlabel('samples')
    plt.ylabel('normalized amplitude')
    plt.xlim(xlim[0],xlim[1])
    lt.title('Average waveform VS individual waveform; ')
    """
    plt.title('Average waveform VS individual waveform; '+strings+ '\n Residual = %s'%(
            sum_square_remainder(average_wf, individual_wfs)))
    """
    plt.ylim(ylim[0],ylim[1])
    plt.show()


def sum_square_remainder(average_wf, individual_wfs):
    """The average summed square difference remainder per sample. We use this to evaluate how 
        typical the average waveform can be.
    Args:
        average_wf (ndarray): One vector of average waveform from some alignment techniques.
        individual_wfs ([type]): individual_wfs (ndarray): 2d arrays with each row (axis0) 
            to be an individual waveform.
    Returns:
        (float): The average summed square difference remainder per sample.
    """
    length = len(average_wf)
    sr = np.sum((average_wf[np.newaxis, :] - individual_wfs)**2, axis=1).mean()/length
    return sr


def align_gatti(peaks, dt=10):
    '''Function self-align peaks based on the best signal correlation between them. 
    Notes by Daniel:
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:wenz:comissioning:tpc:gatti_filter

    Args:
        peaks (ndarray): Peak level data. 
        dt (int, optional): Time length of each sample in unit of ns in the waveform. Defaults to 10.
    '''
    peaks = peaks[peaks['dt']=dt]
    n_peaks = len(peaks)
    # Get first peak and align according to maximum.
    p1 = peaks[0]['data'][:peaks[0]['length']]/peaks[0]['area']
    start_index = 300-np.argmax(p1)

    res = np.zeros((n_peaks, 600))
    res[0][start_index:start_index+len(p1)] += p1
    for i in range(1, n_peaks):
        p2 = peaks[i]['data'][:peaks[i]['length']]/peaks[i]['area']
        template = np.mean(res[:i], axis=0)
        corr = np.correlate(template, p2)
        shift = np.argmax(corr)
        res[i][shift:shift+len(p2)] = p2
    
    return res
