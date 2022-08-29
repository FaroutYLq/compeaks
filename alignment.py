import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats, interpolate, optimize
import numba


METHODS = {"first_phr", "area_range", "self_adjusted"}


def align_first_phr(peaklets, dt=10, min_area=18, align_at=500):
    """Align the waveforms at first photon recorded (no alignment).
    Technically this function just normalized waveforms.

    Args:
        peaklets (ndarray): Peak level data.
        dt (int, optional): [description]. Defaults to 10.
        min_area (int, optional): [description]. Defaults to 18.
        align_at (int, optional): The output waveforms will be aligned at this time in unit of ns. Defaults to 500.

    Returns:
        (2darray): 2d array of waveforms of the aligned peaks.
    """
    peaklets = peaklets[(peaklets["dt"] == dt) & (peaklets["area"] > min_area)]
    time_ns = np.arange(10 * len(peaklets[0]["data"]))
    time_10ns = np.arange(len(peaklets[0]["data"])) * 10
    aligned_wfs = np.zeros((len(peaklets), 10 * len(peaklets[0]["data"])))
    normalized = peaklets["data"] / peaklets["area"][:, np.newaxis]
    f = interpolate.interp1d(
        time_10ns, normalized, axis=1, fill_value=(0, 0), bounds_error=False
    )

    aligned_wfs = f(time_ns)
    aligned_wfs[:, align_at:] = aligned_wfs[
        :, : len(peaklets[0]["data"]) * 10 - align_at
    ]
    aligned_wfs[:, :align_at] = 0
    return aligned_wfs


def align_area_range(peaklets, percent=20, align_at=500, dt=10, min_area=18):
    """Align the waveform of peak level data at a certain point of area range.

    Args:
        peaklets (ndarray): Peak level data.
        percent (int/float, optional): How many percent area range you want to find the time. Defaults to 20.
        align_at (int, optional): The output waveforms will be aligned at this index. Defaults to 50.
        dt (int, optional): Assumed time length for each sample in the waveform. Defaults to 10 ns.
        min_area (float, optional): Only align waveforms for these peaks who are larger than this number to kill bias in efficinecy.

    Returns:
        (2darray): 2d array of waveforms of the aligned peaks.
    """
    peaklets = peaklets[(peaklets["dt"] == dt) & (peaklets["area"] > min_area)]
    area_percent_time, interped_wfs = area_percent_times(peaklets, percent)
    aligned_wfs = align_peaks_at_times(interped_wfs, area_percent_time, align_at)

    return aligned_wfs


def align_peaks_at_times(peaklets, align_time, align_at=500):
    """Align the waveform of peak level data at a certain point. We assign trivial 0s outside the
    alignment range.

    Args:
        peaklets (ndarray): Interpolated wfs, at time resolution of 1 ns.
        align_time (ndarray): 1d array containing the time point to align
        align_at (int, optional): The output waveforms will be aligned at this time in unit of ns. Defaults to 500.

    Returns:
        (2darray): 2d array of waveforms of the aligned peaks.
    """
    aligned_wfs = np.zeros_like(peaklets)

    for i, p in enumerate(peaklets):
        # Find the closest sample to the alignment time
        area_percent_sample_i = align_time[i]

        start_sample_i = max(area_percent_sample_i - align_at, 0)
        end_sample_i = min(area_percent_sample_i + (len(p) - align_at - 1), len(p))

        aligned_wfs[i][
            align_at
            - (area_percent_sample_i - start_sample_i) : align_at
            + (end_sample_i - area_percent_sample_i)
        ] = p[start_sample_i:end_sample_i]

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
    # interpolate waveforms to have 1 ns time resolution
    time_ns = np.arange(10 * len(peaklets[0]["data"]))
    time_10ns = np.arange(len(peaklets[0]["data"])) * 10
    interped_wfs = np.zeros((len(peaklets), 10 * len(peaklets[0]["data"])))
    normalized = peaklets["data"] / peaklets["area"][:, np.newaxis]
    f = interpolate.interp1d(
        time_10ns, normalized, axis=1, fill_value=(0, 0), bounds_error=False
    )

    interped_wfs = f(time_ns)
    # Manually find the 50% area point by computing CDF ourselves.
    percent_times = np.argmin(
        abs(
            np.cumsum(interped_wfs, axis=1)
            - percent / 100 * np.sum(interped_wfs, axis=1)[:, np.newaxis]
        ),
        axis=1,
    )

    return percent_times, interped_wfs


def delayed_sum(peaks, samples_delayed=4):
    """Delay each event by samples_delayed and overlap it with the original waveform
    parameters.

    Args:
        peaks (ndarray): Peak level data.
        samples_delayed (int, optional): [description]. Defaults to 4.

    Returns:
        (type): 2d array of overlapped waveforms.
    """
    peaks_length = len(peaks[0]["data"])
    summed_waveforms = np.zeros((len(peaks), peaks_length + samples_delayed))

    for i in range(len(peaks)):
        summed_waveforms[i, :peaks_length] = peaks[i]["data"]
        summed_waveforms[i, samples_delayed:] -= peaks[i]["data"]

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
    x_old = dt * np.arange(length)
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
            positive_ind = np.where(waveform[:mini] > 0)[0].max()
            positive_ind = round(positive_ind / dt)
            to_align.append(positive_ind)
        except:
            print("Cannot find null point")
    return np.array(to_align)


def align_cfd(peaks, samples_delayed=4, x_new=np.arange(1000), dt=10, min_area=18):
    """Align waveforms based on the constant fraction discriminator.

    Args:
        peaks (ndarray): Peak level data.
        samples_delayed (int, optional): The delay of numebr of sample in CFD. Defaults to 4.
        x_new ([type], ndarray): 1d array containing the time stamp of waveforms. Defaults to np.arange(1000).
        dt (int, optional): Assumed time length for each sample in the waveform. Defaults to 10 ns.
        min_area (float, optional): Only align waveforms for these peaks who are larger than this number to kill bias in efficinecy.

    Returns:
        (2darray): 2d array of waveforms of the aligned peaks.
    """
    peaks = peaks[(peaks["dt"] == dt) & (peaks["area"] > min_area)]
    dt = peaks[0]["dt"]
    waveforms_d = delayed_sum(peaks=peaks, samples_delayed=4)
    interp_waveforms = interp_summed_waveforms(
        summed_waveforms=waveforms_d, x_new=np.arange(1000), dt=10
    )
    align_ind = find_null_point(interp_waveforms=interp_waveforms, dt=10)

    aligned_wfs = np.zeros((len(peaks), 110))
    for i, p in enumerate(peaks):
        start_sample_i = max(align_ind[i] - 30, 0)
        end_sample_i = min(align_ind[i] + 79, len(p["data"]))

        # fix null point from cfd at index 30
        aligned_wfs[i][
            30 - (align_ind[i] - start_sample_i) : 30 + (end_sample_i - align_ind[i])
        ] = p["data"][start_sample_i:end_sample_i]

        # Normalization
        aligned_wfs[i] = aligned_wfs[i] / aligned_wfs[i].sum()

    return aligned_wfs


def overlay_wfs(
    average_wf, individual_wfs, strings="", xlim=(10, 60), ylim=(-0.01, 0.15)
):
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
        plt.plot(wf / wf.sum(), alpha=0.01, color="k")
    plt.plot(average_wf / average_wf.sum(), color="r")
    plt.xlabel("samples")
    plt.ylabel("normalized amplitude")
    plt.xlim(xlim[0], xlim[1])
    lt.title("Average waveform VS individual waveform; ")
    """
    plt.title('Average waveform VS individual waveform; '+strings+ '\n Residual = %s'%(
            sum_square_remainder(average_wf, individual_wfs)))
    """
    plt.ylim(ylim[0], ylim[1])
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
    sr = (
        np.sum((average_wf[np.newaxis, :] - individual_wfs) ** 2, axis=1).mean()
        / length
    )
    return sr


def align_self_adjusted(peaks, dt=10, min_area=18, max_peaks=5000, align_at=500):
    """Function self-align peaks based on the best signal correlation between them.
    Notes by Daniel:
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:wenz:comissioning:tpc:gatti_filter

    Args:
        peaks (ndarray): Peak level data.
        dt (int, optional): Time length of each sample in unit of ns in the waveform. Defaults to 10.
        min_area (float, optional): Only align waveforms for these peaks who are larger than this number to kill bias in efficinecy.
        max_peaks (int, optional): For sake of computation, we cannot align too many peaks at once...
    """
    peaks = peaks[(peaks["dt"] == dt) & (peaks["area"] > min_area)]
    peaks = peaks[: min(len(peaks), max_peaks)]
    n_peaks = len(peaks)
    time_ns = np.arange(10 * len(peaks[0]["data"]))
    time_10ns = np.arange(len(peaks[0]["data"])) * 10
    interped_wfs = np.zeros((len(peaks), 10 * len(peaks[0]["data"])))
    normalized = peaks["data"] / peaks["area"][:, np.newaxis]
    f = interpolate.interp1d(
        time_10ns, normalized, axis=1, fill_value=(0, 0), bounds_error=False
    )

    interped_wfs = f(time_ns)
    interped_wfs = interped_wfs / np.sum(interped_wfs, axis=1)[:, np.newaxis]

    # Get first peak and align according to maximum.
    p1 = interped_wfs[0]
    start_index = 3000 - np.argmax(p1)

    res = np.zeros((n_peaks, 6000))
    res[0][start_index : start_index + len(p1)] += p1
    for i in range(1, n_peaks):
        p2 = interped_wfs[i]
        template = np.mean(res[:i], axis=0)
        corr = np.correlate(template, p2)
        shift = np.argmax(corr)
        res[i][shift : shift + len(p2)] = p2

    result = np.zeros_like(interped_wfs)
    result = res[:, 3000 - align_at : 3000 - align_at + len(interped_wfs)]

    return result[:, : 10 * len(peaks[0]["data"])]


def get_avgwf(
    peaks,
    method="first_phr",
    dt=10,
    min_area=18,
    n_slices=10,
    align_at=500,
    plot=True,
    xlims=False,
):
    assert (
        method in METHODS
    ), "Please use one of the following alignment techniques: %s" % (METHODS.keys())
    peaks = peaks[(peaks["dt"] == dt) & (peaks["area"] > min_area)]
    z_slices = np.linspace(np.min(peaks["z"]), np.max(peaks["z"]), n_slices + 1)

    avg_wf_mean = np.zeros((n_slices, 10 * len(peaks[0]["data"])))
    avg_wf_err = np.zeros((n_slices, 10 * len(peaks[0]["data"])))

    for i in range(len(z_slices) - 1):
        z_mask = (peaks["z"] >= z_slices[i]) & (peaks["z"] <= z_slices[i + 1])
        if method == "first_phr":
            aligned_wfs = align_first_phr(
                peaks[z_mask], dt=dt, min_area=min_area, align_at=align_at
            )
        elif method == "area_range":
            aligned_wfs = align_area_range(
                peaks[z_mask], dt=dt, min_area=min_area, align_at=align_at
            )
        elif method == "self_adjusted":
            aligned_wfs = align_self_adjusted(
                peaks[z_mask], dt=dt, min_area=min_area, align_at=align_at
            )

        avg_wf_mean[i] = np.mean(aligned_wfs, axis=0)
        normalization = np.sum(avg_wf_mean[i])
        avg_wf_mean[i] = avg_wf_mean[i] / normalization
        avg_wf_err[i] = np.std(aligned_wfs, axis=0) / normalization

    if plot:
        import matplotlib as mpl

        plt.figure(dpi=200)
        colors = plt.get_cmap("jet", 10 * n_slices + 1)
        for i in range((len(z_slices) - 1)):
            plt.plot(
                np.arange(10 * len(peaks[0]["data"])),
                avg_wf_mean[i],
                color=colors(1 + i * 10),
                alpha=0.3,
                linewidth=1,
            )

        norm = mpl.colors.Normalize(vmin=z_slices[0], vmax=z_slices[-1])
        sm1 = plt.cm.ScalarMappable(cmap=colors, norm=norm)
        sm1.set_array([])
        cb1 = plt.colorbar(sm1)
        cb1.set_label("depth [cm]")
        plt.xlabel("time [ns]")
        plt.title("Average waveform at different positions [method=%s]" % (method))
        plt.xlim(max(0, align_at - 100), max(0, align_at - 100) + 600)
        plt.grid()
        if xlims:
            plt.xlim(xlims[0], xlims[1])
        else:
            plt.xlim(align_at - 100, align_at + 500)
        plt.show()

    return avg_wf_mean, avg_wf_err


def shift_avg_wfs(wf0_dt1, wf1_dt1, wf2_dt1):
    """Align three average waveforms to plot, based on mean time.
    Assumed wf0_dt1 is always leftmost. The returns will usually reduce the length of waveforms.

    Args:
        wf0_dt1 (1darray): waveform with dt=1ns. Usually wfsim original template.
        wf1_dt1 (1darray): waveform with dt=1ns. Usually data reconstructed with alignment.
        wf2_dt1 (1darray): waveform with dt=1ns. Usually wfsim reconstructed with alignment.
    """
    xs0_dt1 = np.arange(len(wf0_dt1))
    xs1_dt1 = np.arange(len(wf1_dt1))
    xs2_dt1 = np.arange(len(wf2_dt1))

    init_mean_wf0 = np.sum(xs0_dt1 * wf0_dt1)
    init_mean_wf1 = np.sum(xs1_dt1 * wf1_dt1)
    init_mean_wf2 = np.sum(xs2_dt1 * wf2_dt1)

    # moving one sample, how much will the mean time be changed
    d_shift0 = 1
    d_shift1 = 1
    d_shift2 = 1

    # moving wf1 to first place left to wf0
    d_sample1 = int((init_mean_wf1 - init_mean_wf0) / d_shift1) + 1
    assert d_sample1 >= 0
    mean_wf1 = init_mean_wf1 - d_sample1 * d_shift1
    wf1_dt1 = wf1_dt1[d_sample1:]

    # moving wf2 to anywhere closest to wf1
    d_sample2 = int(np.around((init_mean_wf2 - mean_wf1) / d_shift2))
    assert d_sample2 >= 0
    mean_wf2 = init_mean_wf2 - d_sample2 * d_shift2
    wf2_dt1 = wf2_dt1[d_sample2:]

    # moving wf0 to anywhere closest to wf2
    d_sample0 = int(np.around((init_mean_wf0 - mean_wf2) / d_shift0))
    d_sample0 = max(d_sample0, 0)
    wf0_dt1 = wf0_dt1[d_sample0:]

    xs1 = np.arange(len(wf1_dt1))
    xs2 = np.arange(len(wf2_dt1))

    return wf0_dt1, wf1_dt1, wf2_dt1


def shift_avg_wf(wf1_dt1, wf2_dt1, align_at=190):
    """Align two average waveforms to plot, based on mean time.
    The returns will usually reduce the length of waveforms.

    Args:
        wf1_dt1 (1darray): waveform with dt=10ns. Usually data reconstructed with alignment.
        wf2_dt1 (1darray): waveform with dt=10ns. Usually wfsim reconstructed with alignment.
        align_at (float): align point of mean time in unit of ns.
    """
    xs1_dt1 = np.arange(len(wf1_dt1))
    xs2_dt1 = np.arange(len(wf2_dt1))

    init_mean_wf1 = np.sum(xs1_dt1 * wf1_dt1)
    init_mean_wf2 = np.sum(xs2_dt1 * wf2_dt1)

    # moving one sample, how much will the mean time be changed
    d_shift1 = 10
    d_shift2 = 10

    # moving wf1 to first place left to align point
    d_sample1 = int((init_mean_wf1 - align_at) / d_shift1) + 1
    assert d_sample1 >= 0
    mean_wf1 = init_mean_wf1 - d_sample1 * d_shift1
    wf1_dt1 = wf1_dt1[d_sample1:]

    # moving wf2 to anywhere closest to wf1
    d_sample2 = int(np.around((init_mean_wf2 - mean_wf1) / d_shift2))
    assert d_sample2 >= 0
    mean_wf2 = init_mean_wf2 - d_sample2 * d_shift2
    wf2_dt1 = wf2_dt1[d_sample2:]

    return wf1_dt1, wf2_dt1
