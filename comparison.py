import numpy as np
import sys
import matplotlib.pyplot as plt
import sims1
import simwrap
import alignment
import extraction
import gc
from simwrap import SIM_DATA_PATH
from simwrap import SIM_CONTEXT

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

SUPPORTED_SIGNALS = [
    "KrS1A",
    "KrS1B",
    "ArS1",
    "sim_KrS1A",
    "sim_KrS1B",
    "sim_ArS1",
    "sim_AmBe",
]
ENERGY_DEPOSIT = {
    "sim_KrS1A": 32.1,
    "sim_KrS1B": 9.4,
    "sim_ArS1": 2.5,
    "sim_AmBe": [2, 20],
}  # KrS1B set to 9.35 instead of 9.4 to bypass weird nest feature https://github.com/NESTCollaboration/nestpy/blob/cddb082d4c69d87831872c67fc1fd48bebb8d28f/src/nestpy/NEST.cpp#L922
INTERACTION_TYPES = {"sim_KrS1A": 11, "sim_KrS1B": 11, "sim_ArS1": 7, "sim_AmBe": 4}
DEFAULT_SIM_RUNS = {
    "sim_KrS1A": "kr83ms1a_t1",
    "sim_KrS1B": "kr83ms1b_t1",
    "sim_ArS1": "ar37s1_t1",
    "sim_AmBe": "ambes1_t1",
}
"""
DEFAULT_SIM_FULL_EVENT_RUNS = np.array(['sim_KrS1_full_event_00', 'sim_KrS1_full_event_01', 'sim_KrS1_full_event_02', 
                                        'sim_KrS1_full_event_03', 'sim_KrS1_full_event_04', 'sim_KrS1_full_event_05',
                                        'sim_KrS1_full_event_06', 'sim_KrS1_full_event_07', 'sim_KrS1_full_event_08',
                                        'sim_KrS1_full_event_09', 'sim_KrS1_full_event_10', 'sim_KrS1_full_event_11', 
                                        'sim_KrS1_full_event_12', 'sim_KrS1_full_event_13', 'sim_KrS1_full_event_14', 
                                        'sim_KrS1_full_event_15', 'sim_KrS1_full_event_16', 'sim_KrS1_full_event_17', 
                                        'sim_KrS1_full_event_18', 'sim_KrS1_full_event_19'])
"""
DEFAULT_SIM_FULL_EVENT_RUNS = np.array(
    [
        "sim_KrS1_full_event_00",
        "sim_KrS1_full_event_02",
        "sim_KrS1_full_event_03",
        "sim_KrS1_full_event_04",
        "sim_KrS1_full_event_06",
        "sim_KrS1_full_event_07",
        "sim_KrS1_full_event_08",
        "sim_KrS1_full_event_09",
        "sim_KrS1_full_event_10",
        "sim_KrS1_full_event_11",
        "sim_KrS1_full_event_12",
        "sim_KrS1_full_event_13",
        "sim_KrS1_full_event_14",
        "sim_KrS1_full_event_16",
        "sim_KrS1_full_event_17",
        "sim_KrS1_full_event_18",
        "sim_KrS1_full_event_19",
    ]
)
COMPARISON_SPACES2D = [
    ("z", "area_fraction_top"),
    ("z", "rise_time"),
    ("z", "range_50p_area"),
    ("z", "range_90p_area"),
    ("z", "area"),
    ("z", "tight_coincidence"),
    ("area_fraction_top", "rise_time"),
    ("area", "range_50p_area"),
    ("area", "rise_time"),
]
COMPARISON_SPACES1D = [
    "z",
    "area",
    "area_fraction_top",
    "range_50p_area",
    "range_90p_area",
    "rise_time",
    "n_channels",
    "tight_coincidence",
    "n_hits",
]

ZSLIACES = np.array([-128, -116, -104, -92, -79, -67, -55, -43, -31, -19])


def get_peak_extra(
    signal_type, runid=False, sim_full_Kr_event=True, straxen_config={}, **kargs
):
    """Wrapper around data/wfsim peak_extra getter.

    Args:
        signal_type (str): examples: ['KrS1A', 'KrS1B', 'ArS1', 'sim_KrS1A', 'sim_KrS1B', 'sim_ArS1', 'sim_AmBe']
        sim_full_Kr_event (bool, optional): wfsim method, either single peak or full event. For Krypton matching, we are encouraged to use 'full event'.
        **kargs: keyword arguements for fax_config_overide applied to simulation.

    Returns:
        (ndarray): peak extra for simulation or data of a specific calibration source.
    """
    assert (
        signal_type in SUPPORTED_SIGNALS
    ), "Input signal type not supported yet. Please try one of the following: %s" % (
        SUPPORTED_SIGNALS
    )
    # data
    print("Loading peak_extra now, please be patient...")
    if signal_type[:3] != "sim":
        print("Loading peak extra from data")
        if type(runid) == bool:
            print("Loading default runs from data")
            peak_extra = extraction.get_peak_extra_from_events(signal_type=signal_type)
        else:
            print("Loading selected runs from data")
            peak_extra = extraction.get_peak_extra_from_events(
                signal_type=signal_type, runs=runid
            )
    # wfsim
    else:
        print("Loading peak extra from wfsim")
        if type(runid) == bool:
            print("Loading default runs from sim")
            if sim_full_Kr_event and ("Kr" in signal_type):
                print("Simulating full events for Kryptons")
                peak_extra = extraction.get_peak_extra_from_events(
                    signal_type=signal_type,
                    runs=DEFAULT_SIM_FULL_EVENT_RUNS,
                    version=SIM_CONTEXT,
                    output_folder=SIM_DATA_PATH,
                    interaction_type=11,
                    energy=41.5,
                    N=30000,
                    **kargs
                )
            else:
                print("Simulating single peaks for %s" % (signal_type))
                peak_extra = simwrap.get_sim_peak_extra(
                    runid=DEFAULT_SIM_RUNS[signal_type],
                    interaction_type=INTERACTION_TYPES[signal_type],
                    energy=ENERGY_DEPOSIT[signal_type],
                    **kargs
                )
        else:
            print("Loading selected runs from sim")
            if sim_full_Kr_event and ("Kr" in signal_type):
                print("Simulating full events for Kryptons")
                peak_extra = extraction.get_peak_extra_from_events(
                    signal_type=signal_type,
                    runs=runid,
                    version=SIM_CONTEXT,
                    output_folder=SIM_DATA_PATH,
                    interaction_type=11,
                    energy=41.5,
                    N=30000,
                    **kargs
                )
            else:
                print("Simulating single peaks for %s" % (signal_type))
                peak_extra = simwrap.get_sim_peak_extra(
                    runid=runid,
                    interaction_type=INTERACTION_TYPES[signal_type],
                    energy=ENERGY_DEPOSIT[signal_type],
                    **kargs
                )
    return peak_extra


def get_avgwfs(
    peak_extra,
    signal_type,
    method="first_phr",
    xlims=(400, 900),
    spline_file=None,
    pattern_map_file=None,
    plot=False,
):
    """Wrapper around data/wfsim average waveform getter.

    Args:
        peak_extra (ndarray): peak extra for simulation or data of a specific calibration source.
        signal_type (str): examples: ['KrS1A', 'KrS1B', 'ArS1', 'sim_KrS1A', 'sim_KrS1B', 'sim_ArS1', 'sim_AmBe']
        method (str, optional): alignment technique. For example: {'first_phr', 'area_range', 'self_adjusted'}. Defaults to 'first_phr'.
        xlims (tuple, optional): range of plot of the average waveforms. Defaults to (40,90).
        spline_file: pointer to s1 optical propagation splines from resources.
        pattern_map_file (str): path to map.
        plot (bool): whether to show the plots for average waveforms?

    Returns:
        wfsim_template (ndarray, optional): Will be returned only if signal type is sim. Analytic S1 template in wfsim. axis0 = depth, axis1 = wf samples.
        avg_wf_mean (ndarray): Will be returned for all cases. Average waveform in different depth, aligned by specified techniques. axis0 = depth, axis1 = wf samples.
    """
    if signal_type[:3] == "sim":
        print("Computing analytic %s wfsim S1 template..." % (signal_type))
        if spline_file != None and pattern_map_file != None:
            wfsim_template = sims1.get_s1_templates(
                interaction_type=INTERACTION_TYPES[signal_type],
                e_dep=ENERGY_DEPOSIT[signal_type],
                spline_file=spline_file,
                pattern_map_file=pattern_map_file,
                plot=plot,
            )
        else:
            # use default maps
            wfsim_template = sims1.get_s1_templates(
                interaction_type=INTERACTION_TYPES[signal_type],
                e_dep=ENERGY_DEPOSIT[signal_type],
                plot=plot,
            )

        wfsim_template = wfsim_template / np.sum(wfsim_template, axis=1)[:, np.newaxis]
        print(
            "Computing aligned reconstucted wfsim %s average waveform with method %s..."
            % (signal_type, method)
        )
        avg_wf_mean, _ = alignment.get_avgwf(
            peak_extra, method=method, xlims=xlims, plot=plot
        )
        avg_wf_mean = avg_wf_mean / np.sum(avg_wf_mean, axis=1)[:, np.newaxis]
        return wfsim_template, avg_wf_mean

    else:
        print(
            "Computing aligned reconstucted data %s average waveform with method %s..."
            % (signal_type, method)
        )
        avg_wf_mean, _ = alignment.get_avgwf(
            peak_extra, method=method, xlims=xlims, plot=plot
        )
        avg_wf_mean = avg_wf_mean / np.sum(avg_wf_mean, axis=1)[:, np.newaxis]
        return avg_wf_mean


def compute_rise_time(wf):
    """Compute the rise time based on average waveform.

    Args:
        wf (1darray): average waveform in a 1d vector

    Returns:
        (int): number of sample characterizing rise time.
    """
    cdf = np.cumsum(wf)
    ind_10 = np.argmin(abs(cdf - 0.1 * cdf[-1]))
    ind_50 = np.argmin(abs(cdf - 0.5 * cdf[-1]))
    return ind_50 - ind_10


def compare_avgwfs(
    signal_type0, signal_type1, avg_wf_mean0, avg_wf_mean1, method, wfsim_template=False
):
    """Plot generator for average pulse shape comparison at certain Z slices.

    Args:
        signal_type0 (str): Please put data here if you want to involve data in comparison! examples: ['KrS1A', 'KrS1B', 'ArS1', 'sim_KrS1A', 'sim_KrS1B', 'sim_ArS1', 'sim_AmBe']
        signal_type1 (str): Please put wfsim here if you want to involve wfsim comparison! examples: ['KrS1A', 'KrS1B', 'ArS1', 'sim_KrS1A', 'sim_KrS1B', 'sim_ArS1', 'sim_AmBe']
        avg_wf_mean0 (ndarray): Average waveform in different depth, aligned by specified techniques. axis0 = depth, axis1 = wf samples.
        avg_wf_mean1 (ndarray): Average waveform in different depth, aligned by specified techniques. axis0 = depth, axis1 = wf samples.
        method (str, optional): alignment technique. For example: {'first_phr', 'area_range', 'self_adjusted'}. Defaults to 'first_phr'.
        wfsim_template (ndarray, optional): Analytic S1 template in wfsim. axis0 = depth, axis1 = wf samples.
    """
    is_sim0 = False
    is_sim1 = False
    if signal_type0[:3] == "sim":
        is_sim0 = True
    if signal_type1[:3] == "sim":
        is_sim1 = True

    # Comparing data VS data
    print(
        "Now comparing %s and %s average waveforms at different depth..."
        % (signal_type0, signal_type1)
    )
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(15, 15), dpi=400)
    for i in range(10):
        j = i // 4

        if is_sim0 == False and is_sim1 == False:
            wf0, wf1 = alignment.shift_avg_wf(avg_wf_mean0[i], avg_wf_mean1[i])

            axs[j, i - 4 * j].plot(
                np.arange(len(wf0)),
                wf0,
                label=signal_type0 + ":" + str(compute_rise_time(wf0)) + "ns",
            )
            axs[j, i - 4 * j].plot(
                np.arange(len(wf1)),
                wf1,
                label=signal_type1 + ":" + str(compute_rise_time(wf1)) + "ns",
            )
            axs[j, i - 4 * j].legend()
            if signal_type1[:3] == "sim" or signal_type0[:3] == "sim":
                axs[j, i - 4 * j].set_xlim(0, 600)
            else:
                axs[j, i - 4 * j].set_xlim(300, 900)
            axs[j, i - 4 * j].grid()
            axs[j, i - 4 * j].set_xlabel("time [ns]")
            axs[j, i - 4 * j].set_title("%s at %scm" % (method, ZSLIACES[i]))

        else:
            assert (
                type(wfsim_template) != bool
            ), "You are comparing data and wfsim, you must input a wfsim template for reference"
            template, wf0, wf1 = alignment.shift_avg_wfs(
                wf0_dt1=wfsim_template[i],
                wf1_dt1=avg_wf_mean0[i],
                wf2_dt1=avg_wf_mean1[i],
            )
            axs[j, i - 4 * j].plot(
                np.arange(len(template)), template, label="%s template" % (signal_type1)
            )
            axs[j, i - 4 * j].plot(
                np.arange(len(wf0)),
                wf0,
                label=signal_type0 + ":" + str(compute_rise_time(wf0)) + "ns",
            )
            axs[j, i - 4 * j].plot(
                np.arange(len(wf1)),
                wf1,
                label=signal_type1 + ":" + str(compute_rise_time(wf1)) + "ns",
            )
            axs[j, i - 4 * j].legend()
            axs[j, i - 4 * j].grid()
            if signal_type1[:3] == "sim" or signal_type0[:3] == "sim":
                axs[j, i - 4 * j].set_xlim(0, 600)
            else:
                axs[j, i - 4 * j].set_xlim(300, 900)
            axs[j, i - 4 * j].set_xlabel("time [ns]")
            axs[j, i - 4 * j].set_title("%s at %scm" % (method, ZSLIACES[i]))

    fig.suptitle("%s VS %s" % (signal_type0, signal_type1), fontsize=25, y=0.93)
    fig.show()


def compare_2para(
    peak_extra0,
    peak_extra1,
    signal_type0,
    signal_type1,
    comparison_spaces=COMPARISON_SPACES2D,
    errorbar="std",
):
    """Compare peak_extra in 2D parameter spaces you specified.

    Args:
        peak_extra0 (ndarray): peak extra for simulation or data of a specific calibration source.
        peak_extra1 (ndarray): peak extra for simulation or data of a specific calibration source.
        signal_type0 (str): Please put data here if you want to involve data in comparison! examples: ['KrS1A', 'KrS1B', 'ArS1', 'sim_KrS1A', 'sim_KrS1B', 'sim_ArS1', 'sim_AmBe']
        signal_type1 (str): Please put wfsim here if you want to involve wfsim comparison! examples: ['KrS1A', 'KrS1B', 'ArS1', 'sim_KrS1A', 'sim_KrS1B', 'sim_ArS1', 'sim_AmBe']
        comparison_spaces (array-like, optional): axis0=parameter spaces, axis1=parameter names. Defaults to COMPARISON_SPACES2D.
        errorbar (str): what error bar to show? "std" for standard deviation in each slice, "mean_error" for error estimated for mean. Default to be "std".
    """
    assert errorbar in ["std", "mean_error"]
    if errorbar == "std":
        sigma_mu = False
    elif errorbar == "mean_error":
        sigma_mu = True

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), dpi=300)
    for i, space in enumerate(comparison_spaces):
        j = i // 3

        if space == ("z", "area_normalized"):
            compare2d(
                x1s=peak_extra0[space[0]],
                y1s=peak_extra0["area"],
                x2s=peak_extra1[space[0]],
                y2s=peak_extra1["area"],
                n_x=20,
                xlabel=space[0],
                ylabel=space[1],
                label1=signal_type0,
                label2=signal_type1,
                sigma_mu=sigma_mu,
                ax=axs[j, i - j * 3],
            )
        else:
            compare2d(
                x1s=peak_extra0[space[0]],
                y1s=peak_extra0[space[1]],
                x2s=peak_extra1[space[0]],
                y2s=peak_extra1[space[1]],
                n_x=20,
                xlabel=space[0],
                ylabel=space[1],
                label1=signal_type0,
                label2=signal_type1,
                sigma_mu=sigma_mu,
                ax=axs[j, i - j * 3],
            )

    fig.suptitle("%s VS %s" % (signal_type0, signal_type1), fontsize=25, y=0.93)
    fig.show()


def compare_1para(
    peak_extra0,
    peak_extra1,
    signal_type0,
    signal_type1,
    comparison_spaces=COMPARISON_SPACES1D,
    n_x=50,
):
    """Compare peak_extra in 2D parameter spaces you specified.

    Args:
        peak_extra0 (ndarray): peak extra for simulation or data of a specific calibration source.
        peak_extra1 (ndarray): peak extra for simulation or data of a specific calibration source.
        signal_type0 (str): Please put data here if you want to involve data in comparison! examples: ['KrS1A', 'KrS1B', 'ArS1', 'sim_KrS1A', 'sim_KrS1B', 'sim_ArS1', 'sim_AmBe']
        signal_type1 (str): Please put wfsim here if you want to involve wfsim comparison! examples: ['KrS1A', 'KrS1B', 'ArS1', 'sim_KrS1A', 'sim_KrS1B', 'sim_ArS1', 'sim_AmBe']
        comparison_spaces (array-like, optional): axis0=parameter spaces. Defaults to COMPARISON_SPACES2D.
        n_x (int): number of bins for the histograms.
    """
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), dpi=300)
    for i, space in enumerate(comparison_spaces):
        j = i // 3

        if space == "center":
            x1s = peak_extra0["center_time"] - peak_extra0["time"]
            x2s = peak_extra1["center_time"] - peak_extra1["time"]
        else:
            x1s = peak_extra0[space]
            x2s = peak_extra1[space]

        compare1d(
            x1s=x1s,
            x2s=x2s,
            n_x=n_x,
            xlabel=space,
            label1=signal_type0,
            label2=signal_type1,
            ax=axs[j, i - j * 3],
        )

    fig.suptitle("%s VS %s" % (signal_type0, signal_type1), fontsize=25, y=0.93)
    fig.show()


def compare1d(
    x1s,
    x2s,
    x_range=False,
    n_x=50,
    logx=False,
    title="",
    xlabel="",
    ylabel="normalized counts",
    label1="",
    label2="",
    x3s=False,
    ax=None,
):
    """1D parameter space comparison."""
    if not x_range:
        x_range = (min(np.min(x1s), np.min(x2s)), max(np.max(x1s), np.max(x2s)))

    x_bins = np.linspace(x_range[0], x_range[1], n_x + 1)
    x_bins_center = (x_bins[1:] + x_bins[:-1]) / 2
    x_bin_size = x_bins[1] - x_bins[0]
    x1_std = np.zeros(n_x)
    x2_std = np.zeros(n_x)
    x1_cnt = np.zeros(n_x)
    x2_cnt = np.zeros(n_x)
    x1_total_cnt = len(x1s)
    x2_total_cnt = len(x2s)

    for i in range(len(x_bins) - 1):
        mask1 = (x1s >= x_bins[i]) & (x1s <= x_bins[i + 1])
        mask2 = (x2s >= x_bins[i]) & (x2s <= x_bins[i + 1])
        x1_cnt[i] = len(x1s[mask1])
        x2_cnt[i] = len(x2s[mask2])
        x1_std[i] = np.sqrt(len(x1s[mask1]))
        x2_std[i] = np.sqrt(len(x2s[mask2]))

    ax.errorbar(
        x_bins_center, x1_cnt / x1_total_cnt, x1_std / x1_total_cnt, label=label1
    )
    # artificial horizontal shift to distinguish errorbars
    ax.errorbar(
        x_bins_center + 0.15 * x_bin_size,
        x2_cnt / x2_total_cnt,
        x2_std / x2_total_cnt,
        label=label2,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if not (label1 == "" and label2 == ""):
        ax.legend()
    if logx:
        ax.set_xscale("log")

    if x3s == False:
        pass
    elif x3s.any() and y3s.any():
        ax.plot(x3s, y3s, color="r")

    ax.grid()


def compare2d(
    x1s,
    y1s,
    x2s,
    y2s,
    x_range=False,
    y_range=False,
    n_x=20,
    logx=False,
    logy=False,
    sigma_mu=False,
    title="",
    xlabel="",
    ylabel="",
    label1="",
    label2="",
    x3s=False,
    y3s=False,
    ax=None,
):
    """2D parameter space comparison."""
    if not x_range:
        x_range = (min(np.min(x1s), np.min(x2s)), max(np.max(x1s), np.max(x2s)))
    if not y_range:
        y_range = (min(np.min(y1s), np.min(y2s)), max(np.max(y1s), np.max(y2s)))

    x_bins = np.linspace(x_range[0], x_range[1], n_x + 1)
    x_bins_center = (x_bins[1:] + x_bins[:-1]) / 2
    x_bin_size = x_bins[1] - x_bins[0]
    y1_avg = np.zeros(n_x)
    y2_avg = np.zeros(n_x)
    y1_std = np.zeros(n_x)
    y2_std = np.zeros(n_x)
    y1_cnt = np.zeros(n_x)
    y2_cnt = np.zeros(n_x)

    for i in range(len(x_bins) - 1):
        mask1 = (x1s >= x_bins[i]) & (x1s <= x_bins[i + 1])
        mask2 = (x2s >= x_bins[i]) & (x2s <= x_bins[i + 1])
        y1_avg[i] = np.mean(y1s[mask1])
        y2_avg[i] = np.mean(y2s[mask2])
        y1_std[i] = np.std(y1s[mask1])
        y2_std[i] = np.std(y2s[mask2])
        y1_cnt[i] = len(y1s[mask1])
        y2_cnt[i] = len(y2s[mask2])

    if ylabel == "area_normalized":
        # Put the first good samples at 1
        y1_normalizer = y1_avg[(y1_avg != 0) & (~np.isnan(y1_avg))][0]
        y2_normalizer = y2_avg[(y2_avg != 0) & (~np.isnan(y2_avg))][0]

        # overlay the curves at first good point in the shorter one
        if len(y2_avg[~np.isnan(y2_avg)]) <= len(y1_avg[~np.isnan(y1_avg)]):
            y2_normalizer /= (
                y1_avg[(y2_avg != 0) & (~np.isnan(y2_avg))][0] / y1_normalizer
            )
        else:
            y1_normalizer /= (
                y2_avg[(y1_avg != 0) & (~np.isnan(y1_avg))][0] / y2_normalizer
            )

        y1_std = y1_std / y1_normalizer
        y2_std = y2_std / y2_normalizer
        y1_avg = y1_avg / y1_normalizer
        y2_avg = y2_avg / y2_normalizer

    if sigma_mu:
        ax.errorbar(x_bins_center, y1_avg, y1_std / np.sqrt(y1_cnt), label=label1)
        # artificial horizontal shift to distinguish errorbars
        ax.errorbar(
            x_bins_center + 0.15 * x_bin_size,
            y2_avg,
            y2_std / np.sqrt(y2_cnt),
            label=label2,
        )
    else:
        ax.errorbar(x_bins_center, y1_avg, y1_std, label=label1)
        # artificial horizontal shift to distinguish errorbars
        ax.errorbar(x_bins_center + 0.15 * x_bin_size, y2_avg, y2_std, label=label2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if not (label1 == "" and label2 == ""):
        ax.legend()
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    if x3s == False and y3s == False:
        pass
    elif x3s.any() and y3s.any():
        ax.plot(x3s, y3s, color="r")

    ax.grid()


def sr0_auto_plots(
    signal_type=["ArS1", "KrS1B", "KrS1A"],
    method="area_range",
    sim_full_Kr_event=True,
    s1_pattern_map="XENONnT_s1_xyz_patterns_LCE_MCvf051911_wires.pkl",
    s1_time_spline="XENONnT_s1_proponly_pc_reflection_optPhot_perPMT_S1_local_20220510.json.gz",
    errorbar="mean_error",
    **kargs
):
    """Automatically generate comparison plots given optical maps.

    Args:
        signal_type (list, optional): please put data here if you want to involve data in comparison!. Defaults to ['ArS1', 'KrS1A'].
        method (str, optional): method (str, optional): alignment technique. For example: {'first_phr', 'area_range', 'self_adjusted'}. Defaults to 'first_phr'.
        sim_full_Kr_event (bool, optional): wfsim method, either single peak or full event. For Krypton matching, we are encouraged to use 'full event'.
        s1_pattern_map (str, optional): path to s1 pattern map from resources. Defaults to 'XENONnT_s1_xyz_patterns_LCE_MCvf051911_wires.pkl'.
        s1_time_spline (str, optional): path to s1 optical propagation splines from resources. Defaults to 'XENONnT_s1_proponly_va43fa9b_wires_20200625.json.gz'.
        **kargs: keyword arguements for fax_config_overide applied to simulation.
    """

    for sig_type in signal_type:
        gc.collect()
        print("Comparing %s" % (sig_type))
        sim_peak_extra = get_peak_extra(
            "sim_" + sig_type,
            s1_pattern_map=s1_pattern_map,
            s1_time_spline=s1_time_spline,
            sim_full_Kr_event=sim_full_Kr_event,
            **kargs
        )
        peak_extra = get_peak_extra(signal_type=sig_type)

        avg_wf_mean = get_avgwfs(
            peak_extra=peak_extra, signal_type=sig_type, method=method
        )
        template, sim_avg_wf_mean = get_avgwfs(
            peak_extra=sim_peak_extra,
            signal_type="sim_" + sig_type,
            method=method,
            spline_file=s1_time_spline,
            pattern_map_file=s1_pattern_map,
        )

        compare_avgwfs(
            signal_type0=sig_type,
            signal_type1="sim_" + sig_type,
            avg_wf_mean0=avg_wf_mean,
            avg_wf_mean1=sim_avg_wf_mean,
            method=method,
            wfsim_template=template,
        )

        compare_1para(
            peak_extra0=peak_extra,
            peak_extra1=sim_peak_extra,
            signal_type0=sig_type,
            signal_type1="sim_" + sig_type,
        )

        compare_2para(
            peak_extra0=peak_extra,
            peak_extra1=sim_peak_extra,
            signal_type0=sig_type,
            signal_type1="sim_" + sig_type,
            errorbar="mean_error",
        )
