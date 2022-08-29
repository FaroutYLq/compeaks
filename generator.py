# By Andrii Terliuk, April 2022
import numpy as np
import nestpy
import wfsim
from packaging import version


def generate_vertex(r_range=(0, 66.4), z_range=(-148.15, 0), size=1):
    phi = np.random.uniform(size=size) * 2 * np.pi
    r = r_range[1] * np.sqrt(
        np.random.uniform((r_range[0] / r_range[1]) ** 2, 1, size=size)
    )
    z = np.random.uniform(z_range[0], z_range[1], size=size)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z


def generate_times(rate, size, timemode):

    # generating event times from exponential
    if timemode == "realistic":
        dt = np.random.exponential(1 / rate, size=size - 1)
        times = np.append([1.0], 1.0 + dt.cumsum()) * 1e9
        times = times.round().astype(np.int64)
        return times
    elif timemode == "uniform":
        dt = (1 / rate) * np.ones(size - 1)
        times = np.append([1.0], 1.0 + dt.cumsum()) * 1e9
        times = times.round().astype(np.int64)
        return times


def generator_Ar37(
    n_tot=1000,
    rate=20.0,
    fmap=None,
    nc=None,
    r_range=(0, 66.4),
    z_range=(-148.15, 0),
    timemode="realistic",
):

    """
    Function that generates given number of events for a given rate.
    """
    times = generate_times(rate=rate, size=n_tot, timemode=timemode)
    # generating instructions with 2x size to account S1 and S2
    instr = np.zeros(2 * n_tot, dtype=wfsim.instruction_dtype)
    instr["event_number"] = np.arange(1, n_tot + 1).repeat(2)
    instr["type"][:] = np.tile([1, 2], n_tot)
    instr["time"][:] = times.repeat(2)
    # generating unoformely distributed events for give R and Z range
    x, y, z = generate_vertex(r_range=r_range, z_range=z_range, size=n_tot)
    instr["x"][:] = x.repeat(2)
    instr["y"][:] = y.repeat(2)
    instr["z"][:] = z.repeat(2)
    # Setting properties - energy 2.82 keV and recoil of gamma from NEST
    instr["recoil"][:] = 7
    instr["e_dep"][:] = 2.82
    # getting local field from field map
    instr["local_field"] = fmap(np.array([np.sqrt(x**2 + y**2), z]).T).repeat(2)
    # And generating quantas from nest
    for i in range(0, n_tot):
        y = nc.GetYields(
            interaction=nestpy.INTERACTION_TYPE(instr["recoil"][2 * i]),
            energy=instr["e_dep"][2 * i],
            drift_field=instr["local_field"][2 * i],
        )
        q_ = nc.GetQuanta(y)
        instr["amp"][2 * i] = q_.photons
        instr["amp"][2 * i + 1] = q_.electrons
        instr["n_excitons"][2 * i : 2 * (i + 1)] = q_.excitons
    return instr


def generator_flat(
    en_range=(0, 30.0),
    recoil=7,
    n_tot=1000,
    rate=20.0,
    fmap=None,
    nc=None,
    r_range=(0, 66.4),
    z_range=(-148.15, 0),
    mode="all",
    timemode="realistic",
):
    times = generate_times(rate=rate, size=n_tot, timemode=timemode)
    instr = np.zeros(2 * n_tot, dtype=wfsim.instruction_dtype)
    instr["event_number"] = np.arange(1, n_tot + 1).repeat(2)
    instr["type"][:] = np.tile([1, 2], n_tot)
    instr["time"][:] = times.repeat(2)
    # generating unoformely distributed events for give R and Z range
    x, y, z = generate_vertex(r_range=r_range, z_range=z_range, size=n_tot)
    instr["x"][:] = x.repeat(2)
    instr["y"][:] = y.repeat(2)
    instr["z"][:] = z.repeat(2)
    # making energy
    ens = np.random.uniform(en_range[0], en_range[1], size=n_tot)
    instr["recoil"][:] = recoil
    instr["e_dep"][:] = ens.repeat(2)
    # getting local field from field map
    instr["local_field"] = fmap(np.array([np.sqrt(x**2 + y**2), z]).T).repeat(2)
    # And generating quantas from nest
    for i in range(0, n_tot):
        y = nc.GetYields(
            interaction=nestpy.INTERACTION_TYPE(instr["recoil"][2 * i]),
            energy=instr["e_dep"][2 * i],
            drift_field=instr["local_field"][2 * i],
        )
        q_ = nc.GetQuanta(y)
        instr["amp"][2 * i] = q_.photons
        instr["amp"][2 * i + 1] = q_.electrons
        instr["n_excitons"][2 * i : 2 * (i + 1)] = q_.excitons
    if mode == "s1":
        instr = instr[instr["type"] == 1]
    elif mode == "s2":
        instr = instr[instr["type"] == 2]
    elif mode == "all":
        pass
    else:
        raise RuntimeError("Unknown mode: ", mode)
    return instr


def generator_Kr83m(
    n_tot=1000,
    recoil=11,
    rate=30.0,
    fmap=None,
    nc=None,
    r_range=(0, 64),
    z_range=(-142, -6),
    mode="all",
    timemode="realistic",
    filterzero=True,
):
    """
    Generator function for Kr83m with 2 consequtive energy deposits
    of 32.1 and 9.4 keV

    """
    times = generate_times(rate=rate, size=n_tot, timemode=timemode)
    instr = np.zeros(4 * n_tot, dtype=wfsim.instruction_dtype)
    instr["event_number"] = np.arange(1, n_tot + 1).repeat(4)
    instr["time"][:] = times.repeat(4)
    instr["type"][:] = np.tile([1, 2, 1, 2], n_tot)
    dTs = np.random.exponential(scale=154.4 / np.log(2), size=n_tot)
    dTs_add = np.stack([np.zeros_like(dTs), dTs]).T
    dTs_add = dTs_add.repeat(2)
    instr["time"] += dTs_add.round().astype(np.int64)
    # generating unoformely distributed events for give R and Z range
    x, y, z = generate_vertex(r_range=r_range, z_range=z_range, size=n_tot)
    instr["x"][:] = x.repeat(4)
    instr["y"][:] = y.repeat(4)
    instr["z"][:] = z.repeat(4)
    instr["e_dep"][:] = np.tile(np.array([32.1, 32.1, 9.4, 9.4]), n_tot)
    # getting local field from field map
    instr["local_field"] = fmap(np.array([np.sqrt(x**2 + y**2), z]).T).repeat(4)
    instr["recoil"][:] = recoil
    ####
    if recoil == 11:
        if version.parse(nestpy.__version__) < version.parse("1.5.4"):
            print(
                "WARNING! Using nestpy before 1.5.4, NEST yields for 9 and 41 keV lines are not reliable!"
            )
        for i in range(0, n_tot):
            assert instr["type"][4 * i] == 1
            assert instr["type"][4 * i + 1] == 2
            y_ = nc.GetYieldKr83m(
                energy=instr["e_dep"][4 * i],
                drift_field=instr["local_field"][4 * i],
                maxTimeSeparation=dTs[i],
                minTimeSeparation=dTs[i],
            )
            q_ = nc.GetQuanta(y_)
            instr["amp"][4 * i] = q_.photons
            instr["amp"][4 * i + 1] = q_.electrons
            instr["n_excitons"][4 * i : 4 * i + 2] = q_.excitons
            ####
            assert instr["type"][4 * i + 2] == 1
            assert instr["type"][4 * i + 3] == 2
            y_ = nc.GetYieldKr83m(
                energy=instr["e_dep"][4 * i + 2],
                drift_field=instr["local_field"][4 * i + 2],
                maxTimeSeparation=dTs[i],
                minTimeSeparation=dTs[i],
            )
            q_ = nc.GetQuanta(y_)
            instr["amp"][4 * i + 2] = q_.photons
            instr["amp"][4 * i + 3] = q_.electrons
            instr["n_excitons"][4 * i + 2 : 4 * i + 4] = q_.excitons

        # raise NotImplemented("ERROR! Kr83m quanta generation in NEST is not implemented yet")
    else:
        for i in range(0, 2 * n_tot):
            y_ = nc.GetYields(
                interaction=nestpy.INTERACTION_TYPE(instr["recoil"][2 * i]),
                energy=instr["e_dep"][2 * i],
                drift_field=instr["local_field"][2 * i],
            )
            q_ = nc.GetQuanta(y_)
            instr["amp"][2 * i] = q_.photons
            instr["amp"][2 * i + 1] = q_.electrons
            instr["n_excitons"][2 * i : (2 * i + 2)] = q_.excitons
    instr = instr[np.argsort(instr["time"])]
    if filterzero:
        instr = instr[instr["amp"] > 0]
    return instr
