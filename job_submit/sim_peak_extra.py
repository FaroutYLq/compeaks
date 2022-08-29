import strax
import straxen
import numpy as np
import sys

sys.path.append("/home/yuanlq/xenon/compeaks")
import simwrap

print("Finished importing.")
print(straxen.print_versions())

_, runid, interaction_type, energy, N = sys.argv

print("runid: ", runid)
print("type: ", interaction_type)
print("energy: ", energy)
print("N: ", N)

sim_ambe_peak_extra = simwrap.get_sim_peak_extra(
    runid=runid, interaction_type=eval(interaction_type), energy=eval(energy), N=eval(N)
)

print("Done")
