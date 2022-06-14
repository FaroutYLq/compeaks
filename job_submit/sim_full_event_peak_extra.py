import strax
import straxen
import numpy as np
import sys
sys.path.append('/home/yuanlq/xenon/compeaks')
import comparison

print("Finished importing.")
print(straxen.print_versions(['strax','straxen','cutax','wfsim','pema','nestpy', 'pandas']))

_, runid = sys.argv

print('runid: ', runid)
sim_peak_extra = comparison.get_peak_extra(signal_type='sim_KrS1A', runid=runid) # signal_type doesn't matter.
print('Done')