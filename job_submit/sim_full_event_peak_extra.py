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
sim_peak_extra = comparison.get_peak_extra(signal_type='sim_KrS1A', runid=runid, 
                    s1_pattern_map = 'XENONnT_s1_xyz_patterns_LCE_MCvf051911_wires.pkl', 
                    s1_time_spline = '/project2/lgrandi/yuanlq/shared/s1_optical/XENONnT_s1_proponly_pc_reflection_optPhot_perPMT_S1_local_20220510.json.gz') # signal_type doesn't matter.
print('Done')