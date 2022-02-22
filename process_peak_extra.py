import strax
import straxen
import numpy as np
import sys
sys.path.append('/home/yuanlq/xenon/compeaks')
from extraction import get_peak_extra

print("Finished importing.")
print(straxen.print_versions())

_, runid, signal = sys.argv

print("Loaded the context successfully, and the run id to process:", runid)

get_peak_extra(runs=runid, signal_type=signal, 
               version='xenonnt_v6',
               output_folder='/project2/lgrandi/yuanlq/xenonnt/')

print('Done')