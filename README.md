# compeaks
[Wiki notes](https://xe1t-wiki.lngs.infn.it/doku.php?id=lanqing:wfsim_data_peak_level_s1matching)
## Scope
Systematic comparison between peaks level data. 
<img width="1403" alt="Screen Shot 2022-02-07 at 2 45 58 AM" src="https://user-images.githubusercontent.com/47046530/152754293-332b4772-2bb0-44ca-9c57-c42805789c44.png">

## Usage
Specify the optical maps, then choose signal types to compare and alignment method. Comparison plots will be generated automatically.
```
s1_pattern_map = '/dali/lgrandi/xenonnt/simulations/optphot/mc_v4.1.0/S1_1.69_0.99_0.99_0.99_0.99_10000_100_30/XENONnT_S1_xyz_patterns_LCE_corrected_QEs_MCv4.1.0_wires.pkl'
s1_time_spline ='XENONnT_s1_proponly_va43fa9b_wires_20200625.json.gz'

comparison.sr0_auto_plots(signal_type = ['ArS1', 'KrS1B', 'KrS1A'], method = 'first_phr',
                          s1_pattern_map = s1_pattern_map, 
                          s1_time_spline = s1_time_spline)
```
