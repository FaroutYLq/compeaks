#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=3G
#SBATCH --output=/home/yuanlq/logs/MClogs/histogram_G4.%A_%a.log
#SBATCH --account=pi-lgrandi
#SBATCH --partition=dali
#SBATCH --qos=dali
#SBATCH --job-name=hist_photons_G4
OPTICS=$1
eval "$(/dali/lgrandi/strax/miniconda3/bin/conda shell.bash hook)"
source activate strax
# originally I used MC image
#   singimage=/project2/lgrandi/xenonnt/singularity-images/xenonnt-montecarlo-development.simg

if [ "$SLURM_ARRAY_TASK_ID" = "" ]; then
    echo "No SLURM ID, job is run interactively, assume 1"
    FILE_NR=0
    SLURM_ARRAY_TASK_ID=1
else
    FILE_NR=`expr $SLURM_ARRAY_TASK_ID - 1`
fi
echo "File number : " $FILE_NR


pyscript=/home/yuanlq/xenon/analysiscode/s1_pulse_shape/generate_histograms.py
echo "Python script : " $pyscript
echo "Optical config" : $OPTICS
###
curdir=$PWD
rundir=/dali/lgrandi/terliuk/posrec_patterns/run_folder/
echo "Current dir : " $curdir
echo "Changing to run folder  : " $rundir
cd $rundir
###
python $pyscript -r $FILE_NR -p $OPTICS -n 50
cd $curdir