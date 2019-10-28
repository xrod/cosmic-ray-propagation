import subprocess
import numpy as np

SCRIPT_DIRECTORY = "/lustre/fs23/group/that/xr/ana/blazar_population_cr/temp_scripts/"
PYTHON_SCRIPT = "/afs/ifh.de/group/that/work-xrod/NEUCOS/CRBlazars/trunk/run_propagation_single_population.py"

script_content = "\
export PRINCEROOT=/lustre/fs23/group/that/xr/PriNCe\n\
\
export PATH=/afs/ifh.de/user/x/xavier/.funs:/afs/ifh.de/group/that/work-xrod/sw:\
/afs/ifh.de/group/that/work-xrod/sw/boost_1_67_0:\
/afs/ifh.de/group/that/work-xrod/sw/boost_1_67_0/tools/build:\
$PRINCEROOT/branches/J_branch:$PRINCEROOT/branches/J_branch/prince:\
/afs/ifh.de/group/that/work-xrod/sw/anaconda/bin:\
/afs/ifh.de/group/that/work-xrod/sw/VSCode-linux-x64/bin:$PATH\n\
\
export LD_LIBRARY_PATH=/afs/ifh.de/group/that/work-xrod/NEUCOS/Coupling_AM3_NC/trunk/lib:\
/afs/ifh.de/group/that/work-xrod/sw/boost_1_67_0/stage/lib:$PRINCEROOT/branches/J_branch:\
$PRINCEROOT/branches/J_branch/prince:$PRINCEROOT/analyzer:\
/afs/ifh.de/group/that/work-xrod/NEUCOS/FermiBlazarSequence/trunk/agn_python_modules:\
/afs/ifh.de/group/that/work-xrod/NEUCOS/Coupling_AM3_NC/trunk/lib:$LD_LIBRARY_PATH\n\
\
export PYTHONPATH=$PRINCEROOT/branches/J_branch:$PRINCEROOT/branches/J_branch/prince:\
$PRINCEROOT/analyzer:/afs/ifh.de/group/that/work-xrod/NEUCOS/Coupling_AM3_NC/trunk/lib:\
/afs/ifh.de/group/that/work-xrod/NEUCOS/FermiBlazarSequence/trunk/agn_python_modules:\
$PYTHONPATH\n\
\
export MKL_NUM_THREADS=1\n\
\
python " + PYTHON_SCRIPT

submit_command = "\
qsub -l h_rt=05:00:00 -l h_rss=4G -N p{0:.1f}_{1:d}_{2:.0f} \
-j y -b y -o /lustre/fs23/group/that/xr/ana/blazar_population_cr/logs \
-m a -M xavier.rodrigues@desy.de \
sh "

# File with the emitted CR and neutrino fluxes computed with NeuCosmA
INPUT_FILE_PATH = "/lustre/fs23/group/that/xr/ana/nc_emitted_fluxes/191018fermi\
/frm190701_cr_nu_fluxes.h5"

isotope_arr = np.array([101])
radius_arr = np.array([6.]) # log10(R'_blob/c [s]) 
blazar_type_arr = ["FSRQ"]
escape_type_arr = ["log"]
accel_efficiency_arr = [0.] # log10(acceleration efficiency)
lum_arr = np.arange(43.5,51.5)

for eta in accel_efficiency_arr:
    for blazar_type in blazar_type_arr:
        for escape in escape_type_arr:
            for log_tvar in radius_arr:
                for composition in isotope_arr:
                    for lum in lum_arr:
                       
                        param_str = " {0:s} {1:.0f} {2:s} {3:s} {4:d} {5:.1f} {6:.1f} 1e-3".format(
                            INPUT_FILE_PATH,
                            eta,
                            blazar_type,
                            escape,
                            composition,
                            log_tvar,
                            lum
                        )

                        shell_script = (SCRIPT_DIRECTORY 
                                        + "submit_1e{0:.0f}_{1:s}_{2:s}_{3:.1f}_{4:d}_{5:.1f}.sh".format(
                                            eta,
                                            blazar_type,
                                            escape,
                                            log_tvar,
                                            composition,
                                            lum)
                        )
                        f = open(shell_script, "w")
                        f.write(script_content + param_str)
                        f.close()

                        formatted_command = submit_command.format(log_tvar,composition,round(lum))
                        subprocess.call(formatted_command + shell_script, shell=True)
    