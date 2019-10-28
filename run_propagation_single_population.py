# Script for running Prince on CR and neutrino emission from a single
# AGN population
#
# See further documentation in main function
#
# X. Rodrigues, 2019

import os
import sys
sys.path.insert(1, "/afs/ifh.de/group/that/work-jh/git/numpy/")
sys.path.insert(1, "/afs/ifh.de/group/that/work-jh/git/scipy/")
import numpy as np
import scipy as sc
from silx.io.dictdump import dicttoh5
from silx.io.dictdump import h5todict
import h5py
import astropy.units as u
import matplotlib.pyplot as plt
import cPickle as pickle
import inspect
import subprocess
import copy

from m_agn_population import Population

OUTPUT_PATH = "/lustre/fs23/group/that/xr/ana/blazar_population_cr/191028_"


def main(script, input_file_path, log_eta, blazar_type, escape, composition, log_tvar, log_lum, dz=1e-3):
    '''
    Creates and AGN population from NeuCosmA flux files and propagates it with Prince

    The parameters of main can be provided directly in the shell following the 
    script name. This function will find NeuCosmA files with the CR and neutrino
    spectra emitted from AGNs with the given parameters in a range of luminosities,
    import them into a Population object (defined in module m_agn_propagation), 
    propagate the fluxes with Prince, and save the propagated Population objects with
    pickle.
    For running this script in batch, use pyscript_propagation_submission.py. 
    The propagated spectra can be combined and plotted with the notebook
    t_agn_propagation.ipynb.


    Complete workflow (for nerds):
        Each AGN emission is simulated with NeuCosmA test code
        /afs/ifh.de/group/that/work-xrod/ana/NC/Neucosma/trunk/examples/TestFermiSequnce.c, 
        compiled into /afs/ifh.de/group/that/work-xrod/ana/NC/Neucosma/trunk/FRM190701.
        NeuCosmA can be run in batch on the cluster using script 
        /afs/ifh.de/group/that/work-xrod/ana/NC/Neucosma/trunk/run_fermi_sequence.sh.
        NeuCosmA results from TestFermiSequence output to the directory 
        /lustre/fs23/group/that/xr/ana/nc/. Those files contain all the information 
        about each AGN simulation. This info can then be processed with the notebook
        /afs/ifh.de/group/that/work-xrod/ana/nb/t_extract_fluxes_from_neucosma_output.ipynb,
        which writes only the CR and neutrino spectra into the file
        /lustre/fs23/group/that/xr/ana/nc_emitted_fluxes/191018fermi/frm190701_cr_nu_fluxes.h5.
        The spectra in that file are loaded in this main function and propagated using Prince.
        After propagation, this function saves the propagation results as HDF5 files under
        OUTPUT_PATH.
        
    parameters
    ----------
    Same as the attributes of Population class with a single luminosity bin.
    '''
    
    log_eta, log_tvar, composition, log_lum, dz = (np.float(log_eta),
                                                   np.float(log_tvar),
                                                   np.int(composition),
                                                   np.float(log_lum),
                                                   np.float(dz))

    pop = Population(input_file_path, log_eta, blazar_type, escape, composition, log_tvar, [log_lum])
    # Propagate spectra
    pop.propagate(dz=dz)

    # Save propagated population
    outfname = "{0:s}1e{1:.0f}_{2:s}_{3:s}_{4:04d}_{5:.1f}_{6:.1f}.h5\
".format(OUTPUT_PATH,
         log_eta,
         blazar_type, 
         escape,
         composition,
         log_tvar,
         log_lum)
    
    file = open(outfname, "w")
    pop.solvers = {} # No need to pickle the solvers,  
                     # only the respective results
    pop.sources = {} # No need to pickle the source object
    pickle.dump(pop, file, protocol=-1) 
    file.close()
                 
    return


if __name__ == "__main__":
    main(*sys.argv)
