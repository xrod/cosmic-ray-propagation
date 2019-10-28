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
# print inspect.getmodule(np)
# print inspect.getmodule(sc)
import subprocess     
import copy
import prince_config
from prince import core, util, photonfields, cross_sections
from prince import util as pru
from prince.solvers import UHECRPropagationSolverBDF
from prince.cr_sources import SpectrumSource # Source type for the source evolution

lustre = os.path.expanduser("/lustre/fs23/group/that/jheinze/prince_kernels/")
with open(lustre + 'prince_run_PSB_no_photons.ppo','rb') as thefile:
    prince_run = pickle.load(thefile)
     



def load_spectra(log_eta, blazar_type, esc, iso, logtvar, loglum, filepath):
    '''
    Loads spectra emitted by the source in the cosmological comoving frame.

    parameters
    ----------

    L : log10(gamma-ray luminosity [erg/s])
    iso : code of the injected isotope (eg 101 for protons)
    blazar_type : string with type of blazar ("BL" for BL Lac or "FSRQ")
    esc : CR escape type (see Population)

    return types: np.array, np.array, dict, np.array
    '''
    
    h5path = "/LOGACCEFF_{0:.1f}/TYPE_{1:s}/ESC_{2:s}/INJ_{3:02d}/LOGTVAR_{4:.1f}/LOGLGAMMA_{5:.1f}".format(
              log_eta,
              blazar_type,
              esc,
              iso%100,
              logtvar,
              loglum
              )
    dct = h5todict(filepath, path=h5path)
    
    # [GeV]
    egrid = dct['E']
    cr_injection = dct['inj']
    cr_spectrum  = dct['cr']
    nu_spectrum  = dct['nu']
    cr_spectrum = cr_spectrum.reshape((cr_spectrum.size 
                                       / (egrid.size + 1)),
                                      egrid.size + 1)
                         

    # Convert spectra
    cr_ids = np.array(cr_spectrum[:,0], dtype=np.int)

    cr_spectra_at_source = {}
    for pid, spectrum in zip(cr_ids,cr_spectrum):
        cr_spectra_at_source[pid] = spectrum[1:]

    return egrid, cr_injection, cr_spectra_at_source, nu_spectrum


def find_nearest_species(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def find_stable(pids, stableids):
    from prince_config import spec_data
    # ignore all known ids, that are not in pdis
    stableids = [pid for pid in pids if pid in stableids]
    unstableids = [pid for pid in pids if pid not in stableids]

    unstable_mapping = {}
    for pid in unstableids:
        unstable_mapping[pid] = find_nearest_species(stableids,pid)

    return stableids, unstableids, unstable_mapping

def thin_out_species(cosmicray, stableids):
    '''
    Thin out unstable species by adding them to the closest stable one
    '''
    from prince_config import spec_data
    stableids, unstableids, unstable_mapping = find_stable(cosmicray.keys(), stableids)
    
    for pid in unstableids:
        cosmicray[unstable_mapping[pid]] += cosmicray[pid]
        del cosmicray[pid]


        
class AjelloBlazar(SpectrumSource): 
    def __init__(self, *args, **kwargs):
        self.blazar_type = kwargs['blazar_type']
        self.luminosity = kwargs['luminosity']
        SpectrumSource.__init__(self, *args, **kwargs)
        
    def ajello_distribution(self,blazar_type, redshift, 
                            luminosity_min, luminosity_max):
        dh = 1.302905445498157*10**28 #Hubble distance in cm
        dhubble = 4.222428985915492 #Hubble distance in Gpc
        ergtogev = 1/(1.602176565*10**(-3))

        def hh(z):
            return np.sqrt(0.73+0.27*(1+z)**3)

        def ddc(z):
            return z/(0.9991563386954221 + 0.20853140019016303*z + 
            0.08984834223064098*z**2 - 0.029962175208216008*z**3 + 
            0.00575706733304888*z**4 - 0.0006417831690412458*z**5 + 
            0.00003833989451773751*z**6 - 9.471437700005367*10**-7*z**7)

        def jacobus(z):
            return 4*np.pi*dhubble**3*ddc(z)**2/hh(z)

        binned_luminosity = 1 if luminosity_min < luminosity_max else 0

        if luminosity_min > luminosity_max:
            print "[ajello_distribution]Error: must have ll_max > ll_min"
            return -1

        if blazar_type == 'BL':
            aaGpc = (3.39*10**4*10**-13)*10**9
            gam1 = 0.27
            llstar = np.log10(0.28*10**48.0)
            gam2 = 1.86
            zcstar = 1.34
            p1star = 2.24
            tau = 4.92
            p2 = -7.37
            alpha = 0.0453
            mustar = 2.10
            beta = 0.0646
            sigma = 0.26
            g1 = 1.45
            g2 = 2.80 
            z1 = 0.03
            z2 = 6.
            ll1 = np.log10(7*10**43.0)
            llmid = 46.
            ll2 = 52.

            def mumu(ll):
                return mustar+beta*(ll-llmid)
            def phig(g,mu):
                return np.exp(-(g-mu)**2/(2*sigma**2))
            def phill(ll):
                return 1/((10**(gam1*(ll-llstar)))+(10**(gam2*(ll-llstar))))
            def p1(ll):
                return p1star+tau*(ll-llmid)
            def zc(ll):
                return zcstar*10**(alpha*(ll-48))
            def eey(y,ll):
                return 1/(y**(-p1(ll))+y**(-p2))
            def ee(z,ll): 
                return eey((1+z)/(1+zc(ll)),ll)
            
            def distribution(z,ll,g):
                #distribution of BL Lacs as a function of redshift, 
                # luminosity, and spectral index
                return aaGpc*ee(z,ll)*phill(ll)*phig(g,mumu(ll))   

            ll_min = luminosity_min if luminosity_min else ll1
            ll_max = luminosity_max if luminosity_max else ll2       
            g_min = g1
            g_max = g2

        elif blazar_type == 'FSRQ':
            aaGpcf = (3.06*10**4*10**-13)*10**9
            gam1f = 0.21
            llstarf = np.log10(0.84*10**48.0)
            gam2f = 1.58
            zcstarf = 1.47
            p1starf = 7.35
            tauf = 0
            p2f = -6.51
            alphaf = 0.21
            mustarf = 2.44
            betaf = 0.0
            sigmaf = 0.18
            g1f = 1.8
            g2f = 3 
            z1f = 0
            z2f = 6.0
            ll1f = 44.0
            llmidf = 46.0
            ll2f = 52.0

            ll_min = np.log10(luminosity_min) if luminosity_min else ll1f
            ll_max = np.log10(luminosity_max) if luminosity_max else ll2f

            def mumuf(ll):
                return mustarf+betaf*(ll-llmidf)
            def phigf(g,mu):
                return np.exp(-(g-mu)**2/(2*sigmaf**2))
            def phillf(ll):
                return 1/((10**(gam1f*(ll-llstarf)))+(10**(gam2f*(ll-llstarf))))
            def p1f(ll):
                return p1starf+tauf*(ll-llmidf)
            def zcf(ll):
                return zcstarf*10**(alphaf*(ll-48))
            def eeyf(y,ll):
                return 1/(y**(-p1f(ll))+y**(-p2f))
            def eef(z,ll): 
                return eeyf((1+z)/(1+zcf(ll)),ll)

            def distribution(z,ll,g):
                # Distribution of FSRQs as a function of redshift, 
                # luminosity, and spectral index
                return aaGpcf*eef(z,ll)*phillf(ll)*phigf(g,mumuf(ll)) 

            ll_min = luminosity_min if luminosity_min else ll1f
            ll_max = luminosity_max if luminosity_max else ll2f       
            g_min = g1f
            g_max = g2f
        else:
            print "[ajello_distribution]Error: unknown blazar type"
            return -1

        if binned_luminosity:
            func = lambda ll, g: distribution(redshift, ll, g)
            result = sc.integrate.nquad(func, [[ll_min, ll_max], [g_min, g_max]])
        else:
            func = lambda g: distribution(redshift, ll_min, g)
            result = sc.integrate.quad(func, g_min, g_max)

        return result[0]
   

    def evolution(self, redshift):
        lum_min = self.luminosity - 0.5
        lum_max = self.luminosity + 0.5
        blazar_type = self.blazar_type
        
        dn_dz = self.ajello_distribution(blazar_type, redshift, lum_min, lum_max)
        
        from prince import cosmology as pc
        dn_dt = dn_dz * u.Unit('Gpc^-3').to('cm^-3')# [cm^-3 s^-1]     
    
        return dn_dt



class Population():
    '''
    AGN/blazar population that can be initlaized from files with emitted CR and neutrino fluxes
    and propagated later with Prince.

    attributes
    ----------
    
    inputfile : the path to the file containing the ejected spectra
                to be loaded. 
    log_eta : log10(acceleration efficiency)
    blazar_type : 'FSRQ' or 'BL'.
    escape : string with CR escape mechanism ('dif', 'adv' or 'log' for log-
             parabola)
    composition : array with IDs of the injected species
    log_tvar : log10(tvar [s]) where tvar is the light-crossing timescale of 
               the blob in its own rest frame 
    log_lum_arr : array with values of log10(L [erg/s]) where L is the central
                  value of the gamma-ray luminosity bin of an AGN population   
    pop_dict : if inputfile is None, pop_dict should be a dictionary of pairs
               {L : p} where p is a Population object and L is the log_lum value
               of the bin to be imported
    '''
    def __init__(self, inputfile, log_eta, blazar_type, escape, composition, log_tvar, log_lum_arr, pop_dict=None):

        self.acceleration_effciency = 10 ** log_eta
        self.blazar_type = blazar_type
        self.escape = escape
        self.composition = composition
        self.log_tvar = log_tvar
        try:
            self.luminosities = np.sort(log_lum_arr)
        except:
            self.luminosities = np.array([log_lum_arr])
        
        self.egrid_source = {}
        self.injection = {}
        self.spectra_at_source = {}
        self.sources = {}
        self.solvers = {}
        self.results = {}

        for L in self.luminosities:
            
            # If no inputfile is givem, try loading already propagated Population objects
            if inputfile is None:
                                
                for L, obj in pop_dict.items():
                    if L not in self.luminosities:
                        print "[Population]Warning: new population has new luminosity bin, updating."
                        self.luminosities = np.append(self.luminosities, L)
                    try:
                        newpop = pickle.load(obj)
                    except:
                        newpop = obj
                    self.spectra_at_source[L] = newpop.spectra_at_source[L] 
                    self.sources[L] = newpop.sources[L]
                    self.solvers[L] = newpop.solvers[L]
                    self.results[L] = newpop.results[L]

                    if (self.escape != newpop.escape or
                        self.blazar_type != newpop.blazar_type or
                        self.composition != newpop.composition):
                        print "[Population]Warning: loaded population has different properties"

            # If inputfile is given, assume file contains the emission spectra
            else:

                # All arrays in GeV
                egrid, injection, cr, nu = load_spectra(log_eta, 
                                                        blazar_type, 
                                                        escape, 
                                                        composition, 
                                                        log_tvar, 
                                                        L,
                                                        inputfile)

                thin_out_species(cr,prince_run.spec_man.known_species)
                
                # params: spectra in GeV^-1
                params = {key:(egrid, cr[key] / egrid**2) for key in cr if key >= 100} 
                params[16] = egrid, nu / egrid**2 # Store source neutrinos
                                                  # under the "tau neutrino" key

                source = AjelloBlazar(prince_run, params = params, norm = 1., 
                                      blazar_type=self.blazar_type, luminosity=L)

                self.egrid_source[L] = egrid
                self.injection[L] = injection
                self.spectra_at_source[L] = dict(cr)
                self.spectra_at_source[L][16] = nu # Store source neutrinos 
                                                   # as tau neutrinos
                self.sources[L] = source
        
    def get_result(self, L):
        return self.results[L]
    
    def get_spectrum_at_source(self, L, species_id, epow=2):
        egrid , spect = self.egrid_source, self.spectra_at_source[L][species_id]
        return egrid, spect *  egrid ** (epow - 2)
        
    def get_spectrum_at_earth(self, L, species_id, epow=2):
        result = self.get_result(L)
        x, y = result.get_solution_scale(species_id, epow)
        return x, y
    
    def get_injection_spectrum(self, L, species_id):
        '''
        Injection spectrum from the acceleration region into the source
        '''
        egrid_source = self.egrid_source
        spect = self.injection[L][species_id]
        
        return egrid, spect *  egrid ** (epow - 2)
    
    def get_species_at_source(self):
        lum0 = self.luminosities[0]
        try:
            return np.array(self.spectra_at_source[lum0].keys())
        except:
            try:
                return np.array(self.spectra_at_source[lum0].keys())
            except:
                return np.array([])
    
    def propagate(self, dz=1e-3):
        '''
        Runs Prince on the population.

        After propagation, the solver of luminosity bin L is saved under 
        self.solvers[L] and the respective result under self.results[L].
        If only the result is important, set self.solvers={}  after propagation.
        '''
        for L in self.luminosities:
            solver = UHECRPropagationSolverBDF(initial_z=5., final_z = 0.,
                                               prince_run=prince_run, atol= 1e68, 
                                               rtol=1e-6, enable_pairprod_losses = True, 
                                               enable_adiabatic_losses = True, 
                                               enable_injection_jacobian=True,
                                               enable_partial_diff_jacobian=True)
            solver.add_source_class(self.sources[L])
            solver.solve(dz=dz,verbose=False,full_reset=False,progressbar='notebook')
            self.solvers[L] = solver
            self.results[L] = solver.res
        return
