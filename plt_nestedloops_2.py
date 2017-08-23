#### This script is going to plot the variation in the sky level vs the DFT and photon-shooting time as I keep all other variables constant. I will plot 4 subplots each one representing the gaussian PSF and gaussian galaxy profile with a different flux variation for each subplot. Within each plot I will have 5 sets of data each one varying the pixel scale. This will show the change in the sky levels being varied against constant pixel scale values.

import sys
import os
import math
import numpy
import logging
import time
import galsim
import matplotlib.pyplot as plt
import numpy as np
import pdb

def func(file_name, random_seed, pixel_scale, nx, ny, sky_level, gal_flux_min, gal_flux_max, gal_hlr_min, gal_hlr_max, gal_e_min, gal_e_max, psf_fwhm, gsparams, psf1, psf2, psf3_inner, psf3_outer, psf3, atmos, aberrations, psf4, optics,  psf5, psfs, psf_names, psf_times, psf_fft_times, psf_phot_times, gal1, gal2, gal3, gal4, bulge, disk, gal5, gals, gal_names, gal_times, gal_fft_times, gal_phot_times, setup_times, fft_times, phot_times, noise_times, k, all_fluxes_i, psf, psf_name, gal, gal_name):

    #Functioning
    #---> start here

    logger = logging.getLogger("demo7")
    
    ### this willl return the time in seconds
    t1 = time.time()
    
    ### generate randomly a number
    # Initialize the random number generator we will be using.
    rng = galsim.UniformDeviate(random_seed+k+1)
    
    ### use the random numbers for the flux
    # Generate random variates:
    flux = all_fluxes_i
    y_i = flux
    
    # Use a new variable name, since we'll want to keep the original unmodified.
    this_gal = gal.withFlux(flux)
    
    ### use the random numbers for the hlr
    hlr = rng() * (gal_hlr_max-gal_hlr_min) + gal_hlr_min
    
    # Use a new variable name, since we'll want to keep the original unmodified.
    this_gal = this_gal.dilate(hlr)
    
    ### use the random numbers for the ellipticity
    beta_ellip = rng() * 2*math.pi * galsim.radians
    ellip = rng() * (gal_e_max-gal_e_min) + gal_e_min
    gal_shape = galsim.Shear(e=ellip, beta=beta_ellip)
    
    # Use a new variable name, since we'll want to keep the original unmodified.
    this_gal = this_gal.shear(gal_shape)
    
    ### build the final object by combinging the galaxy and psf
    final = galsim.Convolve([this_gal, psf])
    
    ### create the images and save them in an empty array
    image = galsim.ImageF(2*nx+2, ny, scale=pixel_scale)
    
    ### make a views subset of the larger image
    fft_image = image[galsim.BoundsI(1, nx, 1, ny)]
    phot_image = image[galsim.BoundsI(nx+3, 2*nx+2, 1, ny)]
    
    logger.debug('      Read in training sample galaxy and PSF from file')
    
    ### record time
    
    t2 = time.time()
    
    ### Draw the profile explicity through fft for the convolution
    final.drawImage(fft_image, method='fft')
    
    logger.debug('      Drew fft image.  Total drawn flux = %f.  .flux = %f',
                 fft_image.array.sum(),final.getFlux())
    ### record time
    t3 = time.time()

    ### Add Poisson noise to the fft image convolution
    sky_level_pixel = sky_level * pixel_scale**2
    fft_image.addNoise(galsim.PoissonNoise(rng, sky_level=sky_level_pixel))
    ### record time
    t4 = time.time()

    # The next two lines are just to get the output from this demo script
    # to match the output from the parsing of demo7.yaml.
    rng = galsim.UniformDeviate(random_seed+k+1)
    rng(); rng(); rng(); rng();

    ### repeat for photon-shooting
    final.drawImage(phot_image, method='phot', max_extra_noise=sky_level_pixel/100,
                 rng=rng)

    ### record time
    t5 = time.time()

    ### For photon shooting, galaxy already has Poisson noise, so we want to make sure not to add that noise again.
    ### Thus, we just add sky noise, which is Poisson with the mean = sky_level_pixel
    pd = galsim.PoissonDeviate(rng, mean=sky_level_pixel)

    # DeviateNoise just adds the action of the given deviate to every pixel.
    phot_image.addNoise(galsim.DeviateNoise(pd))

    # For PoissonDeviate, the mean is not zero, so for a background-subtracted
    # image, we need to subtract the mean back off when we are done.
    phot_image -= sky_level_pixel

    logger.debug('      Added Poisson noise.  Image fluxes are now %f and %f',
              fft_image.array.sum(), phot_image.array.sum())

    ### record time
    t6 = time.time()
    #<----end here (return stuff)
    
                 
    return image, t1, t2, t3, t4, t5, t6, k, flux, hlr, gal_shape, y_i, psfs, gals, file_name



def main(argv):
    
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo7")
    
    ### Make output directory if not already present.
    if not os.path.isdir('output'):
        os.mkdir('output')
    
    file_name = os.path.join('output','cube_phot.fits.gz')


    yimg=[]
    DFT=[]
    Photon=[]

    ranges_n = np.linspace(10,50,5)
    ranges_hlrmin= np.linspace(0.1,.95,5)
    ranges_hlrmax= np.linspace(0.0,1,5)
    ranges_emin = np.linspace(0.0,1,5)
    ranges_emax = np.linspace(0.0,1,5)


    ranges_maxfold = np.linspace(.01,.1,5)
    ranges_excludedk = np.linspace(2.e-6,2.e-1,5)
    ranges_DFTaim = np.linspace(1.e-7,1.e-1,5)
    ranges_photonaim = np.linspace(1.e-6,1.e-1,5)
    ranges_DFTmin = [20,40,60,80,100]

    for n in ranges_n:
        nx = n
        ny = n
        for gal_hlr_min in ranges_hlrmin:
            for gal_hlr_max in ranges_hlrmax:
                for gal_e_min in ranges_emin:
                    for gal_e_max in ranges_emax:
                        ### Define some parameters we'll use below.
                        random_seed = 553728
                        gal_flux_min = 1.e1     # Range for galaxy flux
                        gal_flux_max = 1.e5     # Range for ellipticity
                        sky_level = 1.e4
                        psf_fwhm = 0.65
                        pixel_scale = .28
                        spaceaim = 1.e-4
                        for maxfold in ranges_maxfold:
                            for excludedk in ranges_excludedk:
                                for DFTaim in ranges_DFTaim:
                                    for photonaim in ranges_photonaim:
                                        for DFTmin in ranges_DFTmin:

                                            # We encapsulate these parameters with an object called GSParams.  The default values
                                            # are intended to be accurate enough for normal precision shear tests, without sacrificing
                                            # too much speed.Any PSF or galaxy object can be given a gsparams argument on construction that can
                                            # have different values to make the calculation more or less accurate (typically trading
                                            # off for speed or memory).
                                        
                                            gsparams = galsim.GSParams(
                                               folding_threshold=maxfold, # maximum fractional flux that may be folded around edge of FFT
                                               maxk_threshold=excludedk,    # k-values less than this may be excluded off edge of FFT
                                               xvalue_accuracy=spaceaim,   # approximations in real space aim to be this accurate
                                               kvalue_accuracy=DFTaim,   # approximations in fourier space aim to be this accurate
                                               shoot_accuracy=photonaim,    # approximations in photon shooting aim to be this accurate
                                               minimum_fft_size=DFTmin)     # minimum size of ffts

                                            logger.info('Starting psf')
                                               
                                            # Make the PSF profiles:
                                            ### psf 1
                                            psf1 = galsim.Gaussian(fwhm = psf_fwhm, gsparams=gsparams)
                                           
                                            ### psf 2
                                            psf2 = galsim.Moffat(fwhm = psf_fwhm, beta = 2.4, gsparams=gsparams)
                                           
                                            ### psf 3
                                            psf3_inner = galsim.Gaussian(fwhm = psf_fwhm, flux = 0.8, gsparams=gsparams)
                                            psf3_outer = galsim.Gaussian(fwhm = 2*psf_fwhm, flux = 0.2, gsparams=gsparams)
                                            psf3 = psf3_inner + psf3_outer
                                            atmos = galsim.Gaussian(fwhm = psf_fwhm, gsparams=gsparams)
                                           
                                            ### defining the telescope
                                       
                                            # The OpticalPSF and set of Zernike values chosen below correspond to a reasonably well aligned,
                                            # smallish ~0.3m / 12 inch diameter telescope with a central obscuration of ~0.12m or 5 inches
                                            # diameter, being used in optical wavebands.
                                           
                                            aberrations = [ 0.0 ] * 12          # Set the initial size.
                                            aberrations[4] = 0.06               # Noll index 4 = Defocus
                                            aberrations[5:7] = [ 0.12, -0.08 ]  # Noll index 5,6 = Astigmatism
                                            aberrations[7:9] = [ 0.07, 0.04 ]   # Noll index 7,8 = Coma
                                            aberrations[11] = -0.13             # Noll index 11 = Spherical
                                               
                                            optics = galsim.OpticalPSF(
                                                              lam_over_diam = 0.6 * psf_fwhm, obscuration = 0.4, aberrations = aberrations,
                                                              gsparams=gsparams)
                                               
                                            ### psf 4
                                            psf4 = galsim.Convolve([atmos, optics]) # Convolve inherits the gsparams from the first
                                            # item in the list.  (Or you can supply a gsparams
                                            # argument explicitly if you want to override this.)
                                            atmos = galsim.Kolmogorov(fwhm = psf_fwhm, gsparams=gsparams)
                                            optics = galsim.Airy(lam_over_diam = 0.3 * psf_fwhm, gsparams=gsparams)

                                            ### psf 5
                                            psf5 = galsim.Convolve([atmos,optics])

                                            ### define where to keep the psfs info
                                            psfs = [psf1, psf2, psf3, psf4, psf5]
                                            psf_names = ["Gaussian", "Moffat", "Double Gaussian", "OpticalPSF", "Kolmogorov * Airy"]
                                            psf_times = [0,0,0,0,0]
                                            psf_fft_times = [0,0,0,0,0]
                                            psf_phot_times = [0,0,0,0,0]

                                            # Make the galaxy profiles:

                                            ### gal 1
                                            gal1 = galsim.Gaussian(half_light_radius = 1, gsparams=gsparams)

                                            ### gal 2
                                            gal2 = galsim.Exponential(half_light_radius = 1, gsparams=gsparams)

                                            ### gal 3
                                            gal3 = galsim.DeVaucouleurs(half_light_radius = 1, gsparams=gsparams)

                                            ### gal 4
                                            gal4 = galsim.Sersic(half_light_radius = 1, n = 2.5, gsparams=gsparams)
                                            bulge = galsim.Sersic(half_light_radius = 0.7, n = 3.2, trunc = 8.5, gsparams=gsparams)
                                            disk = galsim.Sersic(half_light_radius = 1.2, n = 1.5, gsparams=gsparams)

                                            ### gal 5
                                            gal5 = 0.4*bulge + 0.6*disk  # Net half-light radius is only approximate for this one.

                                            ### define where to keep the galaxys info
                                            gals = [gal1, gal2, gal3, gal4, gal5]
                                            gal_names = ["Gaussian", "Exponential", "Devaucouleurs", "n=2.5 Sersic", "Bulge + Disk"]
                                            gal_times = [0,0,0,0,0]
                                            gal_fft_times = [0,0,0,0,0]
                                            gal_phot_times = [0,0,0,0,0]

                                            ### initial time conditions
                                            # Other times to keep track of:
                                            setup_times = 0
                                            fft_times = 0
                                            phot_times = 0
                                            noise_times = 0

                                            # Loop over combinations of psf, gal, and make 4 random choices for flux, size, shape.
                                            all_images = []
                                            k = 0

                                            n=[]
                                            x_DFT=[]
                                            x_Photon=[]
                                            y=[]
                                            all_fluxes = np.linspace(gal_flux_min,gal_flux_max,100)

                                            ### will loop through the numbers (amount) of psfs
                                            for ipsf in range(len(psfs)):
                                        
                                               ### calls on the 5 psfs and their names
                                               psf = psfs[ipsf]
                                               psf_name = psf_names[ipsf]
                                                       
                                               ### outputs the psf number and the psf information needed to create an object
                                               logger.info('psf %d: %s',ipsf+1,psf)
                                                           
                                               logger.debug('repr = %r',psf)
                                               
                                               ### will loop through the numbers (amount) of galaxies
                                               for igal in range(len(gals)):
                                                   
                                                       ### calls on the 5 galaxies and their names
                                                       gal = gals[igal]
                                                       gal_name = gal_names[igal]
                                                           
                                                       ### outputs the psf number and the psf information needed to create an object
                                                       logger.info('   galaxy %d: %s',igal+1,gal)
                                                       logger.debug('   repr = %r',gal)
                                                       
                                                       ### will loop though 0,1,2,3 flux, size, and shape to create 4 images for each
                                                       ### combination of galaxy and psf
                                                       for i in range(4):
                                                           logger.debug('      Start work on image %d',i)
                                                               
                                                           all_fluxes_i = all_fluxes[i]
                                                           image, t1, t2, t3, t4, t5, t6, k, flux, hlr, gal_shape, y_i, psfs, gals, file_name = func(file_name, random_seed, pixel_scale, nx, ny, sky_level, gal_flux_min, gal_flux_max, gal_hlr_min, gal_hlr_max, gal_e_min, gal_e_max, psf_fwhm, gsparams, psf1, psf2, psf3_inner, psf3_outer, psf3, atmos, aberrations, psf4, optics,  psf5, psfs, psf_names, psf_times, psf_fft_times, psf_phot_times, gal1, gal2, gal3, gal4, bulge, disk, gal5, gals, gal_names, gal_times, gal_fft_times, gal_phot_times, setup_times, fft_times, phot_times, noise_times, k, all_fluxes_i, psf, psf_name, gal, gal_name)
            
                                                           ### Store that into the list of all images
                                                           all_images += [image]
                    
                                                           ### add an itteration though the loop for the psf and galaxy combination images
                                                           k = k+1
                                                               
                                                           ### express the flux,hlr, and ellip of each image combination,4 for every loop
                                                           logger.info('      %d: flux = %.2e, hlr = %.2f, ellip = (%.2f,%.2f)',
                                                                       k, flux, hlr, gal_shape.getE1(), gal_shape.getE2())
                                                               
                                                                                                                                                                                                                                           
                                                           logger.debug('      Times: %f, %f, %f, %f, %f',t2-t1, t3-t2, t4-t3, t5-t4, t6-t5)
                                                        
                                                           psf_times[ipsf] += t6-t1
                                                           psf_fft_times[ipsf] += t3-t2
                                                           psf_phot_times[ipsf] += t5-t4
                                                           gal_times[igal] += t6-t1
                                                           gal_fft_times[igal] += t3-t2
                                                           gal_phot_times[igal] += t5-t4
                                                           setup_times += t2-t1
                                                           fft_times += t3-t2
                                                           phot_times += t5-t4
                                                           noise_times += t4-t3 + t6-t5
                                                           x_DFT = np.append(x_DFT,gal_fft_times[igal])
                                                           x_Photon = np.append(x_Photon,gal_phot_times[igal])
                                                               
                                                           #### flux and time of each galaxy profile with each PSF
                                            yimg=np.append(yimg,y)
                                            DFT=np.append(DFT,x_DFT)
                                            Photon=np.append(Photon,x_Photon)

###define when y=0
    y = [0,0,0,0,0]

###DFTMIN

    #### Time for the first PSF and first Gaussian

    DFT_flux10_DFTmin = (DFT[0],DFT[100],DFT[200],DFT[300],DFT[400])
    DFT_flux1020_DFTmin = (DFT[1],DFT[101],DFT[201],DFT[301],DFT[401])
    DFT_flux2030_DFTmin = (DFT[2],DFT[102],DFT[203],DFT[302],DFT[402])
    DFT_flux3040_DFTmin = (DFT[3],DFT[103],DFT[203],DFT[303],DFT[403])

    Photon_flux10_DFTmin = (Photon[0],Photon[100],Photon[200],Photon[300],Photon[400])
    Photon_flux1020_DFTmin = (Photon[1],Photon[101],Photon[201],Photon[301],Photon[401])
    Photon_flux2030_DFTmin = (Photon[2],Photon[102],Photon[203],Photon[302],Photon[402])
    Photon_flux3040_DFTmin = (Photon[3],Photon[103],Photon[203],Photon[303],Photon[403])
                       
    ### Flux for 1st PSF and 1st Galaxy
    y_flux10_DFTmin = (yimg[0],yimg[100],yimg[200],yimg[300],yimg[400])
    y_flux1020_DFTmin = (yimg)[1],yimg[101],yimg[201],yimg[301],yimg[401])
    y_flux2030_DFTmin = (yimg[2],yimg[102],yimg[203],yimg[302],yimg[402])
    y_flux3040_DFTmin = (yimg[3],yimg[103],yimg[203],yimg[303],yimg[403])

    #### the DFT minus Photon-Shooting time for the first psf and first galaxy

    xnew_1_DFTmin=[]
    xnew_2_DFTmin=[]
    xnew_3_DFTmin=[]
    xnew_4_DFTmin=[]
    xnew_5_DFTmin=[]
    for a,b,c,d,e,f,g,h in zip (DFT_flux10_DFTmin, DFT_flux1020_DFTmin, DFT_flux2030_DFTmin, DFT_flux3040_DFTmin, Photon_flux10_DFTmin, Photon_flux1020_DFTmin, Photon_flux2030_DFTmin, Photon_flux3040_DFTmin):
        x1=a-b
        x2=c-d
        x3=e-f
        x4=g-h
        xnew_1_DFTmin= np.append(xnew_1_DFTmin,x1)
        xnew_2_DFTmin= np.append(xnew_2_DFTmin,x2)
        xnew_3_DFTmin= np.append(xnew_3_DFTmin,x3)
        xnew_4_DFTmin= np.append(xnew_4_DFTmin,x4)


### Photon Shooting Accuracy

    #### Time for the first PSF and first Gaussian

    DFT_flux10_aim = (DFT[0],DFT[500],DFT[1000],DFT[1500],DFT[2000])
    DFT_flux1020_aim = (DFT[1],DFT[501],DFT[1001],DFT[1501],DFT[2001])
    DFT_flux2030_aim = (DFT[2],DFT[502],DFT[1002],DFT[1502],DFT[2002])
    DFT_flux3040_aim = (DFT[3],DFT[503],DFT[1003],DFT[1503],DFT[2003])
    
    Photon_flux10_aim = (Photon[0],Photon[500],Photon[1000],Photon[1500],Photon[2000])
    Photon_flux1020_aim = (Photon[1],Photon[501],Photon[1001],Photon[1501],Photon[2001])
    Photon_flux2030_aim = (Photon[2],Photon[502],Photon[1002],Photon[1502],Photon[2002])
    Photon_flux3040_aim = (Photon[3],Photon[503],Photon[1003],Photon[1503],Photon[2003])

    ### Flux for 1st PSF and 1st Galaxy
    y_flux10_aim = (yimg[0],yimg[500],yimg[1000],yimg[1500],yimg[2000])
    y_flux1020_aim = (yimg[1],yimg[501],yimg[1001],yimg[1501],yimg[2001])
    y_flux2030_aim = (yimg[2],yimg[502],yimg[1002],yimg[1502],yimg[2002])
    y_flux3040_aim = (yimg[3],yimg[503],yimg[1003],yimg[1503],yimg[2003])
    
    #### the DFT minus Photon-Shooting time for the first psf and first galaxy
    
    xnew_1_aim=[]
    xnew_2_aim=[]
    xnew_3_aim=[]
    xnew_4_aim=[]
    xnew_5_aim=[]
    for a,b,c,d,e,f,g,h in zip (DFT_flux10_aim, DFT_flux1020_aim, DFT_flux2030_aim, DFT_flux3040_aim, Photon_flux10_aim, Photon_flux1020_aim, Photon_flux2030_aim, Photon_flux3040_aim):
        x1=a-b
        x2=c-d
        x3=e-f
        x4=g-h
        xnew_1_aim= np.append(xnew_1_aim,x1)
        xnew_2_aim= np.append(xnew_2_aim,x2)
        xnew_3_aim= np.append(xnew_3_aim,x3)
        xnew_4_aim= np.append(xnew_4_aim,x4)

### K-Value

#### Time for the first PSF and first Gaussian

    DFT_flux10_k = (DFT[0],DFT[2500],DFT[5000],DFT[7500],DFT[10000])
    DFT_flux1020_k = (DFT[1],DFT[2501],DFT[5001],DFT[7501],DFT[10001])
    DFT_flux2030_k = (DFT[2],DFT[2502],DFT[5002],DFT[7502],DFT[10002])
    DFT_flux3040_k = (DFT[3],DFT[2503],DFT[5003],DFT[7503],DFT[10003])
    
    Photon_flux10_k = (Photon[0],Photon[2500],Photon[5000],Photon[7500],Photon[10000])
    Photon_flux1020_k = (Photon[1],Photon[2501],Photon[5001],Photon[7501],Photon[10001])
    Photon_flux2030_k = (Photon[2],Photon[2502],Photon[5002],Photon[7502],Photon[10002])
    Photon_flux3040_k = (Photon[3],Photon[2503],Photon[5003],Photon[7503],Photon[10003])
    
    ### Flux for 1st PSF and 1st Galaxy
    y_flux10_k = (yimg[0],yimg[2500],yimg[5000],yimg[7500],yimg[10000])
    y_flux1020_k = (yimg[1],yimg[2501],yimg[5001],yimg[7501],yimg[10001])
    y_flux2030_k = (yimg[2],yimg[2502],yimg[5002],yimg[7502],yimg[10002])
    y_flux3040_k = (yimg[3],yimg[2503],yimg[5003],yimg[7503],yimg[10003])
    
    #### the DFT minus Photon-Shooting time for the first psf and first galaxy
    
    xnew_1_k=[]
    xnew_2_k=[]
    xnew_3_k=[]
    xnew_4_k=[]
    xnew_5_k=[]
    for a,b,c,d,e,f,g,h in zip (DFT_flux10_k, DFT_flux1020_k, DFT_flux2030_k, DFT_flux3040_k, Photon_flux10_k, Photon_flux1020_k, Photon_flux2030_k, Photon_flux3040_k):
        x1=a-b
        x2=c-d
        x3=e-f
        x4=g-h
        xnew_1_k= np.append(xnew_1_k,x1)
        xnew_2_k= np.append(xnew_2_k,x2)
        xnew_3_k= np.append(xnew_3_k,x3)
        xnew_4_k= np.append(xnew_4_k,x4)
    
    ### max k
    
    #### Time for the first PSF and first Gaussian
    
    DFT_flux10_maxk = (DFT[0],DFT[12500],DFT[25000],DFT[37500],DFT[50000])
    DFT_flux1020_maxk = (DFT[1],DFT[12501],DFT[25001],DFT[37501],DFT[50001])
    DFT_flux2030_maxk = (DFT[2],DFT[12502],DFT[25002],DFT[37502],DFT[50002])
    DFT_flux3040_maxk = (DFT[3],DFT[12503],DFT[25003],DFT[37503],DFT[50003])

    Photon_flux10_maxk = (Photon[0],Photon[12500],Photon[25000],Photon[37500],Photon[50000])
    Photon_flux1020_maxk = (Photon[1],Photon[12501],Photon[25001],Photon[37501],Photon[50001])
    Photon_flux2030_maxk = (Photon[2],Photon[12502],Photon[25002],Photon[37502],Photon[50002])
    Photon_flux3040_maxk = (Photon[3],Photon[12503],Photon[25003],Photon[37503],Photon[50003])
    
    ### Flux for 1st PSF and 1st Galaxy
    y_flux10_maxk = (yimg[0],yimg[12500],yimg[25000],yimg[37500],yimg[50000])
    y_flux1020_maxk = (yimg[1],yimg[12501],yimg[25001],yimg[37501],yimg[50001])
    y_flux2030_maxk = (yimg[2],yimg[12502],yimg[25002],yimg[37502],yimg[50002])
    y_flux3040_maxk = (yimg[3],yimg[12503],yimg[25003],yimg[37503],yimg[50003])
    
    #### the DFT minus Photon-Shooting time for the first psf and first galaxy
    
    xnew_1_maxk=[]
    xnew_2_maxk=[]
    xnew_3_maxk=[]
    xnew_4_maxk=[]
    xnew_5_maxk=[]
    for a,b,c,d,e,f,g,h in zip (DFT_flux10_maxk, DFT_flux1020_maxk, DFT_flux2030_maxk, DFT_flux3040_maxk, Photon_flux10_maxk, Photon_flux1020_maxk, Photon_flux2030_maxk, Photon_flux3040_maxk):
        x1=a-b
        x2=c-d
        x3=e-f
        x4=g-h
        xnew_1_maxk= np.append(xnew_1_maxk,x1)
        xnew_2_maxk= np.append(xnew_2_maxk,x2)
        xnew_3_maxk= np.append(xnew_3_maxk,x3)
        xnew_4_maxk= np.append(xnew_4_maxk,x4)


### maxfold

#### Time for the first PSF and first Gaussian

    DFT_flux10_maxfold = (DFT[0],DFT[62500],DFT[125000],DFT[187500],DFT[250000])
    DFT_flux1020_maxfold = (DFT[1],DFT[62501],DFT[125001],DFT[187501],DFT[250001])
    DFT_flux2030_maxfold = (DFT[2],DFT[62502],DFT[125002],DFT[187502],DFT[250002])
    DFT_flux3040_maxfold = (DFT[3],DFT[62503],DFT[125003],DFT[187503],DFT[250003])
    
    Photon_flux10_maxfold = (Photon[0],Photon[62500],Photon[125000],Photon[187500],Photon[250000])
    Photon_flux1020_maxfold = (Photon[1],Photon[62501],Photon[125001],Photon[187501],Photon[250001])
    Photon_flux2030_maxfold = (Photon[2],Photon[62502],Photon[125002],Photon[187502],Photon[250002])
    Photon_flux3040_maxfold = (Photon[3],Photon[62503],Photon[125003],Photon[187503],Photon[250003])
    
    ### Flux for 1st PSF and 1st Galaxy
    y_flux10_maxfold = (yimg[0],yimg[62500],yimg[125000],yimg[187500],yimg[250000])
    y_flux1020_maxfold = (yimg[1],yimg[62501],yimg[125001],yimg[187501],yimg[250001])
    y_flux2030_maxfold = (yimg[2],yimg[62502],yimg[125002],yimg[187502],yimg[250002])
    y_flux3040_maxfold = (yimg[3],yimg[62503],yimg[125003],yimg[187503],yimg[250003])
    
    #### the DFT minus Photon-Shooting time for the first psf and first galaxy
    
    xnew_1_maxfold=[]
    xnew_2_maxfold=[]
    xnew_3_maxfold=[]
    xnew_4_maxfold=[]
    xnew_5_maxfold=[]
    for a,b,c,d,e,f,g,h in zip (DFT_flux10_maxfold, DFT_flux1020_maxfold, DFT_flux2030_maxfold, DFT_flux3040_maxfold, Photon_flux10_maxfold, Photon_flux1020_maxfold, Photon_flux2030_maxfold, Photon_flux3040_maxfold):
        x1=a-b
        x2=c-d
        x3=e-f
        x4=g-h
        xnew_1_maxfold= np.append(xnew_1_maxfold,x1)
        xnew_2_maxfold= np.append(xnew_2_maxfold,x2)
        xnew_3_maxfold= np.append(xnew_3_maxfold,x3)
        xnew_4_maxfold= np.append(xnew_4_maxfold,x4)

### emax

    #### Time for the first PSF and first Gaussian

    DFT_flux10_emax = (DFT[0],DFT[312500],DFT[625000],DFT[937500],DFT[1250000])
    DFT_flux1020_emax = (DFT[1],DFT[312501],DFT[625001],DFT[937501],DFT[1250001])
    DFT_flux2030_emax = (DFT[2],DFT[312502],DFT[625002],DFT[937502],DFT[1250002])
    DFT_flux3040_emax = (DFT[3],DFT[312503],DFT[625003],DFT[937503],DFT[1250003])
    
    Photon_flux10_emax = (Photon[0],Photon[312500],Photon[625000],Photon[937500],Photon[1250000])
    Photon_flux1020_emax = (Photon[1],Photon[312501],Photon[625001],Photon[937501],Photon[1250001])
    Photon_flux2030_emax = (Photon[2],Photon[312502],Photon[625002],Photon[937502],Photon[1250002])
    Photon_flux3040_emax = (Photon[3],Photon[312503],Photon[625003],Photon[937503],Photon[1250003])
    
    ### Flux for 1st PSF and 1st Galaxy
    y_flux10_emax = (yimg[0],yimg[312500],yimg[625000],yimg[937500],yimg[1250000])
    y_flux1020_emax = (yimg[1],yimg[312501],yimg[625001],yimg[937501],yimg[1250001])
    y_flux2030_emax = (yimg[2],yimg[312502],yimg[625002],yimg[937502],yimg[1250002])
    y_flux3040_emax = (yimg[3],yimg[312503],yimg[625003],yimg[937503],yimg[1250003])
    
    #### the DFT minus Photon-Shooting time for the first psf and first galaxy
    
    xnew_1_emax=[]
    xnew_2_emax=[]
    xnew_3_emax=[]
    xnew_4_emax=[]
    xnew_5_emax=[]
    for a,b,c,d,e,f,g,h in zip (DFT_flux10_emax, DFT_flux1020_emax, DFT_flux2030_emax, DFT_flux3040_emax, Photon_flux10_emax, Photon_flux1020_emax, Photon_flux2030_emax, Photon_flux3040_emax):
        x1=a-b
        x2=c-d
        x3=e-f
        x4=g-h
        xnew_1_emax= np.append(xnew_1_emax,x1)
        xnew_2_emax= np.append(xnew_2_emax,x2)
        xnew_3_emax= np.append(xnew_3_emax,x3)
        xnew_4_emax= np.append(xnew_4_emax,x4)
    
    
### emin
    
    #### Time for the first PSF and first Gaussian
    
    DFT_flux10_emin = (DFT[0],DFT[1562500],DFT[3125000],DFT[4687500],DFT[6250000])
    DFT_flux1020_emin = (DFT[1],DFT[1562500],DFT[3125000],DFT[4687500],DFT[6250000])
    DFT_flux2030_emin = (DFT[2],DFT[1562500],DFT[3125000],DFT[4687500],DFT[6250000])
    DFT_flux3040_emin = (DFT[3],DFT[1562500],DFT[3125000],DFT[4687500],DFT[6250000])
    
    Photon_flux10_emin = (Photon[0],Photon[1562500],Photon[3125000],Photon[4687500],Photon[6250000])
    Photon_flux1020_emin = (Photon[1],Photon[1562500],Photon[3125000],Photon[4687500],Photon[6250000])
    Photon_flux2030_emin = (Photon[2],Photon[1562500],Photon[3125000],Photon[4687500],Photon[6250000])
    Photon_flux3040_emin = (Photon[3],Photon[1562500],Photon[3125000],Photon[4687500],Photon[6250000])
    
    ### Flux for 1st PSF and 1st Galaxy
    y_flux10_emin = (yimg[0],yimg[1562500],yimg[3125000],yimg[4687500],yimg[6250000])
    y_flux1020_emin = (yimg[1],yimg[1562500],yimg[3125000],yimg[4687500],yimg[6250000])
    y_flux2030_emin = (yimg[2],yimg[1562500],yimg[3125000],yimg[4687500],yimg[6250000])
    y_flux3040_emin = (yimg[3],yimg[1562500],yimg[3125000],yimg[4687500],yimg[6250000])
    
    #### the DFT minus Photon-Shooting time for the first psf and first galaxy
    
    xnew_1_emin=[]
    xnew_2_emin=[]
    xnew_3_emin=[]
    xnew_4_emin=[]
    xnew_5_emin=[]
    for a,b,c,d,e,f,g,h in zip (DFT_flux10_emin, DFT_flux1020_emin, DFT_flux2030_emin, DFT_flux3040_emin, Photon_flux10_emin, Photon_flux1020_emin, Photon_flux2030_emin, Photon_flux3040_emin):
        x1=a-b
        x2=c-d
        x3=e-f
        x4=g-h
        xnew_1_emin= np.append(xnew_1_emin,x1)
        xnew_2_emin= np.append(xnew_2_emin,x2)
        xnew_3_emin= np.append(xnew_3_emin,x3)
        xnew_4_emin= np.append(xnew_4_emin,x4)


### hlrmax

#### Time for the first PSF and first Gaussian

    DFT_flux10_hlrmax = (DFT[0],DFT[7812500],DFT[15625000],DFT[23437500],DFT[3125000])
    DFT_flux1020_hlrmax = (DFT[1],DFT[7812501],DFT[15625001],DFT[23437501],DFT[3125001])
    DFT_flux2030_hlrmax = (DFT[2],DFT[7812502],DFT[15625002],DFT[23437502],DFT[3125002])
    DFT_flux3040_hlrmax = (DFT[3],DFT[7812503],DFT[15625003],DFT[23437503],DFT[3125003])
    
    Photon_flux10_hlrmax = (Photon[0],Photon[7812500],Photon[15625000],Photon[23437500],Photon[3125000])
    Photon_flux1020_hlrmax = (Photon[1],Photon[7812501],Photon[15625001],Photon[23437501],Photon[3125001])
    Photon_flux2030_hlrmax = (Photon[2],Photon[7812502],Photon[15625002],Photon[23437502],Photon[3125002])
    Photon_flux3040_hlrmax = (Photon[3],Photon[7812503],Photon[15625003],Photon[23437503],Photon[3125003])
    
    ### Flux for 1st PSF and 1st Galaxy
    y_flux10_hlrmax = (yimg[0],yimg[7812500],yimg[15625000],yimg[23437500],yimg[3125000])
    y_flux1020_hlrmax = (yimg[1],yimg[7812501],yimg[15625001],yimg[23437501],yimg[3125001])
    y_flux2030_hlrmax = (yimg[2],yimg[7812502],yimg[15625002],yimg[23437502],yimg[3125002])
    y_flux3040_hlrmax = (yimg[3],yimg[7812503],yimg[15625003],yimg[23437503],yimg[3125003])
    
    #### the DFT minus Photon-Shooting time for the first psf and first galaxy
    
    xnew_1_hlrmax=[]
    xnew_2_hlrmax=[]
    xnew_3_hlrmax=[]
    xnew_4_hlrmax=[]
    xnew_5_hlrmax=[]
    for a,b,c,d,e,f,g,h in zip (DFT_flux10_hlrmax, DFT_flux1020_hlrmax, DFT_flux2030_hlrmax, DFT_flux3040_hlrmax, Photon_flux10_hlrmax, Photon_flux1020_hlrmax, Photon_flux2030_hlrmax, Photon_flux3040_hlrmax):
        x1=a-b
        x2=c-d
        x3=e-f
        x4=g-h
        xnew_1_hlrmax= np.append(xnew_1_hlrmax,x1)
        xnew_2_hlrmax= np.append(xnew_2_hlrmax,x2)
        xnew_3_hlrmax= np.append(xnew_3_hlrmax,x3)
        xnew_4_hlrmax= np.append(xnew_4_hlrmax,x4)

### hlrmin

    #### Time for the first PSF and first Gaussian

    DFT_flux10_hlrmin = (DFT[0],DFT[39062500],DFT[78125000],DFT[117187500],DFT[156250000])
    DFT_flux1020_hlrmin = (DFT[1],DFT[39062501],DFT[78125001],DFT[117187501],DFT[156250001])
    DFT_flux2030_hlrmin = (DFT[2],DFT[39062502],DFT[78125002],DFT[117187502],DFT[156250002])
    DFT_flux3040_hlrmin = (DFT[3],DFT[39062503],DFT[78125003],DFT[117187503],DFT[156250003])
    
    Photon_flux10_hlrmin = (Photon[0],Photon[39062500],Photon[78125003],Photon[117187500],Photon[156250003])
    Photon_flux1020_hlrmin = (Photon[1],Photon[39062500],Photon[78125003],Photon[117187500],Photon[156250003])
    Photon_flux2030_hlrmin = (Photon[2],Photon[39062500],Photon[78125003],Photon[117187500],Photon[156250003])
    Photon_flux3040_hlrmin = (Photon[3],Photon[39062500],Photon[78125003],Photon[117187500],Photon[156250003])
    
    ### Flux for 1st PSF and 1st Galaxy
    y_flux10_hlrmin = (yimg[0],yimg[39062500],yimg[78125003],yimg[117187500],yimg[156250003])
    y_flux1020_hlrmin = (yimg[1],yimg[39062500],yimg[78125003],yimg[117187500],yimg[156250003])
    y_flux2030_hlrmin = (yimg[2],yimg[39062500],yimg[78125003],yimg[117187500],yimg[156250003])
    y_flux3040_hlrmin = (yimg[3],yimg[39062500],yimg[78125003],yimg[117187500],yimg[156250003])
    
    #### the DFT minus Photon-Shooting time for the first psf and first galaxy
    
    xnew_1_hlrmin=[]
    xnew_2_hlrmin=[]
    xnew_3_hlrmin=[]
    xnew_4_hlrmin=[]
    xnew_5_hlrmin=[]
    for a,b,c,d,e,f,g,h in zip (DFT_flux10_hlrmin, DFT_flux1020_hlrmin, DFT_flux2030_hlrmin, DFT_flux3040_hlrmin, Photon_flux10_hlrmin, Photon_flux1020_hlrmin, Photon_flux2030_hlrmin, Photon_flux3040_hlrmin):
        x1=a-b
        x2=c-d
        x3=e-f
        x4=g-h
        xnew_1_hlrmin= np.append(xnew_1_hlrmin,x1)
        xnew_2_hlrmin= np.append(xnew_2_hlrmin,x2)
        xnew_3_hlrmin= np.append(xnew_3_hlrmin,x3)
        xnew_4_hlrmin= np.append(xnew_4_hlrmin,x4)
    
### n
    
    #### Time for the first PSF and first Gaussian
    
    DFT_flux10_n = (DFT[0],DFT[195312500],DFT[390625000],DFT[585937500],DFT[781250000])
    DFT_flux1020_n = (DFT[1],DFT[195312501],DFT[390625001],DFT[585937501],DFT[781250001])
    DFT_flux2030_n = (DFT[2],DFT[195312502],DFT[390625002],DFT[585937502],DFT[781250002])
    DFT_flux3040_n = (DFT[3],DFT[195312503],DFT[390625003],DFT[585937503],DFT[781250003])
    
    Photon_flux10_n = (Photon[0],Photon[195312500],Photon[390625000],Photon[585937500],Photon[781250000])
    Photon_flux1020_n = (Photon[1],Photon[195312501],Photon[390625001],Photon[585937501],Photon[781250001])
    Photon_flux2030_n = (Photon[2],Photon[195312502],Photon[390625002],Photon[585937502],Photon[781250002])
    Photon_flux3040_n = (Photon[3],Photon[195312503],Photon[390625003],Photon[585937503],Photon[781250003])
    
    ### Flux for 1st PSF and 1st Galaxy
    y_flux10_n = (yimg[0],yimg[195312500],yimg[390625000],yimg[585937500],yimg[781250000])
    y_flux1020_n = (yimg[1],yimg[195312501],yimg[390625001],yimg[585937501],yimg[781250001])
    y_flux2030_n = (yimg[2],yimg[195312502],yimg[390625002],yimg[585937502],yimg[781250002])
    y_flux3040_n = (yimg[3],yimg[195312503],yimg[390625003],yimg[585937503],yimg[781250003])
    
    #### the DFT minus Photon-Shooting time for the first psf and first galaxy
    
    xnew_1_n=[]
    xnew_2_n=[]
    xnew_3_n=[]
    xnew_4_n=[]
    xnew_5_n=[]
    for a,b,c,d,e,f,g,h in zip (DFT_flux10_n, DFT_flux1020_n, DFT_flux2030_n, DFT_flux3040_n, Photon_flux10_n, Photon_flux1020_n, Photon_flux2030_n, Photon_flux3040_n):
        x1=a-b
        x2=c-d
        x3=e-f
        x4=g-h
        xnew_1_n= np.append(xnew_1_n,x1)
        xnew_2_n= np.append(xnew_2_n,x2)
        xnew_3_n= np.append(xnew_3_n,x3)
        xnew_4_n= np.append(xnew_4_n,x4)


#fig1 for DFTmin
    #subplot command
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    #1st plot
    l1, = ax1.plot(ranges_DFTmin,DFT_flux10_DFTmin, 'b-', label='l1')
    l2, = ax1.plot(ranges_DFTmin,Photon_flux10_DFTmin, 'g-', label='l2')
    l3, = ax1.plot(ranges_DFTmin,DFT_flux1020_DFTmin,'b--',label='l3')
    l4, = ax1.plot(ranges_DFTmin,Photon_flux1020_DFTmin,'g--',label='l4')
    l5, = ax1.plot(ranges_DFTmin,DFT_flux2030_DFTmin,'b*',label='l5')
    l6, = ax1.plot(ranges_DFTmin,Photon_flux2030_DFTmin,'g*',label='l6')
    l7, = ax1.plot(ranges_DFTmin,DFT_flux3040_DFTmin,'b.',label='l7')
    l8, = ax1.plot(ranges_DFTmin,Photon_flux3040_DFTmin,'g.',label='l8')
    #2nd plot
    l9, = ax2.plot(ranges_DFTmin,xnew_1_DFTmin, 'b-', label='l9')
    l10, = ax2.plot(ranges_DFTmin,xnew_2_DFTmin, 'g-', label='l10')
    l11, = ax2.plot(ranges_DFTmin,xnew_3_DFTmin, 'y--',label='l11')
    l12, = ax2.plot(ranges_DFTmin,xnew_4_DFTmin, 'r--',label='l12')
    l13, = ax2.plot(ranges_DFTmin,y, 'b-')
    #3rd plot
    l14, = ax3.plot(ranges_DFTmin, y_flux10_DFTmin, 'b-', label='l14')
    l15, = ax3.plot(ranges_DFTmin, y_flux1020_DFTmin, 'g-', label='l15')
    l16, = ax3.plot(ranges_DFTmin, y_flux2030_DFTmin, 'y--',label='l16')
    l17, = ax3.plot(ranges_DFTmin, y_flux3040_DFTmin, 'r--',label='l17')
    #labeling
    ax1.set_title('Gaussian PSF with Gaussian Galaxy Profile for DFTmin')
    ax1.legend([l1,l2,l3,l4,l5,l6,l7,l8],['DFT: Flux 10','Photon: Flux 10','DFT: Flux 1020','Photon: Flux 1020','DFT: Flux 2030','Photon: Flux 2030','DFT: Flux 3040','Photon: Flux 3040'], loc='best')
    ax2.legend([l9,l10,l11,l2],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    ax3.legend([l14,l15,l16,l17],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    plt.xlabel('DFT min')
    ax1.set_ylabel('Time')
    ax2.set_ylabel('DFT minus Photon-Shooting')
    ax3.set_ylabel('Flux')
    #arrange subplots and save
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.savefig('DFTmin_varried_1.png')

#fig2 for photon shooting accuracy
    #subplot command
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    #1st plot
    l1, = ax1.plot(ranges_photonaim,DFT_flux10_aim, 'b-', label='l1')
    l2, = ax1.plot(ranges_photonaim,Photon_flux10_aim, 'g-', label='l2')
    l3, = ax1.plot(ranges_photonaim,DFT_flux1020_aim,'b--',label='l3')
    l4, = ax1.plot(ranges_photonaim,Photon_flux1020_aim,'g--',label='l4')
    l5, = ax1.plot(ranges_photonaim,DFT_flux2030_aim,'b*',label='l5')
    l6, = ax1.plot(ranges_photonaim,Photon_flux2030_aim,'g*',label='l6')
    l7, = ax1.plot(ranges_photonaim,DFT_flux3040_aim,'b.',label='l7')
    l8, = ax1.plot(ranges_photonaim,Photon_flux3040_aim,'g.',label='l8')
    #2nd plot
    l9, = ax2.plot(ranges_photonaim,xnew_1_aim, 'b-', label='l9')
    l10, = ax2.plot(ranges_photonaim,xnew_2_aim, 'g-', label='l10')
    l11, = ax2.plot(ranges_photonaim,xnew_3_aim, 'y--',label='l11')
    l12, = ax2.plot(ranges_photonaim,xnew_4_aim, 'r--',label='l12')
    l13, = ax2.plot(ranges_photonaim,y, 'b-')
    #3rd plot
    l14, = ax3.plot(ranges_photonaim, y_flux10_aim, 'b-', label='l14')
    l15, = ax3.plot(ranges_photonaim, y_flux1020_aim, 'g-', label='l15')
    l16, = ax3.plot(ranges_photonaim, y_flux2030_aim, 'y--',label='l16')
    l17, = ax3.plot(ranges_photonaim, y_flux3040_aim, 'r--',label='l17')
    #labeling
    ax1.set_title('Gaussian PSF with Gaussian Galaxy Profile for Photon-shooting aim')
    ax1.legend([l1,l2,l3,l4,l5,l6,l7,l8],['DFT: Flux 10','Photon: Flux 10','DFT: Flux 1020','Photon: Flux 1020','DFT: Flux 2030','Photon: Flux 2030','DFT: Flux 3040','Photon: Flux 3040'], loc='best')
    ax2.legend([l9,l10,l11,l2],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    ax3.legend([l14,l15,l16,l17],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    plt.xlabel('photon-shooting aim')
    ax1.set_ylabel('Time')
    ax2.set_ylabel('DFT minus Photon-Shooting')
    ax3.set_ylabel('Flux')
    #arrange subplots and save
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.savefig('aim_varried_1.png')

#fig3 for k-value
    #subplot command
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    #1st plot
    l1, = ax1.plot(ranges_DFTaim,DFT_flux10_k, 'b-', label='l1')
    l2, = ax1.plot(ranges_DFTaim,Photon_flux10_k, 'g-', label='l2')
    l3, = ax1.plot(ranges_DFTaim,DFT_flux1020_k,'b--',label='l3')
    l4, = ax1.plot(ranges_DFTaim,Photon_flux1020_k,'g--',label='l4')
    l5, = ax1.plot(ranges_DFTaim,DFT_flux2030_k,'b*',label='l5')
    l6, = ax1.plot(ranges_DFTaim,Photon_flux2030_k,'g*',label='l6')
    l7, = ax1.plot(ranges_DFTaim,DFT_flux3040_k,'b.',label='l7')
    l8, = ax1.plot(ranges_DFTaim,Photon_flux3040_k,'g.',label='l8')
    #2nd plot
    l9, = ax2.plot(ranges_DFTaim,xnew_1_k, 'b-', label='l9')
    l10, = ax2.plot(ranges_DFTaim,xnew_2_k, 'g-', label='l10')
    l11, = ax2.plot(ranges_DFTaim,xnew_3_k, 'y--',label='l11')
    l12, = ax2.plot(ranges_DFTaim,xnew_4_k, 'r--',label='l12')
    l13, = ax2.plot(ranges_DFTaim,y, 'b-')
    #3rd plot
    l14, = ax3.plot(ranges_DFTaim, y_flux10_k, 'b-', label='l14')
    l15, = ax3.plot(ranges_DFTaim, y_flux1020_k, 'g-', label='l15')
    l16, = ax3.plot(ranges_DFTaim, y_flux2030_k, 'y--',label='l16')
    l17, = ax3.plot(ranges_DFTaim, y_flux3040_k, 'r--',label='l17')
    #labeling
    ax1.set_title('Gaussian PSF with Gaussian Galaxy Profile for K-value')
    ax1.legend([l1,l2,l3,l4,l5,l6,l7,l8],['DFT: Flux 10','Photon: Flux 10','DFT: Flux 1020','Photon: Flux 1020','DFT: Flux 2030','Photon: Flux 2030','DFT: Flux 3040','Photon: Flux 3040'], loc='best')
    ax2.legend([l9,l10,l11,l2],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    ax3.legend([l14,l15,l16,l17],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    plt.xlabel('k-value')
    ax1.set_ylabel('Time')
    ax2.set_ylabel('DFT minus Photon-Shooting')
    ax3.set_ylabel('Flux')
    #arrange subplots and save
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.savefig('kvalue_varried_1.png')

#fig4 for maxk
    #subplot command
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    #1st plot
    l1, = ax1.plot(ranges_excludedk,DFT_flux10_maxk, 'b-', label='l1')
    l2, = ax1.plot(ranges_excludedk,Photon_flux10_maxk, 'g-', label='l2')
    l3, = ax1.plot(ranges_excludedk,DFT_flux1020_maxk,'b--',label='l3')
    l4, = ax1.plot(ranges_excludedk,Photon_flux1020_maxk,'g--',label='l4')
    l5, = ax1.plot(ranges_excludedk,DFT_flux2030_maxk,'b*',label='l5')
    l6, = ax1.plot(ranges_excludedk,Photon_flux2030_maxk,'g*',label='l6')
    l7, = ax1.plot(ranges_excludedk,DFT_flux3040_maxk,'b.',label='l7')
    l8, = ax1.plot(ranges_excludedk,Photon_flux3040_maxk,'g.',label='l8')
    #2nd plot
    l9, = ax2.plot(ranges_excludedk,xnew_1_maxk, 'b-', label='l9')
    l10, = ax2.plot(ranges_excludedk,xnew_2_maxk, 'g-', label='l10')
    l11, = ax2.plot(ranges_excludedk,xnew_3_maxk, 'y--',label='l11')
    l12, = ax2.plot(ranges_excludedk,xnew_4_maxk, 'r--',label='l12')
    l13, = ax2.plot(ranges_excludedk,y, 'b-')
    #3rd plot
    l14, = ax3.plot(ranges_excludedk, y_flux10_maxk, 'b-', label='l14')
    l15, = ax3.plot(ranges_excludedk, y_flux1020_maxk, 'g-', label='l15')
    l16, = ax3.plot(ranges_excludedk, y_flux2030_maxk, 'y--',label='l16')
    l17, = ax3.plot(ranges_excludedk, y_flux3040_maxk, 'r--',label='l17')
    #labeling
    ax1.set_title('Gaussian PSF with Gaussian Galaxy Profile for maxk')
    ax1.legend([l1,l2,l3,l4,l5,l6,l7,l8],['DFT: Flux 10','Photon: Flux 10','DFT: Flux 1020','Photon: Flux 1020','DFT: Flux 2030','Photon: Flux 2030','DFT: Flux 3040','Photon: Flux 3040'], loc='best')
    ax2.legend([l9,l10,l11,l2],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    ax3.legend([l14,l15,l16,l17],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    plt.xlabel('maxk')
    ax1.set_ylabel('Time')
    ax2.set_ylabel('DFT minus Photon-Shooting')
    ax3.set_ylabel('Flux')
    #arrange subplots and save
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.savefig('maxk_varried_1.png')

#fig4 for maxfold
    #subplot command
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    #1st plot
    l1, = ax1.plot(ranges_maxfold,DFT_flux10_maxfold, 'b-', label='l1')
    l2, = ax1.plot(ranges_maxfold,Photon_flux10_maxfold, 'g-', label='l2')
    l3, = ax1.plot(ranges_maxfold,DFT_flux1020_maxfold,'b--',label='l3')
    l4, = ax1.plot(ranges_maxfold,Photon_flux1020_maxfold,'g--',label='l4')
    l5, = ax1.plot(ranges_maxfold,DFT_flux2030_maxfold,'b*',label='l5')
    l6, = ax1.plot(ranges_maxfold,Photon_flux2030_maxfold,'g*',label='l6')
    l7, = ax1.plot(ranges_maxfold,DFT_flux3040_maxfold,'b.',label='l7')
    l8, = ax1.plot(ranges_maxfold,Photon_flux3040_maxfold,'g.',label='l8')
    #2nd plot
    l9, = ax2.plot(ranges_maxfold,xnew_1_maxfold, 'b-', label='l9')
    l10, = ax2.plot(ranges_maxfold,xnew_2_maxfold, 'g-', label='l10')
    l11, = ax2.plot(ranges_maxfold,xnew_3_maxfold, 'y--',label='l11')
    l12, = ax2.plot(ranges_maxfold,xnew_4_maxfold, 'r--',label='l12')
    l13, = ax2.plot(ranges_maxfold,y, 'b-')
    #3rd plot
    l14, = ax3.plot(ranges_maxfold, y_flux10_maxfold, 'b-', label='l14')
    l15, = ax3.plot(ranges_maxfold, y_flux1020_maxfold, 'g-', label='l15')
    l16, = ax3.plot(ranges_maxfold, y_flux2030_maxfold, 'y--',label='l16')
    l17, = ax3.plot(ranges_maxfold, y_flux3040_maxfold, 'r--',label='l17')
    #labeling
    ax1.set_title('Gaussian PSF with Gaussian Galaxy Profile for max threshold fold')
    ax1.legend([l1,l2,l3,l4,l5,l6,l7,l8],['DFT: Flux 10','Photon: Flux 10','DFT: Flux 1020','Photon: Flux 1020','DFT: Flux 2030','Photon: Flux 2030','DFT: Flux 3040','Photon: Flux 3040'], loc='best')
    ax2.legend([l9,l10,l11,l2],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    ax3.legend([l14,l15,l16,l17],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    plt.xlabel('max fold')
    ax1.set_ylabel('Time')
    ax2.set_ylabel('DFT minus Photon-Shooting')
    ax3.set_ylabel('Flux')
    #arrange subplots and save
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.savefig('maxfold_varried_1.png')


#fig5 for emax
    #subplot command
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    #1st plot
    l1, = ax1.plot(ranges_emax,DFT_flux10_emax, 'b-', label='l1')
    l2, = ax1.plot(ranges_emax,Photon_flux10_emax, 'g-', label='l2')
    l3, = ax1.plot(ranges_emax,DFT_flux1020_emax,'b--',label='l3')
    l4, = ax1.plot(ranges_emax,Photon_flux1020_emax,'g--',label='l4')
    l5, = ax1.plot(ranges_emax,DFT_flux2030_emax,'b*',label='l5')
    l6, = ax1.plot(ranges_emax,Photon_flux2030_emax,'g*',label='l6')
    l7, = ax1.plot(ranges_emax,DFT_flux3040_emax,'b.',label='l7')
    l8, = ax1.plot(ranges_emax,Photon_flux3040_emax,'g.',label='l8')
    #2nd plot
    l9, = ax2.plot(ranges_emax,xnew_1_emax, 'b-', label='l9')
    l10, = ax2.plot(ranges_emax,xnew_2_emax, 'g-', label='l10')
    l11, = ax2.plot(ranges_emax,xnew_3_emax, 'y--',label='l11')
    l12, = ax2.plot(ranges_emax,xnew_4_emax, 'r--',label='l12')
    l13, = ax2.plot(ranges_emax,y, 'b-')
    #3rd plot
    l14, = ax3.plot(ranges_emax, y_flux10_memax, 'b-', label='l14')
    l15, = ax3.plot(ranges_emax, y_flux1020_emax, 'g-', label='l15')
    l16, = ax3.plot(ranges_emax, y_flux2030_emax, 'y--',label='l16')
    l17, = ax3.plot(ranges_emax, y_flux3040_emax, 'r--',label='l17')
    #labeling
    ax1.set_title('Gaussian PSF with Gaussian Galaxy Profile for e max')
    ax1.legend([l1,l2,l3,l4,l5,l6,l7,l8],['DFT: Flux 10','Photon: Flux 10','DFT: Flux 1020','Photon: Flux 1020','DFT: Flux 2030','Photon: Flux 2030','DFT: Flux 3040','Photon: Flux 3040'], loc='best')
    ax2.legend([l9,l10,l11,l2],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    ax3.legend([l14,l15,l16,l17],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    plt.xlabel('e max')
    ax1.set_ylabel('Time')
    ax2.set_ylabel('DFT minus Photon-Shooting')
    ax3.set_ylabel('Flux')
    #arrange subplots and save
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.savefig('e max_varried_1.png')

#fig6 for emin
    #subplot command
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    #1st plot
    l1, = ax1.plot(ranges_emin,DFT_flux10_emin, 'b-', label='l1')
    l2, = ax1.plot(ranges_emin,Photon_flux10_emin, 'g-', label='l2')
    l3, = ax1.plot(ranges_emin,DFT_flux1020_emin,'b--',label='l3')
    l4, = ax1.plot(ranges_emin,Photon_flux1020_emin,'g--',label='l4')
    l5, = ax1.plot(ranges_emin,DFT_flux2030_emin,'b*',label='l5')
    l6, = ax1.plot(ranges_emin,Photon_flux2030_emin,'g*',label='l6')
    l7, = ax1.plot(ranges_emin,DFT_flux3040_emin,'b.',label='l7')
    l8, = ax1.plot(ranges_emin,Photon_flux3040_emin,'g.',label='l8')
    #2nd plot
    l9, = ax2.plot(ranges_emin,xnew_1_emin, 'b-', label='l9')
    l10, = ax2.plot(ranges_emin,xnew_2_memin, 'g-', label='l10')
    l11, = ax2.plot(ranges_emin,xnew_3_emin, 'y--',label='l11')
    l12, = ax2.plot(ranges_emin,xnew_4_emin, 'r--',label='l12')
    l13, = ax2.plot(ranges_emin,y, 'b-')
    #3rd plot
    l14, = ax3.plot(ranges_emin, y_flux10_emin, 'b-', label='l14')
    l15, = ax3.plot(ranges_emin, y_flux1020_emin, 'g-', label='l15')
    l16, = ax3.plot(ranges_emin, y_flux2030_emin, 'y--',label='l16')
    l17, = ax3.plot(ranges_emin, y_flux3040_emin, 'r--',label='l17')
    #labeling
    ax1.set_title('Gaussian PSF with Gaussian Galaxy Profile for e min')
    ax1.legend([l1,l2,l3,l4,l5,l6,l7,l8],['DFT: Flux 10','Photon: Flux 10','DFT: Flux 1020','Photon: Flux 1020','DFT: Flux 2030','Photon: Flux 2030','DFT: Flux 3040','Photon: Flux 3040'], loc='best')
    ax2.legend([l9,l10,l11,l2],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    ax3.legend([l14,l15,l16,l17],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    plt.xlabel('e min')
    ax1.set_ylabel('Time')
    ax2.set_ylabel('DFT minus Photon-Shooting')
    ax3.set_ylabel('Flux')
    #arrange subplots and save
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.savefig('emin_varried_1.png')


#fig7 for hlrmax
    #subplot command
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    #1st plot
    l1, = ax1.plot(ranges_hlrmax,DFT_flux10_hlrmax, 'b-', label='l1')
    l2, = ax1.plot(ranges_hlrmax,Photon_flux10_hlrmax, 'g-', label='l2')
    l3, = ax1.plot(ranges_hlrmax,DFT_flux1020_hlrmax,'b--',label='l3')
    l4, = ax1.plot(ranges_hlrmax,Photon_flux1020_hlrmax,'g--',label='l4')
    l5, = ax1.plot(ranges_hlrmax,DFT_flux2030_hlrmax,'b*',label='l5')
    l6, = ax1.plot(ranges_hlrmax,Photon_flux2030_hlrmax,'g*',label='l6')
    l7, = ax1.plot(ranges_hlrmax,DFT_flux3040_hlrmax,'b.',label='l7')
    l8, = ax1.plot(ranges_hlrmax,Photon_flux3040_hlrmax,'g.',label='l8')
    #2nd plot
    l9, = ax2.plot(ranges_hlrmax,xnew_1_hlrmax, 'b-', label='l9')
    l10, = ax2.plot(ranges_hlrmax,xnew_2_hlrmax, 'g-', label='l10')
    l11, = ax2.plot(ranges_hlrmax,xnew_3_hlrmax, 'y--',label='l11')
    l12, = ax2.plot(ranges_hlrmax,xnew_4_hlrmax, 'r--',label='l12')
    l13, = ax2.plot(ranges_hlrmax,y, 'b-')
    #3rd plot
    l14, = ax3.plot(ranges_hlrmax, y_flux10_hlrmax, 'b-', label='l14')
    l15, = ax3.plot(ranges_hlrmax, y_flux1020_hlrmax, 'g-', label='l15')
    l16, = ax3.plot(ranges_hlrmax, y_flux2030_hlrmax, 'y--',label='l16')
    l17, = ax3.plot(ranges_hlrmax, y_flux3040_hlrmax, 'r--',label='l17')
    #labeling
    ax1.set_title('Gaussian PSF with Gaussian Galaxy Profile for hlr max')
    ax1.legend([l1,l2,l3,l4,l5,l6,l7,l8],['DFT: Flux 10','Photon: Flux 10','DFT: Flux 1020','Photon: Flux 1020','DFT: Flux 2030','Photon: Flux 2030','DFT: Flux 3040','Photon: Flux 3040'], loc='best')
    ax2.legend([l9,l10,l11,l2],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    ax3.legend([l14,l15,l16,l17],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    plt.xlabel('hlr max')
    ax1.set_ylabel('Time')
    ax2.set_ylabel('DFT minus Photon-Shooting')
    ax3.set_ylabel('Flux')
    #arrange subplots and save
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.savefig('hlrmax_varried_1.png')

#fig8 for hlrmin
    #subplot command
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    #1st plot
    l1, = ax1.plot(ranges_hlrmin,DFT_flux10_hlrmin, 'b-', label='l1')
    l2, = ax1.plot(ranges_hlrmin,Photon_flux10_hlrmin, 'g-', label='l2')
    l3, = ax1.plot(ranges_hlrmin,DFT_flux1020_hlrmin,'b--',label='l3')
    l4, = ax1.plot(ranges_hlrmin,Photon_flux1020_hlrmin,'g--',label='l4')
    l5, = ax1.plot(ranges_hlrmin,DFT_flux2030_hlrmin,'b*',label='l5')
    l6, = ax1.plot(ranges_hlrmin,Photon_flux2030_hlrmin,'g*',label='l6')
    l7, = ax1.plot(ranges_hlrmin,DFT_flux3040_hlrmin,'b.',label='l7')
    l8, = ax1.plot(ranges_hlrmin,Photon_flux3040_hlrmin,'g.',label='l8')
    #2nd plot
    l9, = ax2.plot(ranges_hlrmin,xnew_1_hlrmin, 'b-', label='l9')
    l10, = ax2.plot(ranges_hlrmin,xnew_2_hlrmin, 'g-', label='l10')
    l11, = ax2.plot(ranges_hlrmin,xnew_3_hlrmin, 'y--',label='l11')
    l12, = ax2.plot(ranges_hlrmin,xnew_4_hlrmin, 'r--',label='l12')
    l13, = ax2.plot(ranges_hlrmin,y, 'b-')
    #3rd plot
    l14, = ax3.plot(ranges_hlrmin, y_flux10_hlrmin, 'b-', label='l14')
    l15, = ax3.plot(ranges_hlrmin, y_flux1020_hlrmin, 'g-', label='l15')
    l16, = ax3.plot(ranges_hlrmin, y_flux2030_hlrmin, 'y--',label='l16')
    l17, = ax3.plot(ranges_hlrmin, y_flux3040_hlrmin, 'r--',label='l17')
    #labeling
    ax1.set_title('Gaussian PSF with Gaussian Galaxy Profile for hlr min')
    ax1.legend([l1,l2,l3,l4,l5,l6,l7,l8],['DFT: Flux 10','Photon: Flux 10','DFT: Flux 1020','Photon: Flux 1020','DFT: Flux 2030','Photon: Flux 2030','DFT: Flux 3040','Photon: Flux 3040'], loc='best')
    ax2.legend([l9,l10,l11,l2],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    ax3.legend([l14,l15,l16,l17],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    plt.xlabel('hlr min')
    ax1.set_ylabel('Time')
    ax2.set_ylabel('DFT minus Photon-Shooting')
    ax3.set_ylabel('Flux')
    #arrange subplots and save
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.savefig('hlrmin_varried_1.png')

#fig9 for n
    #subplot command
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    #1st plot
    l1, = ax1.plot(ranges_n,DFT_flux10_n, 'b-', label='l1')
    l2, = ax1.plot(ranges_n,Photon_flux10_n, 'g-', label='l2')
    l3, = ax1.plot(ranges_n,DFT_flux1020_n,'b--',label='l3')
    l4, = ax1.plot(ranges_n,Photon_flux1020_n,'g--',label='l4')
    l5, = ax1.plot(ranges_n,DFT_flux2030_n,'b*',label='l5')
    l6, = ax1.plot(ranges_n,Photon_flux2030_n,'g*',label='l6')
    l7, = ax1.plot(ranges_n,DFT_flux3040_n,'b.',label='l7')
    l8, = ax1.plot(ranges_n,Photon_flux3040_n,'g.',label='l8')
    #2nd plot
    l9, = ax2.plot(ranges_n,xnew_1_n, 'b-', label='l9')
    l10, = ax2.plot(ranges_n,xnew_2_n, 'g-', label='l10')
    l11, = ax2.plot(ranges_n,xnew_3_n, 'y--',label='l11')
    l12, = ax2.plot(ranges_n,xnew_4_n, 'r--',label='l12')
    l13, = ax2.plot(ranges_n,y, 'b-')
    #3rd plot
    l14, = ax3.plot(ranges_n, y_flux10_n, 'b-', label='l14')
    l15, = ax3.plot(ranges_n, y_flux1020_n, 'g-', label='l15')
    l16, = ax3.plot(ranges_n, y_flux2030_n, 'y--',label='l16')
    l17, = ax3.plot(ranges_n, y_flux3040_n, 'r--',label='l17')
    #labeling
    ax1.set_title('Gaussian PSF with Gaussian Galaxy Profile for nx & ny')
    ax1.legend([l1,l2,l3,l4,l5,l6,l7,l8],['DFT: Flux 10','Photon: Flux 10','DFT: Flux 1020','Photon: Flux 1020','DFT: Flux 2030','Photon: Flux 2030','DFT: Flux 3040','Photon: Flux 3040'], loc='best')
    ax2.legend([l9,l10,l11,l2],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    ax3.legend([l14,l15,l16,l17],['Flux 10','Flux 1020','Flux 2030','Flux 3040'], loc='best')
    plt.xlabel('n')
    ax1.set_ylabel('Time')
    ax2.set_ylabel('DFT minus Photon-Shooting')
    ax3.set_ylabel('Flux')
    #arrange subplots and save
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.savefig('n_varried_1.png')



    ### breakdown of psf and galaxy types as well as overal timing statistics
    logger.info('')
    logger.info('Some timing statistics:')
    logger.info('   Total time for setup steps = %f',setup_times)
    logger.info('   Total time for regular fft drawing = %f',fft_times)
    logger.info('   Total time for photon shooting = %f',phot_times)
    logger.info('   Total time for adding noise = %f',noise_times)
    logger.info('')
    logger.info('Breakdown by PSF type:')
    for ipsf in range(len(psfs)):
        logger.info('   %s: Total time = %f  (fft: %f, phot: %f)',
                    psf_names[ipsf],psf_times[ipsf],psf_fft_times[ipsf],psf_phot_times[ipsf])
    logger.info('')
    logger.info('Breakdown by Galaxy type:')
    for igal in range(len(gals)):
        logger.info('   %s: Total time = %f  (fft: %f, phot: %f)',
                    gal_names[igal],gal_times[igal],gal_fft_times[igal],gal_phot_times[igal])
    logger.info('')
    
    ### compress into a gzip file and save a a cube
    galsim.fits.writeCube(all_images, file_name, compression='gzip')
    logger.info('Wrote fft image to fits data cube %r',file_name)



if __name__ == "__main__":
    main(sys.argv)






