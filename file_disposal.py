#!/usr/bin/env python

import os
import numpy as np
import scipy.io as sio
from matplotlib import pyplot
from ugaudio.load import pad_read

class DailyMinMaxFileDisposal(object):
    """Get min/max for each of X-, Y-, and Z-axis."""
    
    def __init__(self, filename, indir='/misc/yoda/pub/pad', outdir='/misc/yoda/www/plots/batch/results/dailyhistpad'):
        self.filename = filename
        self.outdir = outdir
    
    def run(self):
        # read data from file (not using double type here like MATLAB would, so we get courser demeaning)
        B = pad_read(self.filename)
        
        # demean each column
        A = B - B.mean(axis=0)
        
        #print '{0:s},{1:>.4e},{2:>.4e},{3:>.4e},{4:>.4e},{5:>.4e},{6:>.4e}'.format(filename,
        #                                   A.min(axis=0)[1], A.max(axis=0)[1], A.min(axis=0)[2],
        #                                   A.max(axis=0)[2], A.min(axis=0)[3], A.max(axis=0)[3] )

        # calculate histogram per column (x, y, z)
        xmin = np.min(A[:,1]); xmax = np.max(A[:,1])
        ymin = np.min(A[:,2]); ymax = np.max(A[:,2])
        zmin = np.min(A[:,3]); zmax = np.max(A[:,3])

        return xmin, xmax, ymin, ymax, zmin, zmax


class DailyOtoMinMaxFileDisposal(object):
    """Get OTO band min/max for each of X-, Y-, and Z-axis and for each frequency band."""
    
    def __init__(self, filename, indir='/misc/yoda/www/plots/batch/results/onethird', outdir='/misc/yoda/www/plots/batch/results/dailyhistoto'):
        self.filename = filename
        self.outdir = outdir
    
    def run(self):
        # read data from file (not using double type here like MATLAB would, so we get courser demeaning)
        B = pad_read(self.filename)
        
        # demean each column
        A = B - B.mean(axis=0)
        
        #print '{0:s},{1:>.4e},{2:>.4e},{3:>.4e},{4:>.4e},{5:>.4e},{6:>.4e}'.format(filename,
        #                                   A.min(axis=0)[1], A.max(axis=0)[1], A.min(axis=0)[2],
        #                                   A.max(axis=0)[2], A.min(axis=0)[3], A.max(axis=0)[3] )

        # calculate histogram per column (x, y, z)
        xmin = np.min(A[:,1]); xmax = np.max(A[:,1])
        ymin = np.min(A[:,2]); ymax = np.max(A[:,2])
        zmin = np.min(A[:,3]); zmax = np.max(A[:,3])

        return xmin, xmax, ymin, ymax, zmin, zmax


class DailyHistFileDisposal(object):
    """Use bins input to calculate and save histogram along outdir path."""
    
    def __init__(self, filename, bins, vecmag_bins, indir='/misc/yoda/pub/pad', outdir='/misc/yoda/www/plots/batch/results/dailyhistpad'):
        self.filename = filename
        self.bins = bins
        self.vecmag_bins = vecmag_bins
        self.outdir = outdir
        ###self.outfile = os.path.join( filename.replace(indir, outdir) ) + '.mat'
    
    def run(self):
        # read data from file (not using double type here like MATLAB would, so we get courser demeaning)
        B = pad_read(self.filename)
        
        # demean each column
        A = B - B.mean(axis=0)
        
        #print '{0:s},{1:>.4e},{2:>.4e},{3:>.4e},{4:>.4e},{5:>.4e},{6:>.4e}'.format(filename,
        #                                   A.min(axis=0)[1], A.max(axis=0)[1], A.min(axis=0)[2],
        #                                   A.max(axis=0)[2], A.min(axis=0)[3], A.max(axis=0)[3] )

        # calculate histogram per column (x, y, z)
        Nx, Bx = np.histogram(A[:,1], self.bins)
        Ny, By = np.histogram(A[:,2], self.bins)
        Nz, Bz = np.histogram(A[:,3], self.bins)
        
        # now for vecmag
        XYZ = A[:,1:]
        V = np.linalg.norm(XYZ, axis=1)
        Nv, Bv = np.histogram(V, self.vecmag_bins)

        return Nx, Ny, Nz, Nv


class DailyOtoHistFileDisposal(object):
    """Use bins input to calculate and save OTO histogram along outdir path."""
    
    def __init__(self, filename, bins, indir='/misc/yoda/www/plots/batch/results/onethird', outdir='/misc/yoda/www/plots/batch/results/dailyhistoto'):
        self.filename = filename
        self.bins = bins
        self.outdir = outdir
    
    def run(self):
        # read data from file (not using double type here like MATLAB would, so we get courser demeaning)
        B = pad_read(self.filename)
        
        # demean each column
        A = B - B.mean(axis=0)
        
        #print '{0:s},{1:>.4e},{2:>.4e},{3:>.4e},{4:>.4e},{5:>.4e},{6:>.4e}'.format(filename,
        #                                   A.min(axis=0)[1], A.max(axis=0)[1], A.min(axis=0)[2],
        #                                   A.max(axis=0)[2], A.min(axis=0)[3], A.max(axis=0)[3] )

        # calculate histogram per column (x, y, z)
        Nx, Bx = np.histogram(A[:,1], self.bins)
        Ny, By = np.histogram(A[:,2], self.bins)
        Nz, Bz = np.histogram(A[:,3], self.bins)
        
        # now for vecmag
        XYZ = A[:,1:]
        V = np.linalg.norm(XYZ, axis=1)
        Nv, Bv = np.histogram(V, self.vecmag_bins)

        return Nx, Ny, Nz, Nv


def demo():
    
    DEFAULT_HISTFOLDER = '/misc/yoda/www/plots/batch/results/dailyhistpad'
    
    # load bins and vecmag_bins
    a = sio.loadmat(os.path.join(DEFAULT_HISTFOLDER, 'dailyhistpad_bins.mat'))
    vecmag_bins = a['vecmag_bins'][0]
    bins = a['bins'][0]
    
    filename = '/misc/yoda/pub/pad/year2017/month12/day25/sams2_accel_121f05/2017_12_25_00_07_12.234+2017_12_25_00_17_12.249.121f05'
    #bins = np.arange(-0.2, 0.2, 5e-5)
    #vecmag_bins = np.arange(0, 0.5, 5e-5)
    
    dh = DailyHistFileDisposal(filename, bins, vecmag_bins)
    Nx, Ny, Nz, Nv = dh.run()


if __name__ == "__main__":
    demo()
