#!/usr/bin/env python

import os
import re
import sys
import struct
import datetime
import numpy as np
import pandas as pd
from dateutil import parser
from pims.files.utils import mkdir_p
from pims.files.filter_pipeline import FileFilterPipeline, FileExists
from pims.patterns.dailyproducts import _PADPATH_PATTERN # yoda is default base path
from pims.patterns.dailyproducts import _PADHEADERFILES_PATTERN, _PADDATAFILES_PATTERN # yoda is default base path
from pims.patterns.dailyproducts import _HISTMATFILES_PATTERN
from histpad.file_disposal import DailyHistFileDisposal, DailyMinMaxFileDisposal


# FIXME this has assumptions for PAD (number of columns)
def get_duration_minutes(datfile):
    """Calculate data file duration in minutes based on last time step."""
    with open(datfile, 'rb') as f:
        # FIXME THIS IS ONLY VALID WITH 4-COLUMN PAD FILES.
        # Note that *most* PAD files use relative time in seconds with t1 =
        # 0 and next time starting at byte 16, so seek to proper position
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        
        # rewind back one record
        f.seek(pos-16, os.SEEK_SET)
    
        # Now we want just one of these 4-byte floats (float32)
        b = f.read(4)
    
    # Decode time step (delta t) as little-endian float32.
    last_timestep = struct.unpack('<f', b)[0]
    
    # Return calculated duration in minutes
    return round(last_timestep/60.0, 3)


# A callable class for PAD header file
class PadHeader(object):
    """A callable class for PAD header file."""
    
    def __init__(self, regex=_PADHEADERFILES_PATTERN):
        self.regex = regex
        self.reobj = re.compile(regex)
        
    def __call__(self, file_list):
        for f in file_list:
            #print 'file', f
            if self.reobj.match(f):
                yield f

# A callable class for PAD data file
class PadData(object):
    """A callable class for PAD data file."""
    
    def __init__(self, regex=_PADDATAFILES_PATTERN):
        self.regex = regex
        self.reobj = re.compile(regex)
        
    def __call__(self, file_list):
        for f in file_list:
            #print 'file', f
            if self.reobj.match(f):
                yield f
                
    def __str__(self):
        return 'is a PAD file'

# A callable class for PAD histogram mat file for particular day-sensor
class PadHistMat(object):
    """A callable class for PAD histogram mat file for particular day-sensor."""
    
    def __init__(self, regex=_HISTMATFILES_PATTERN):
        self.regex = regex
        self.rxobj = re.compile(regex)
        
    def __call__(self, file_list):
        for f in file_list:
            #print 'file', f
            if self.rxobj.match(f):
                yield f
                
    def __str__(self):
        return 'is a PAD hist mat file'
        
# A callable class for PAD data file for particular day-sensor
class PadDataDaySensor(PadData):
    """A callable class for PAD data file for particular day-sensor."""
    
    def __init__(self, daystr, sensor):
        self.date = parser.parse(daystr).date()
        self.daystr = self.date.strftime('%Y_%m_%d')
        self.sensor = sensor
        regex = self._get_regex_pat()
        super(PadDataDaySensor, self).__init__(regex=regex)

    def _get_regex_pat(self):
        pattern = _PADPATH_PATTERN + "/(?P<subdir>.*_(accel|rad)_(?P<sensor>%s))/" % self.sensor + \
            "(?P<start>%s_\d{2}_\d{2}_\d{2}\.\d{3})" % self.daystr + \
            "(?P<pm>[\+\-])" + \
            "(?P<stop>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.\d{3})" + \
            "\.(?P=sensor)\Z"
        return pattern
                
    def __str__(self):
        return 'is a PAD file on %s for %s' % (self.date, self.sensor)

# A callable class for PAD data file for particular day-sensor that has where clause for header fields
class PadDataDaySensorWhere(PadDataDaySensor):
    """A callable class for PAD data file for particular day-sensor that has where clause for header fields."""

    def __init__(self, daystr, sensor, where={'CutoffFreq': 200}):
        super(PadDataDaySensorWhere, self).__init__(daystr, sensor)
        self.where = where
    
    def __call__(self, file_list):
        for f in file_list:
            #print f
            if self.reobj.match(f):
                hdrfile = self._get_headerfile(f)
                if self._header_satifies_where(hdrfile):
                    yield f
                
    def __str__(self):
        return 'is a PAD file on %s for %s where %s' % (self.date, self.sensor, self.where)

    def _header_satifies_where(self, hdrfile):
        # FIXME this is QUICK TURN AROUND short-circuiting more potential the "wherefulness"
        fc = self._get_cutoff(hdrfile)
        if fc == self.where['CutoffFreq']:
            return True
        else:
            return False
        
    def _get_headerfile(self, filename):
        """Return header filename if it exists; otherwise, None."""
        hdrfile = filename + '.header'
        if os.path.exists(hdrfile):
            return hdrfile
        else:
            return None

    def _get_cutoff(self, hdrfile):
        """Return cutoff from header file; otherwise None."""
        with open(hdrfile, 'r') as f:
            contents = f.read().replace('\n', '')
            m = re.match('.*\<CutoffFreq\>(.*)\</CutoffFreq\>.*', contents)
            if m:
                return float( m.group(1) )
            else:
                return None


class PadDataDaySensorWhereMinDur(PadDataDaySensorWhere):
    """A callable class for PAD data file for particular day-sensor that has where clause for header fields and some minimum duration too."""

    def __init__(self, daystr, sensor, where={'SampleRate': 500.0}, mindur=5):
        super(PadDataDaySensorWhereMinDur, self).__init__(daystr, sensor, where=where)
        self.mindur = mindur
    
    def __call__(self, file_list):
        for f in file_list:
            #print f
            if self.reobj.match(f):
                hdrfile = self._get_headerfile(f)
                if self._header_satifies_where(hdrfile):
                    if get_duration_minutes(f) >= self.mindur:
                        yield f
                
    def __str__(self):
        return 'is a PAD file on %s for %s where %s and duration >= %d minutes' % (self.date, self.sensor, self.where, self.mindur)

    def _header_satifies_where(self, hdrfile):
        # FIXME this is QUICK TURN AROUND short-circuiting more potential the "wherefulness"
        fs = self._get_samplerate(hdrfile)
        if fs == self.where['SampleRate']:
            return True
        else:
            return False

    def _get_samplerate(self, hdrfile):
        """Return samplerate from header file; otherwise None."""
        with open(hdrfile, 'r') as f:
            contents = f.read().replace('\n', '')
            m = re.match('.*\<SampleRate\>(.*)\</SampleRate\>.*', contents)
            if m:
                return float( m.group(1) )
            else:
                return None


def get_pad_day_sensor_files(files, day, sensor):

    # Initialize callable classes that act as filters for our pipeline
    fe = FileExists()
    pdds = PadDataDaySensor(day, sensor)
    
    # Initialize processing pipeline with callable classes, but not using file list as input yet
    ffp = FileFilterPipeline(fe, pdds)
    #print ffp

    # Now apply processing pipeline to file list; at this point, ffp is callable
    return list( ffp(files) )


def get_pad_day_sensor_files_mindur(files, day, sensor):

    # Initialize callable classes that act as filters for our pipeline
    pddsmd = PadDataDaySensorWhereMinDur(day, sensor)
    
    # Initialize processing pipeline with callable classes, but not using file list as input yet
    ffp = FileFilterPipeline(pddsmd)
    #print ffp

    # Now apply processing pipeline to file list; at this point, ffp is callable
    return list( ffp(files) )


def demo_one(sensor='121f03', file_getter=get_pad_day_sensor_files):
    
    import datetime
    from pims.utils.datetime_ranger import next_day
    from pims.utils.pimsdateutil import datetime_to_ymd_path
    
    start = datetime.date(2017, 1, 1)
    stop = datetime.date(2017, 1, 3)
    nd = next_day(start)
    d = start
    while d < stop:
        d = nd.next()
        day = d.strftime('%Y-%m-%d')

        # Get list of PAD data files for particular day and sensor
        pth = os.path.join( datetime_to_ymd_path(d), 'sams2_accel_' + sensor )
        if os.path.exists(pth):
            tmp = os.listdir(pth)
            files = [ os.path.join(pth, f) for f in tmp ]    
                    
            # Run routine to filter files
            #my_files = get_pad_day_sensor_files(files, day, sensor)
            my_files = file_getter(files, day, sensor)
            print '%s gives %d files' % (day, len(my_files))
        else:
            print '%s gives NO FILES' % day


def demo_padrunhist(start, stop, sensor='121f03', file_getter=get_pad_day_sensor_files_mindur):
    
    import datetime
    from pims.utils.datetime_ranger import next_day
    from pims.utils.pimsdateutil import datetime_to_ymd_path
    
    nd = next_day(start)
    d = start
    while d < stop:
        d = nd.next()
        day = d.strftime('%Y-%m-%d')

        # Get list of PAD data files for particular day and sensor
        pth = os.path.join( datetime_to_ymd_path(d), 'sams2_accel_' + sensor )
        if os.path.exists(pth):
            tmp = os.listdir(pth)
            files = [ os.path.join(pth, f) for f in tmp ]    
                    
            # Run routine to filter files
            my_files = file_getter(files, day, sensor)
            print '%s gives %d files' % (day, len(my_files))
        else:
            print '%s gives NO FILES' % day
            
            
def get_pad_day_sensor_where_files(files, day, sensor, where, mindur=5):

    # Initialize callable classes that act as filters for our pipeline
    fe = FileExists()
    #pddsw = PadDataDaySensorWhere(day, sensor, where)
    pddsw = PadDataDaySensorWhereMinDur(day, sensor, where, mindur)
    
    # Initialize processing pipeline with callable classes, but not using file list as input yet
    ffp = FileFilterPipeline(fe, pddsw)
    #print ffp

    # Now apply processing pipeline to file list; at this point, ffp is callable
    return list( ffp(files) )

def sensor2subdir(sensor):
    if sensor.startswith('hirap'):
        return 'mams_accel_' + sensor
    elif sensor.startswith('es0'):
        return 'samses_accel_' + sensor
    elif sensor.startswith('0bb'):
        return 'mma_accel_' + sensor
    else:
        return 'sams2_accel_' + sensor

def do_dailyhistpad(start, stop, sensor='121f03', where={'CutoffFreq': 200}, bins=np.arange(-0.2, 0.2, 5e-5), vecmag_bins=np.arange(0, 0.5, 5e-5), mindur=5):
    
    import datetime
    import scipy.io as sio
    from pims.utils.datetime_ranger import next_day
    from pims.utils.pimsdateutil import datetime_to_ymd_path
    
    indir='/misc/yoda/pub/pad'
    outdir='/misc/yoda/www/plots/batch/results/dailyhistpad'
    
    #sio.savemat(os.path.join(outdir, 'dailyhistpad_bins.mat'), {'bins': bins[:-1], 'vecmag_bins': vecmag_bins[:-1]}); raise SystemExit
    
    nd = next_day(start)
    d = start
    d = nd.next()
    while d <= stop:
        day = d.strftime('%Y-%m-%d')

        # Get list of PAD data files for particular day and sensor
        pth = os.path.join( datetime_to_ymd_path(d), sensor2subdir(sensor) )
        print pth
        if os.path.exists(pth):
            tmp = os.listdir(pth)
            files = [ os.path.join(pth, f) for f in tmp ]

            # Run routine to filter files
            my_files = get_pad_day_sensor_where_files(files, day, sensor, where, mindur)
            print '%s gives %d files' % (day, len(my_files))
            
            len_files = len(my_files)
            if len_files > 0:
                outfile = os.path.join( pth.replace(indir, outdir), 'dailyhistpad.mat')
                if os.path.exists(outfile):
                    Exception('OUTPUT FILE %s ALREADY EXISTS' % outfile)
                else:
                    directory = os.path.dirname(outfile)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                dh = DailyHistFileDisposal(my_files[0], bins, vecmag_bins)
                Nx, Ny, Nz, Nv = dh.run()
                print '>> completed %s' % my_files[0]
                for f in my_files[1:]:
                    dh = DailyHistFileDisposal(f, bins, vecmag_bins)
                    nx, ny, nz, nv = dh.run()
                    Nx += nx
                    Ny += ny
                    Nz += nz
                    Nv += nv
                    print '>> completed %s' % f
                sio.savemat(outfile, {'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 'Nv': Nv})
                print
        else:
            print '%s gives NO FILES' % day
        d = nd.next()

def demo_dailyminmaxpad(sensor='121f04', where={'CutoffFreq': 200}):
    
    import datetime
    import scipy.io as sio
    from pims.utils.datetime_ranger import next_day
    from pims.utils.pimsdateutil import datetime_to_ymd_path
    
    indir='/misc/yoda/pub/pad'
    outdir='/misc/yoda/www/plots/batch/results/dailyminmaxpad'
    
    start = datetime.date(2016, 4, 2)
    stop = datetime.date(2016, 6, 2)
    nd = next_day(start)
    d = start
    d = nd.next()
    while d <= stop:
        day = d.strftime('%Y-%m-%d')

        # Get list of PAD data files for particular day and sensor
        pth = os.path.join( datetime_to_ymd_path(d), 'sams2_accel_' + sensor )
        if os.path.exists(pth):
            tmp = os.listdir(pth)
            files = [ os.path.join(pth, f) for f in tmp ]

            # Run routine to filter files
            my_files = get_pad_day_sensor_where_files(files, day, sensor, where)
            print '%s gives %d files' % (day, len(my_files))

            len_files = len(my_files)
            if len_files > 0:
                outfile = os.path.join( pth.replace(indir, outdir), 'dailyminmaxpad.mat')
                if os.path.exists(outfile):
                    Exception('OUTPUT FILE %s ALREADY EXISTS' % outfile)
                else:
                    directory = os.path.dirname(outfile)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                xmin, xmax = np.inf, -np.inf
                ymin, ymax = np.inf, -np.inf
                zmin, zmax = np.inf, -np.inf
                for f in my_files:
                    dmm = DailyMinMaxFileDisposal(f)
                    x1, x2, y1, y2, z1, z2 = dmm.run()
                    xmin = x1 if x1 < xmin else xmin
                    xmax = x2 if x2 > xmax else xmax
                    ymin = y1 if y1 < ymin else ymin
                    ymax = y2 if y2 > ymax else ymax
                    zmin = z1 if z1 < zmin else zmin
                    zmax = z2 if z2 > zmax else zmax
                    print '>> completed %s' % f
                sio.savemat(outfile, {'xmin': xmin, 'xmax': xmax,
                                      'ymin': ymin, 'ymax': ymax,
                                      'zmin': zmin, 'zmax': zmax})
                print
        else:
            print '%s gives NO FILES' % day
        d = nd.next()


def demo_three(sensor='es03', where={'CutoffFreq': 101.4}):
    
    import datetime
    from pims.utils.datetime_ranger import next_day
    from pims.utils.pimsdateutil import datetime_to_ymd_path
    
    start = datetime.date(2016, 3, 15)
    stop = datetime.date(2016, 3, 19)
    nd = next_day(start)
    d = start
    while d < stop:
        d = nd.next()
        day = d.strftime('%Y-%m-%d')

        # Get list of PAD data files for particular day and sensor
        pth = os.path.join( datetime_to_ymd_path(d), 'samses_accel_' + sensor )
        if os.path.exists(pth):
            tmp = os.listdir(pth)
            files = [ os.path.join(pth, f) for f in tmp ]

            # Run routine to filter files
            my_files = get_pad_day_sensor_where_files(files, day, sensor, where)
            print '%s gives %d files' % (day, len(my_files))
        else:
            print '%s gives NO FILES' % day

def parse_day(f):
    result = None
    regex = re.compile(_HISTMATFILES_PATTERN)
    m = regex.match(f)
    if m:
        y, m, d = m.group('year'), m.group('month'), m.group('day')
        #result = datetime.date(int(y), int(m), int(d))
        result = pd.Timestamp('%s-%s-%s' %(y, m, d), tz=None)
    return result
    
def compare_histmat_files(ymdpat):
    import glob
    import matplotlib.pylab as plt
    import scipy.io as sio
    
    # load bins and vecmag_bins
    a = sio.loadmat('/misc/yoda/www/plots/batch/results/dailyhistpad/dailyhistpad_bins.mat')
    vecmag_bins = a['vecmag_bins'][0]
    bins = a['bins'][0]
    
    # glob all files that match ymdpat and sort them
    glob_pat = '/misc/yoda/www/plots/batch/results/dailyhistpad/%s/sams2_accel_121f0?/dailyhistpad.mat'
    files = glob.glob(glob_pat % ymdpat)
    files.sort()
    #print len(files), 'files match', glob_pat
            
    # use regex matching to get list of unique sensors
    regex = re.compile(_HISTMATFILES_PATTERN)
    sensor_list = [regex.match(s).group('sensor') for s in files]
    uniq_sensors = sorted(set(sensor_list))

    d = []
    for sensor in uniq_sensors:
        #hFig = plt.figure()
        sensor_files = [f for f in files if sensor in f]
        #print sensor, 'has', len(sensor_files), 'files'
        #n = len(sensor_files)
        #colors = plt.cm.jet(np.linspace(0,1,n))
        for i, f in enumerate(sensor_files):
            day = parse_day(f)
            data = sio.loadmat(f)
            Nv = data['Nv'][0]
            #plt.plot(vecmag_bins, Nv, color=colors[i])
            yd = 100 * np.cumsum(Nv) / sum(Nv)
            x = np.interp(50, yd, vecmag_bins/1e-3)  # we swap x and y here for interpolating the way we want it
            d.append({'sensor': sensor, 'day': day, 'median': x})
            #print '%6.3f mg median for %s' % (x, f)
        
    df = pd.DataFrame(d)
    df = df.set_index('day')
    bp = df.boxplot(by='sensor')
    plt.ylim(ymin=0, ymax=2.5)
    plt.show()

def demo_compare_hist_boxplot():
    ymdpat = 'year2017/month0?/day0?'
    compare_histmat_files(ymdpat)

if __name__ == "__main__":
    #demo_one()
    #demo_dailyminmaxpad()
    demo_compare_hist_boxplot()
    raise SystemExit
    
    start = datetime.date(2017, 7, 1)
    stop = datetime.date(2017, 8, 1)
    do_dailyhistpad(start, stop, sensor='121f03')
    
    #demo_three()
    