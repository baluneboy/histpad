#!/usr/bin/env python

import os
import sys
import datetime
import numpy as np
from dateutil import parser
from histpad.pad_filter_pipeline import do_dailyhistpad, PadDataDaySensorWhereMinDur

# some useful relative dates
TODAY = datetime.datetime.today().date()
TWODAYSAGO = TODAY - datetime.timedelta(days=2)

# defaults
defaults = {
'sensorparams': [
    #( '121f03',   {'CutoffFreq': 200} ),
    ( '121f03',   {'SampleRate': 500.0} ),
    #( 'hirap',    {'CutoffFreq': 100} ),    
    ],          
'start':   TWODAYSAGO.strftime('%Y-%m-%d'),
'stop':    TWODAYSAGO.strftime('%Y-%m-%d'),
'dryrun':  'False',  # False for actual, full processing; True to not actually do much
}
parameters = defaults.copy()

# check for reasonableness of parameters
def parameters_ok():
    """check for reasonableness of parameters"""    

    # convert start & stop parameters to date objects
    parameters['start'] = parser.parse( parameters['start'] ).date()
    parameters['stop'] = parser.parse( parameters['stop'] ).date()
    if parameters['stop'] < parameters['start']:
        print 'stop is less than start'
        return False
    
    # boolean dryrun
    try:
        parameters['dryrun'] = eval(parameters['dryrun'])
        assert( isinstance(parameters['dryrun'], bool))
    except Exception, err:
        print 'cound not handle dryrun parameter, was expecting it to eval to True or False'
        return False    
    
    return True # all OK; otherwise, return False somewhere above here

# print helpful text how to run the program
def print_usage():
    """print helpful text how to run the program"""
    print 'usage: %s [options]' % os.path.abspath(__file__)
    print '       options (and default values) are:'
    for i in defaults.keys():
        print '\t%s=%s' % (i, defaults[i])

# convenient test routine
def test_dailyhistpad(start, stop, sensor='121f03', where={'CutoffFreq': 200}, bins=np.arange(-0.2, 0.2, 5e-5), vecmag_bins=np.arange(0, 0.5, 5e-5), mindur=5):
    """convenient test routine"""
    
    import datetime
    import scipy.io as sio
    from pims.utils.datetime_ranger import next_day
    from pims.utils.pimsdateutil import datetime_to_ymd_path
    from histpad.pad_filter_pipeline import get_pad_day_sensor_where_files, sensor2subdir
    
    indir='/misc/yoda/pub/pad'
    outdir='/misc/yoda/www/plots/batch/results/dailyhistpad'
        
    nd = next_day(start)
    d = start
    d = nd.next()
    while d <= stop:
        day = d.strftime('%Y-%m-%d')

        # Get list of PAD data files for particular day and sensor
        pth = os.path.join( datetime_to_ymd_path(d), sensor2subdir(sensor) )
        if os.path.exists(pth):
            tmp = os.listdir(pth)
            files = [ os.path.join(pth, f) for f in tmp ]

            # Run routine to filter files
            my_files = get_pad_day_sensor_where_files(files, day, sensor, where, mindur)
            print '%s gives %d files' % (day, len(my_files))

        else:
            print '%s gives NO FILES' % day
        d = nd.next()

# generate dailyhistpad histogram mat files
def process_data():
    """generate dailyhistpad histogram mat files"""

    start = parameters['start']
    stop  = parameters['stop']
    dryrun = parameters['dryrun']
    for tup in parameters['sensorparams']:
        sensor, where = tup[0], tup[1]
        print sensor, where, '\t',
        if dryrun:
            test_dailyhistpad(start, stop, sensor=sensor, where=where, mindur=5)
        else:
            do_dailyhistpad(start, stop, sensor=sensor, where=where, mindur=5)

# check parameters and run main process
def main(argv):
    """describe main routine here"""
    
    # parse command line
    for p in sys.argv[1:]:
        pair = p.split('=')
        if (2 != len(pair)):
            print 'bad parameter: %s' % p
            break
        else:
            parameters[pair[0]] = pair[1]
    else:
        if parameters_ok():
            process_data()
            return 0
        
    print_usage()  

# run main with cmd line args and return exit code
if __name__ == '__main__':
    """run main with cmd line args and return exit code"""
    sys.exit(main(sys.argv))