import sys
from os import path
import os

from gwpy.segments import DataQualityFlag
from gwpy.time import tconvert

def get_segments(ifo, start=1238112018, stop=-1):
    """
    get_segments(ifo, start=1238112018, stop=-1)

    Query the segment database to find ANALYSIS_READY == 1 segments.

    Inputs:
        ifo   = interferometer ['L' or 'H']
        start = start time (default: beginning of O3)
        stop  = stop time (default: now)
    Outputs:
        segments: a list of segments [seg id, start GPS, end GPS]  
    """

    # get the current GPS time if needed
    if stop == -1:
        stop = tconvert('now').gpsSeconds
    # quesry segment database
    segs = DataQualityFlag.query(ifo + '1:DMT-ANALYSIS_READY:1',
                                 start, stop)
    # convert to numerical (nteger) values
    return [[i, int(s.start), int(s.end)] for i,s in enumerate(segs.active)]


# list all segments
segments = get_segments('H')

# parse arguments
if len(sys.argv) < 2:
    print('Syntax: create_dag_file.py dag_file_name [min_gps]')
    sys.exit(0)

dag_file = sys.argv[1]   # name of the DAG file
if len(sys.argv) > 2:
    # the user specified a minimum GPS time, remove all previous segments
    min_gps = int(sys.argv[2])
    segments = [s for s in segments if s[1] >= min_gps]

# folder where result frames will be saved
out_folder = '/hdfs/user/mcoughlin/DeepClean/'
# name of the submission file
sub_file = '/home/mcoughlin/DeepClean/condor/DeepClean.sub'

# GPS times: if a list is passed, only the listed GPS times are used
#gps = [1244813472, 1244669638, 1244593946, 1243627776, 1242593626, 1242437036, 1242119904, 1241366461, 1240919370]
gps = []


with open(dag_file, 'w') as fid:
    for i,s in enumerate(segments):
        if len(gps) == 0 or (s[1] in gps):
            fid.write('JOB %d %s\n' % (i, sub_file))
            fid.write('RETRY %d 1\n' % i)
            fid.write('VARS %d macrojobnumber="%d" gps_start="%d" gps_stop="%d" out_folder="%s"\n' % \
                           (i, i, s[1], s[2], out_folder)) 
