import sys
import src.audioplayer as audioplayer
from multiprocessing import Process
from pathlib import Path
import os

def playboth(file):
    ''' Plays both timestamps produced by Agglomerative and Spatical clustering (hopefully both synced)
    Input: a path to a csv file in ANY timestamp directory
    Example: python playbothtimestamps.py agtimestamp/hiyis.csv
    '''
    AGTIMESTAMP_DIRECTORY = "data/agtimestamp/"
    SPTIMESTAMP_DIRECTORY = "data/sptimestamp/"
    ag=AGTIMESTAMP_DIRECTORY+Path(file).stem+".csv"
    sp=SPTIMESTAMP_DIRECTORY+Path(file).stem+".csv"
    assert(os.path.exists(ag))
    assert(os.path.exists(sp))
    t1 = Process(target=audioplayer.audioplayer,args=(ag,))
    t2 = Process(target=audioplayer.audioplayer,args=(sp,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print("Audio players closed")

if __name__ == "__main__":
    assert(len(sys.argv)>1)
    playboth(sys.argv[1])