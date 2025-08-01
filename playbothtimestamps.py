import sys
import audioplayer
from multiprocessing import Process
from pathlib import Path
import os

if __name__ == "__main__":
    ''' Plays both timestamps produced by Agglomerative and Spatical clustering (hopefully both synced)
    Input: a path to a csv file in ANY timestamp directory
    Example: python playbothtimestamps.py agtimestamp/hiyis.csv
    '''
    assert(len(sys.argv)>1)
    ag='agtimestamp/'+Path(sys.argv[1]).stem+".csv"
    sp='sptimestamp/'+Path(sys.argv[1]).stem+".csv"
    assert(os.path.exists(ag))
    assert(os.path.exists(sp))
    t1 = Process(target=audioplayer.audioplayer,args=(ag,))
    t2 = Process(target=audioplayer.audioplayer,args=(sp,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print("Audio players closed")