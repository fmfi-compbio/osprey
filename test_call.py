import osprey24dwx as osprey
import numpy as np
import time
from ont_fast5_api.fast5_interface import get_fast5_file
import sys
import os
import deepnano2
from scipy.special import softmax

def med_mad(x, factor=1.4826):
    """
    Calculate signal median and median absolute deviation
    """
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad

def rescale_signal(signal):
    signal = signal.astype(np.float32)
    med, mad = med_mad(signal)
    signal -= med
    signal /= mad
    return np.clip(signal, -2.5, 2.5)

alph = "NACGT"

def call_file(filename):
    out = []
#    try:
    if True:
        with get_fast5_file(filename, mode="r") as f5:
            for read in f5.get_reads():
                read_id = read.get_read_id()
                signal = read.get_raw_data()
                signal = rescale_signal(signal)

#                signal = signal[:2*512*3*4]
#                print(signal)
                basecall = caller.call(signal).reshape((-1,5))
                basecall = softmax(basecall, axis=1)
#                print(basecall)
#                print(basecall.shape)
                basecall = deepnano2.beam_search_py(np.ascontiguousarray(basecall), 5, 0.1)
#               print(len(basecall))
                out.append((read_id, basecall, len(signal)))
#    except OSError:
#        return []
    return out

caller = osprey.CallerDWX()

base_dir = sys.argv[1]
files = [os.path.join(base_dir, fn) for fn in os.listdir(base_dir)]

#files = ["../eval/test_data_476/5210_N125509_20170425_FN2002039725_MN19691_sequencing_run_klebs_033_restart_87298_ch146_read12031_strand.fast5"]

fout = open("outtmp3d.fasta", "w")

ts = 0
start = time.time()
for i, f in enumerate(files):
    res = call_file(f)
    for r_id, basecall, ls in res:
        print(">%s" % r_id, file=fout)
        print(basecall, file=fout)
        print(i, f, ls, len(basecall))
        fout.flush()
        ts += ls

end = time.time()
print("speed", ts / (end - start))

fout.close()
