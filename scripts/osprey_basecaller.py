#!/usr/bin/env python

import osprey
import numpy as np
import time
from ont_fast5_api.fast5_interface import get_fast5_file
import sys
import os
from scipy.special import softmax
import decoder.decoder as d
import pickle
from multiprocessing import Pool
import gzip
import datetime
import argparse

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

                print(len(signal))
                if len(signal) >= 512*3*4:
                    basecall = caller.call(signal)
                    if len(basecall) > 0:
                        basecall = basecall.reshape((-1,48))
                        basecall, quals = decoder.beam_search(np.ascontiguousarray(basecall), args.beam_size, args.beam_cut_threshold)
                    else:
                        basecall = "A"
                        quals = "A"
                else:
                    signal = np.pad(signal, (0, 512*3*4-len(signal)))
                    basecall = caller.call(signal).reshape((-1,48))
                    basecall, quals = decoder.beam_search(np.ascontiguousarray(basecall), args.beam_size, args.beam_cut_threshold)


                out.append((read_id, basecall, quals))
    return out


def write_output(read_id, basecall, quals, output_file, format):
    if len(basecall) == 0:
        return
    if format == "fasta":
        print(">%s" % read_id, file=fout)
        print(basecall, file=fout)
    else: # fastq
        print("@%s" % read_id, file=fout)
        print(basecall, file=fout)
        print("+", file=fout)
        print(quals, file=fout)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast caller for ONT reads')

    parser.add_argument('--directory', type=str, nargs='*', help='One or more directories with reads')
    parser.add_argument('--reads', type=str, nargs='*', help='One or more read files')
    parser.add_argument("--output", type=str, required=True, help="Output FASTA file name")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for basecalling, default 1")
    parser.add_argument("--weights", type=str, default=None, help="Path to network weights, only used for custom weights")
    parser.add_argument("--beam-size", type=int, default=5,
        help="Beam size (default 5)")
    parser.add_argument("--beam-cut-threshold", type=float, default=0.1,
        help="Threshold for creating beams (higher means faster beam search, but smaller accuracy). Values higher than 0.2 might lead to weird errors. Default 0.1 for 48,...,96 and 0.0001 for 256")
    parser.add_argument("--output-format", choices=["fasta", "fastq"], default="fasta")
    parser.add_argument("--gzip-output", action="store_true", help="Compress output with gzip")

    args = parser.parse_args()

    if args.weights is None:
        weights = os.path.join(osprey.__path__[0], "weights", "net24dp.txt")
    else:
        weights = args.weights
    
    caller = osprey.Caller(weights)
    small_tables = pickle.load(open("%s.tabs" % weights, "rb"))
    decoder = d.DecoderTab(small_tables[0],
                           small_tables[1],
                           small_tables[2],
                           small_tables[3],
                           small_tables[4],
                           small_tables[5],
                           small_tables[6], 
    )


    assert args.threads >= 1

    files = args.reads if args.reads else []
    if args.directory:
        for directory_name in args.directory:
            files += [os.path.join(directory_name, fn) for fn in os.listdir(directory_name)]

    if len(files) == 0:
        print("Zero input reads, nothing to do.")
        sys.exit()


    if args.gzip_output:
        fout = gzip.open(args.output, "wt")
    else:
        fout = open(args.output, "w")

    if args.threads <= 1:
        done = 0
        for fn in files:
            start = datetime.datetime.now()
            for read_id, basecall, qual in call_file(fn):
                write_output(read_id, basecall, qual, fout, args.output_format) 
                done += 1
                print("done %d/%d" % (done, len(files)), read_id, datetime.datetime.now() - start, file=sys.stderr)

    else:
        pool = Pool(args.threads)
        done = 0
        for out in pool.imap_unordered(call_file, files):
            for read_id, basecall, qual in out:
                write_output(read_id, basecall, qual, fout, args.output_format)
                done += 1
                print("done %d/%d" % (done, len(files)), read_id, file=sys.stderr)
    
    fout.close()

