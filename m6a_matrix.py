import pandas as pd
import pysam
import numpy as np
from tqdm.notebook import tqdm
from pathlib import Path
from collections import defaultdict
import itertools
import numba
import re

# ——————————————————————————————————————————————————————————————————————————————————————— #

"""
Folder containing .bam and .bai files. 
Must be denoted by the same name with ".reseq" for technological replicates.
Must be denoted by the same name with "rep" for biological replicates.
"""

folder = Path("~/directory-path")

groups_dict = defaultdict(list)

for subdir in folder.iterdir():
    if not subdir.is_dir():
        continue

    name = subdir.name
    name = subdir.name
    # for technological replicates
    key = name.replace(".reseq", "")
    # for biological replicates
    # key = re.sub(r"_rep.*", "", name)

    bam_path = subdir / f"{name}.sorted.nucl.bam"

    # align the file and store in list
    if bam_path.exists():
        bam = pysam.AlignmentFile(str(bam_path))
        groups_dict[key].append(bam)

# ——————————————————————————————————————————————————————————————————————————————————————— #

"""
Call the directory containing bed files with TF motif promoters information.
The "bound" column must specify "yes" or "no".
"""

base_path = "~/directory-path"

tf_names = ["banp", "gabpa", "zfp143"]

cols = ["chrom", "start", "end", "name", "strand", "bound"]

peaks_dict = {}

for tf in tf_names:
    file_path = f"{base_path}/{tf}_motif_promoters.bed"
    
    df = pd.read_csv(file_path, sep="\t", header=None, names=cols)

    yes = df[df["bound"].eq("yes")].reset_index(drop=True)
    no  = df[df["bound"].eq("no")].reset_index(drop=True)

    peaks_dict[f"{tf}_peaks"] = [yes, no]

peaks_dict

# ——————————————————————————————————————————————————————————————————————————————————————— #

@numba.njit
def _check_cigar_pos(pos, cigar):
    if pos > cigar.shape[0]:
        raise ValueError("Invalid CIGAR string (1)")

@numba.njit
def _tokenize_cigar_string( cigar ):

    # constants
    ord_0 = 48  # ord(b'0')
    ord_9 = 57  # ord(b'9')
    
    # get length
    num_ops = 0
    for i in range(len(cigar)):
        if cigar[i] < ord_0 or cigar[i] > ord_9:
            # not a digit, thus an operation
            num_ops += 1

    operations = np.zeros(num_ops, np.uint8)
    step_sizes = np.zeros(num_ops, np.int32)

    cigar_pos = 0
    for i in range(num_ops):
    
        # Decode next CIGAR operation. First find number.
        step = 0
        while cigar[cigar_pos] >= ord_0 and cigar[cigar_pos] <= ord_9:
            step = step*10 + cigar[cigar_pos] - ord_0
            cigar_pos += 1
            _check_cigar_pos(cigar_pos, cigar)

        step_sizes[i] = step

        # Now get CIGAR operation letter
        if cigar_pos >= cigar.shape[0]:
            raise ValueError("Invalid CIGAR string (2)")
            
        operations[i] = cigar[cigar_pos]
        cigar_pos += 1
        
        if i < num_ops:
            _check_cigar_pos(cigar_pos, cigar)

    return operations, step_sizes


def tokenize_cigar_string( cigarstring ):
    """Given a CIGAR string, this function returns a pair of arrays:
    - an np.uint8 array with the CIGAR operation letters
       (as ASCII codes of one the letters M, I, D, N, S, H, P, =, X)
    - an np.int32 vector with the sizes of the operations
    """
    cigar = np.frombuffer( bytes(cigarstring, 'ascii'), np.uint8 )
    return _tokenize_cigar_string( cigar )


_valid_cigar_ops = np.frombuffer( b'MIDNSHP=X', np.uint8 )
_query_consuming_cigar_ops = np.frombuffer( b'MIS=X', np.uint8 )
_reference_consuming_cigar_ops = np.frombuffer( b'MDN=X', np.uint8 )

# ——————————————————————————————————————————————————————————————————————————————————————— #

@numba.njit
def _get_q2r_positions( cigar_ops, cigar_step_sizes, seq_len, ref_start ):
    q2rpos = np.empty( seq_len, dtype=np.int64 )
    query_pos = 0
    ref_pos = ref_start
    cigar_pos = 0
    for i in range(cigar_ops.shape[0]):
        
        # Check whether letter is valied
        if cigar_ops[i] not in _valid_cigar_ops: #b'MIDNSHP=X':
            raise ValueError("Invalid CIGAR string (unknown operation)")
    
        # Check whether operation consumes query and/or reference
        query_consuming = cigar_ops[i] in _query_consuming_cigar_ops
        ref_consuming = cigar_ops[i] in _reference_consuming_cigar_ops
            
        if query_consuming:
            step_end = query_pos + cigar_step_sizes[i]
            while query_pos < step_end:
                if ref_consuming:
                    q2rpos[query_pos] = ref_pos
                    ref_pos += 1
                else:
                    q2rpos[query_pos] = -1
                query_pos += 1
                    
        else:
            if ref_consuming:
                ref_pos += cigar_step_sizes[i]
    
    if query_pos != seq_len:
        raise ValueError("Invalid CIGAR string (Query sequence not exactly consumed)")

    return q2rpos


def get_q2r_positions( cigarstring, query_seq_len, ref_start=0 ):
    """ Given a CIGAR string, find for each base in the query sequence (i.e., the read)
    the position on the reference sequence (i.e., the chromosome) with which the 
    alignment pairs it. 
    The function needs as arguments the CIGAR string, the length of the read and 
    the reference start position of the alignment.
    """
    cigar_ops, cigar_step_sizes = tokenize_cigar_string( cigarstring )
    return _get_q2r_positions( cigar_ops, cigar_step_sizes, query_seq_len, ref_start )

# ——————————————————————————————————————————————————————————————————————————————————————— #

def parse_MM_tag(read):
    d = {}
    for modstr in read.get_tag("MM").split(";"):
        if modstr=='':
            continue
        ll = modstr.split(",")
        d[ ll[0] ] = np.fromiter((int(a) for a in ll[1:]), dtype=np.int64)
    return d

# ——————————————————————————————————————————————————————————————————————————————————————— #

def a_methylation(read):

    """
    Given a read, get the reference positions of the methylated As and unmethylated As.
    The modified bases of the forward read "A+a." and reverse read "T-a." are both accounted for.
    Produces a numpy array of reference positions of methylated adenosines and unmethylated adenosines.
    """
    ref_pos = get_q2r_positions(read.cigarstring, len(read.query_sequence), read.reference_start)
    
    # get the query sequence to get all the adenosines
    seq = read.query_sequence
    seq_as_bytes = bytes( seq, 'ascii' )
    seq_as_numpy_array = np.frombuffer( seq_as_bytes, np.uint8 )
    qu_poss_of_As = np.where( (seq_as_numpy_array == ord('A')) | (seq_as_numpy_array == ord('T')) )[0]

    rposs_of_query = get_q2r_positions(read.cigarstring, 
        len(read.query_sequence), read.reference_start)
    indices_of_mAs = np.sort(np.concatenate([
        np.cumsum(parse_MM_tag(read)["A+a."] + 1) - 1,
        np.cumsum(parse_MM_tag(read)["T-a."] + 1) - 1
    ]))

    As = rposs_of_query[ qu_poss_of_As ]
    mAs = rposs_of_query[ qu_poss_of_As[indices_of_mAs] ]
    
    uAs = np.setdiff1d(As, mAs, assume_unique=True), 

    return np.array(mAs, dtype=np.int32), np.array(uAs, dtype=np.int32)

# ——————————————————————————————————————————————————————————————————————————————————————— #

def a_methylation_matrix(fibfiles, peaks, padding=2000):
    
    """
    Given a dictionary sorted by biological/technical replicates, 
    Produces the frequency matrix for each position around a TSS.
    Padding is half of window being analyzed. Adjust the window size as needed.
    """
    peak_no = peaks.shape[0]
    length = padding * 2

    binary_uAs = np.zeros((peak_no, length), dtype=np.int16)
    binary_mAs = np.zeros((peak_no, length), dtype=np.int16)

    for i, peak in tqdm(peaks.iterrows(), total=peak_no):
        chrom = peak["chrom"]
        strand = peak["strand"]
        midpoint = (peak["start"] + peak["end"]) // 2
        start = midpoint - padding
        end = midpoint + padding
        
        # group replicates
        reads = itertools.chain.from_iterable(
            f.fetch(chrom, start, end)
            for f in fibfiles
        )
        for read in reads:
            
            mAs, uAs = a_methylation(read)

            if mAs is None:
                continue
                
            norm_mAs = mAs - start
            norm_uAs = uAs - start
            
            if strand == "-":
                norm_mAs = length - norm_mAs
                norm_uAs = length - norm_uAs
                
            norm_mAs = norm_mAs[(norm_mAs >= 0) & (norm_mAs < length)]
            norm_uAs = norm_uAs[(norm_uAs >= 0) & (norm_uAs < length)]

            binary_uAs[i, norm_uAs] += 1
            binary_mAs[i, norm_mAs] += 1
    
    return binary_uAs, binary_mAs
