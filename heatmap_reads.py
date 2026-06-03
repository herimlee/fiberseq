import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from m6a_matrix import *

# must have nucleosomes called from ft add-nucleosomes command

def nuc_occupancy(read):
    """
    Extract the nucleosome start and corresponding nucleosome length information.
    Create a list of nucleosome start and end sites.
    """
    try:
        ns = np.array(read.get_tag("ns"))
        nl = np.array(read.get_tag("nl"))
    except KeyError:
        return []
    
    indices_of_nucs = np.concatenate([np.arange(s, s + l) for s, l in zip(ns, nl)])
    nucs = rposs_of_query[indices_of_nucs]
    np.array(nucs, dtype=np.int32)

def add_zeros(al, start, end):
    front = al[start:end]
    pad = (end - start) - len(front)
    padded = np.zeros(pad, dtype=np.int32)
    al = np.concatenate([front, padded])
    return al

def heatmap(group, peak, padding):
    """
    Create a heatmap that marks the locations of methylated and unmethylated adenosines.
    Also marks the areas that are occupied by nucleosomes. 
    """
    length = padding * 2

    chrom = peak["chrom"]
    strand = peak["strand"]
    midpoint = (peak["start"] + peak["end"]) // 2
    start = midpoint - padding
    end = midpoint + padding
    reads = group.fetch(chrom, start, end)

    tracks = []
    mAs_means = []
    
    for read in reads:
        mAs, uAs = a_methylation(read)
        nucs = nuc_occupancy(read)
        norm_mAs  = mAs  - start
        norm_uAs  = uAs  - start
        norm_nucs = nucs - start
        
        if strand == "-":
            norm_mAs  = length - norm_mAs
            norm_uAs  = length - norm_uAs
            norm_nucs = length - norm_nucs
            
        norm_mAs  = norm_mAs[ (norm_mAs  >= 0) & (norm_mAs  < length)]
        norm_uAs  = norm_uAs[ (norm_uAs  >= 0) & (norm_uAs  < length)]
        norm_nucs = norm_nucs[(norm_nucs >= 0) & (norm_nucs < length)]
        
        mAs_set  = set(norm_mAs)
        uAs_set  = set(norm_uAs)
        nucs_set = set(norm_nucs)

        track = np.zeros(length, dtype=int)
        # coded by numbers
        track[sorted(uAs_set - nucs_set)] = 1 # uA only
        track[sorted(mAs_set - nucs_set)] = 2 # mA only
        track[sorted(nucs_set - mAs_set  - uAs_set)]  = 3 # Nucleosome only
        track[sorted(nucs_set & uAs_set)]  = 4 # Nucleosome + uA
        track[sorted(nucs_set & mAs_set)]  = 5 # Nucleosome + mA
        tracks.append(track)
        mAs_means.append(norm_mAs.mean() if len(norm_mAs) > 0 else 0)

    order = np.argsort(mAs_means)[::-1]
    track_matrix = np.vstack(tracks)[order]
    n_reads = len(tracks)
    track_matrix = np.vstack(tracks)

    cmap = matplotlib.colors.ListedColormap(["white", "#b9b9b9", "#79bcf0", "#ccc183", "#f5d817", "#00c478"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(14, max(10, n_reads * 0.5 + 1)))
    ax.imshow(track_matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="none")
    ax.axvline(x=padding, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Position relative to midpoint", fontsize=12)
    ax.set_ylabel("Reads", fontsize=12)
    ax.set_title(f"{chrom}: {start}-{end}", fontsize=13, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color="#b9b9b9", label="uA only"),
        mpatches.Patch(color="#79bcf0", label="mA only"),
        mpatches.Patch(color="#ccc183", label="Nucleosome only"),
        mpatches.Patch(color="#f5d817", label="Nucleosome + uA"),
        mpatches.Patch(color="#00c478", label="Nucleosome + mA"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.show()