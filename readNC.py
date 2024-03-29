import MDAnalysis as mda


def readNC(prm,crd):
    #mpd=mda.coordinates.TRJ.NCDFReader(fileName)
    uni=mda.Universe(prm,crd)
    return uni

