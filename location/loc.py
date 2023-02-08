import polars as pl
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np


dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
os.chdir(dir_path)


def plot_locations():
    data = pl.read_csv(Path('biopac_2000_coords.csv'))

    sources = data.filter(pl.col('Label').str.starts_with('S'))
    optodes = data.filter(pl.col('Label').str.starts_with('O'))
    ref = data.filter(pl.col('Label').str.starts_with('R'))
    mids = midpoints()

    locs = {'sources': sources, 'optodes': optodes, 'ref': ref, 'mids':mids}

    fig, axes = plt.subplots(1,1)

    for label, vals in locs.items():
        # print(f'ele: {vals.height}')
        x = vals['x'].to_numpy()
        y = vals['z'].to_numpy()
        axes.scatter(x,y, label=label)
    axes.legend()
    axes.grid()
    axes.set_ylim(-5,5)
    fig.savefig('optode_channel_locations.png')


def midpoints():
    data = pl.read_csv(Path('biopac_2000_coords.csv'))

    columns= [pl.col('x'), pl.col('y'), pl.col('z')]
    sources = data.filter(pl.col('Label').str.starts_with('S'))
    optodes = data.filter(pl.col('Label').str.starts_with('O'))
    ref = data.filter(pl.col('Label').str.starts_with('R')).select(columns).to_numpy()
    mids = {}

    source_to_detector = {key: list(range(1+(2*(key-1)),5+(2*(key-1)))) for key in range(1,5)}
    
    for key, val in source_to_detector.items():
        print(f'key: {key}, val: {val}')
        for detector in val:
            s1 = sources.filter(pl.col('Label').str.ends_with(str(key))).select(columns).to_numpy()[0]
            d1 = optodes.filter(pl.col('Label').str.ends_with(str(detector))).select(columns).to_numpy()[0]
            vec = d1-s1
            mids[f'S{key}_D{detector}'] = (vec/2)+s1
    
    # Add two reference optodes
    s1 = sources.filter(pl.col('Label').str.ends_with(str(1))).select(columns).to_numpy()[0]
    d1 = ref[0]
    vec = d1-s1
    mids[f'S1_D11'] = (vec/2)+s1

    s1 = sources.filter(pl.col('Label').str.ends_with(str(4))).select(columns).to_numpy()[0]
    d1 = ref[1]
    vec = d1-s1
    mids[f'S4_D12'] = (vec/2)+s1

    labels = mids.keys()
    mids = np.array([val for val in mids.values()])
    mids = mids.T
    midpoints = pl.DataFrame({'label':labels,'x':mids[0], 'y':mids[1], 'z':mids[2]})
    print(midpoints)
    print(f'num channels: {midpoints.height}')
    return midpoints


plot_locations()
