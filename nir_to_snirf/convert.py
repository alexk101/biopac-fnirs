from pathlib import Path
from typing import List, Dict, Tuple, Any
from datetime import datetime
import numpy as np
import polars as pl
from snirf import Snirf
import os
import h5py
from shutil import copy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import transformations as trs

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
os.chdir(dir_path)

origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]

TS_MARKERS = {
    252: 251,
    2: 1,
    4: 3
}
TS_START = list(TS_MARKERS.values())
TARGET = Path('fnirs-pilot-data').resolve()
OUTPUT = Path('pilot-snirfs').resolve()

def parse_header(data: List[str]):
    output = {}
    data_cleaned = [x.strip() for x in data]
    output['device'] = data_cleaned[0].split(' ')[0]
    time = ' '.join(data_cleaned[1].split(' ')[1:]).split('\t')[1].split(':')
    temp = time[2].split('.')
    temp2 = temp[1].split(' ')
    time[2] = '.'.join([temp[0], temp2[0].ljust(6,'0')+' '+temp2[1]])
    if len(time[2]) < 2:
        time = ':'.join(time[:2] + ['0'+time[2]])
    else:
        time = ':'.join(time)
    output['start_time'] = datetime.strptime(time, '%a %b %d %H:%M:%S.%f %Y')
    output['start_code'] = np.array(data_cleaned[3].split(':')[1].split('\t')[1:]).astype(float)
    output['freq_code'] = float(data_cleaned[4].split(':')[1].strip())

    temp = data_cleaned[5].split(':')
    temp.remove('')
    if len(temp) > 1:
        output['current'] = int(temp[1])
    else:
        output['current'] = None

    temp = data_cleaned[6].split(':')
    temp.remove('')
    if len(temp) > 1:
        output['gain'] = int(temp[1])
    else:
        output['gain'] = None
    
    return output


def parse_data(data: List[str]):
    data_cleaned = [x.strip().split('\t')[:-1] for x in data]
    time = np.array([x.pop(0) for x in data_cleaned]).astype(np.float32)  # type: ignore
    data_cleaned = np.array(data_cleaned).astype(np.float32).T
    col_template = ['Optode_{num}_730', 'Optode_{num}_amb', 'Optode_{num}_850']
    data_cleaned = {col_template[x%3].format(num=(x//3)+1): data_cleaned[x] for x in range(data_cleaned.shape[0])}
    data_cleaned['time'] = time
    data_cleaned = pl.DataFrame(data_cleaned)
    return data_cleaned


PARSING = {
    'Baseline Started':parse_header,
    'Baseline values':parse_data,
    'Baseline end':parse_data,
    'Connection Closed':parse_data
}
MARKERS = list(PARSING.keys())


def parse_nir(file: Path, crop: int=0) -> Tuple[Dict, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    current_marker = 0
    all_data = []

    with open(file, 'r') as fp:
        current_data: List[str] = []
        for line in fp:
            if MARKERS[current_marker] in line:
                all_data.append(PARSING[MARKERS[current_marker]](current_data))
                current_data = []
                current_marker+=1
                continue
            current_data.append(line)
    if crop != 0:
        adjusted = crop
        if adjusted < 0:
            adjusted = 0
        all_data[3] = all_data[3].filter(pl.col('time') > adjusted)
    return tuple(all_data)


def parse_ts(file: Path):
    count = 0
    keys = ['time', 'code']

    stacks = [[] for x in range(len(TS_MARKERS.keys()))]
    output = stacks.copy()

    with open(file, 'r') as fp:
        for line in fp:
            if count > 5:
                sample: Dict[str, Any] = dict(zip(keys, line.split('\t')[:2]))
                sample['time'] = float(sample['time'])
                sample['code'] = int(sample['code'])
                if sample['code'] in TS_START:
                    stacks[TS_START.index(sample['code'])].append(sample)
                else:
                    start = stacks[TS_START.index(TS_MARKERS[sample['code']])].pop()
                    output[TS_START.index(TS_MARKERS[sample['code']])].append([start['time'], sample['time']-start['time'],1.0])

            count += 1
    for x in range(len(output)):
        output[x] = np.array(output[x]).astype(float) # type: ignore
    return dict(zip(TS_MARKERS.values(), output))


def parabola_deform(data: pl.DataFrame):
    # parabola deformation * attenuation factor + y 
    output = data.with_columns([((((pl.col('x')**2)/-8)+((pl.col('y')**2)/15)) * ((-(pl.col('x')*0.09).cos()**2)+1) + 1 + pl.col('y')).alias('y_def')])
    output = output.drop('y').rename({'y_def': 'y'})
    return output


def rotate_frame(data: pl.DataFrame, theta: float = -0.2):
    temp = data.select([pl.col('x'), pl.col('y'), pl.col('z')]).with_column(pl.Series('w', np.zeros(data.height))).to_numpy()
    Rz = trs.rotation_matrix(theta, xaxis)
    output = []
    # print(temp)
    for point in temp:
        val = point@Rz
        output.append(val)
    output = np.array(output).T
    output = pl.DataFrame({'label': data['label'], 'x': output[0], 'y': output[1], 'z': output[2]})
    return output

def prefrontal_cortex_map(data: pl.DataFrame):
    data = parabola_deform(data)
    data = rotate_frame(data)
    data = data.with_column(pl.col('y') + 11)
    data = data.with_column(pl.col('z') + 6)
    return data


def get_posititions(debug: bool=False):
    data_path = Path('../location/biopac_2000_coords.csv').resolve()
    data = pl.read_csv(data_path)
    data = prefrontal_cortex_map(data) 

    data = data.with_columns(
        [
            pl.col('x') / 100,
            pl.col('y') / 100,
            pl.col('z') / 100,
        ]
    )
    detectors = data.filter(pl.col('label').str.starts_with('O')).drop('label')
    sources = data.filter(pl.col('label').str.starts_with('S')).drop('label')
    reference = data.filter(pl.col('label').str.starts_with('R')).drop('label')
    data = {'detectors':detectors, 'sources':sources, 'reference':reference}

    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for key,val in data.items():
            ax.scatter(val['x'], val['y'], val['z'], label=key)

        ax.set_xlim(-0.2,0.2)
        ax.set_ylim(-0.2,0.2)
        ax.set_zlim(-0.2,0.2)
        plt.show(block=True)

    all_detectors = pl.concat([detectors, reference])
    return all_detectors.to_numpy(), sources.to_numpy()


def debug_add(data, dtype, mEle: h5py.Group, name: str):
    dbug = {'name': name, 'data': data}
    mEle.create_dataset(name, data=data, dtype=dtype)
    return dbug

def add_measurment_list(data: h5py.Group, debug: bool=True):
    start = 1
    count = 1
    dbug = []

    # Initialize Source -> Detector for optodes in MeasurementList
    for source in range(1,5):
        for detector in range(start,4+start):
            for wavelength in range(1,3):
                cur_list = f"measurementList{count}"
                m_list = {cur_list: []}
                mEle = data.create_group(f"measurementList{count}")
                m_list[cur_list].append(debug_add(1, np.int32, mEle, 'dataType'))
                m_list[cur_list].append(debug_add(1, np.int32, mEle, 'dataTypeIndex'))
                m_list[cur_list].append(debug_add(wavelength, np.int32, mEle, 'wavelengthIndex'))
                m_list[cur_list].append(debug_add(detector, np.int32, mEle, 'detectorIndex'))
                m_list[cur_list].append(debug_add(source, np.int32, mEle, 'sourceIndex'))
                count += 1
                dbug.append(m_list)
        start += 2

    # Add Extra Reference Optodes
    sources = [1,4]
    for detector in range(11,13):
        for x in range(1,2):
            for wavelength in range(1,3):
                cur_list = f"measurementList{count}"
                m_list = {cur_list: []}
                mEle = data.create_group(f"measurementList{count}")
                m_list[cur_list].append(debug_add(1, np.int32, mEle, 'dataType'))
                m_list[cur_list].append(debug_add(1, np.int32, mEle, 'dataTypeIndex'))
                m_list[cur_list].append(debug_add(wavelength, np.int32, mEle, 'wavelengthIndex'))
                m_list[cur_list].append(debug_add(detector, np.int32, mEle, 'detectorIndex'))
                m_list[cur_list].append(debug_add(sources[detector-11], np.int32, mEle, 'sourceIndex'))
                count += 1
                dbug.append(m_list)

    if debug:
        for mlist in dbug:
            for key, val in mlist.items():
                print(f'{key}')
                for mele in val:
                    print(f'\t{mele}')


def convert(folder: Path, crop: bool=True, debug: bool=False):
    ts_file = list(folder.glob('*.mrk'))[0]
    timestamps = parse_ts(ts_file)
    bs_start = timestamps[251][0][0] - 5

    nir_file = list(folder.glob('*.nir'))[0]
    if crop:
        header, _, _, org_data = parse_nir(nir_file, bs_start)
    else:
        header, _, _, org_data = parse_nir(nir_file)

    if crop:
        # Crop Timestamps
        cropped = {}
        for key, val in timestamps.items():
            temp = []
            for sample in val:
                temp2 = sample
                temp2[0] -= bs_start
                temp.append(temp2)
            cropped[key] = np.array(temp)

        timestamps = cropped

    amb_col = [f'Optode_{x}_amb' for x in range(1, ((org_data.shape[1]-1)//3)+1)]
    ambient = org_data.select(amb_col)
    old_time = org_data.get_column('time').to_numpy().copy()
    if crop:
        old_time -= bs_start
    org_data.drop_in_place('time')
    time = np.linspace(0, (len(old_time)-1)*(0.1), int(len(old_time)))

    for column in org_data.columns:
        spline = interp.make_interp_spline(time, org_data[column], k=3)
        org_data.replace(column, pl.Series(spline(time)))

    # Ignore ambient columns for now
    org_data = org_data.drop(amb_col)

    output_file = folder / f'{folder.stem}.snirf'
    # Convert mV to V (my assumption is that the output of the file is mV)
    optodes_V = org_data.to_numpy() * 1e-3

    with h5py.File(output_file, 'w') as fp:
        fp.create_dataset('formatVersion', data='1.1')
        nirs = fp.create_group("/nirs")
        
        # Initialize Metadata
        meta = nirs.create_group("metaDataTags")
        start_time = header.pop('start_time')
        meta.create_dataset("FrequencyUnit", data = 'Hz')
        meta.create_dataset("LengthUnit", data = 'm')
        meta.create_dataset("TimeUnit", data = 's')
        meta.create_dataset("SubjectID", data = 'temp')
        meta.create_dataset('MeasurementDate', data=datetime.strftime(start_time, '%Y-%m-%d'))
        meta.create_dataset('MeasurementTime', data=datetime.strftime(start_time, '%X'))
        for key, val in header.items():
            if val is not None:
                meta.create_dataset(key, data=val)

        # Initialize Data
        data = nirs.create_group('data1')
        data.create_dataset('time', data=time)

        data.create_dataset('dataTimeSeries', data=optodes_V)
        add_measurment_list(data, False)

        # Initialize Probe
        probe = nirs.create_group('probe')
        probe.create_dataset('detectorLabels', data=[f'D{x}' for x in range(1,13)])
        probe.create_dataset('sourceLabels', data=[f'S{x}' for x in range(1,5)])
        probe.create_dataset('wavelengths', data=np.array([730,850]).astype(float))
        detectors, sources = get_posititions(debug)

        probe.create_dataset('detectorPos3D', data=detectors)
        probe.create_dataset('sourcePos3D', data=sources)

        # Initialize Stims
        count = 1
        for key, val in timestamps.items():
            stim = nirs.create_group(f'stim{str(count)}')
            stim.create_dataset('data', data=val)
            stim.create_dataset('name', data=str(float(key)))
            count += 1

    snirf = Snirf(str(output_file))
    validate = snirf.validate()
    assert validate.is_valid


def bulk_convert(debug: bool=False):
    total_subs = len(list(TARGET.iterdir()))
    print(f'Converting {total_subs} subjects to the snirf format...')
    for ind, subject in enumerate(TARGET.iterdir()):
        if not (subject / f'{subject.stem}.snirf').exists():
            if (ind == total_subs-1):
                convert(subject, debug=debug)
            else:
                convert(subject)
            print(f'Converted subject {subject.stem} {ind+1}/{total_subs}')
        else:
            print(f'Subject {subject.stem} already converted {ind+1}/{total_subs}')


def analyze_interpolation(folder: Path, crop: bool=True):
    ts_file = list(folder.glob('*.mrk'))[0]
    timestamps = parse_ts(ts_file)
    bs_start = timestamps[251][0][0] - 5

    nir_file = list(folder.glob('*.nir'))[0]
    if crop:
        _, _, _, org_data = parse_nir(nir_file, bs_start)
        time = org_data.get_column('time').to_numpy() -bs_start
        org_data.drop_in_place('time')
    else:
        _, _, _, org_data = parse_nir(nir_file)
        time = org_data.get_column('time').to_numpy() - bs_start
        org_data.drop_in_place('time')

    opt_1_730 = org_data[org_data.columns[0]].to_numpy()

    
    times_new = np.linspace(0, len(time)*(0.1), int(len(time))+1)
    plt.plot(list(range(len(times_new))), times_new)
    plt.savefig('bruh.png')
    
    fig, axes = plt.subplots(1,1, figsize=(15,5))
    axes.plot(time,opt_1_730, label='org')

    bspl = interp.make_interp_spline(time, opt_1_730, k=3)
    print(bspl(times_new))

    axes.plot(times_new, bspl(times_new), label='interp')
    axes.legend()

    fig.savefig('interpolation_ex.png')


def analyze_linearity(folder: Path, crop: bool=True):
    ts_file = list(folder.glob('*.mrk'))[0]
    timestamps = parse_ts(ts_file)
    bs_start = timestamps[251][0][0] - 5

    nir_file = list(folder.glob('*.nir'))[0]
    if crop:
        _, _, _, org_data = parse_nir(nir_file, bs_start)
        time = org_data.get_column('time').to_numpy() - bs_start
        org_data.drop_in_place('time')
    else:
        _, _, _, org_data = parse_nir(nir_file)
        time = org_data.get_column('time').to_numpy()
        org_data.drop_in_place('time')

    # fig, axes = plt.subplots(len(org_data.columns),1, figsize=(17,25))
    # for ind, column in enumerate(org_data.columns):
    #     axes[ind].plot(time, org_data[column])
    # fig.savefig('all_data.png')

    # print(org_data.columns)
    data_good = org_data[org_data.columns[0]].to_numpy()
    data_bad = org_data[org_data.columns[14]].to_numpy()

    good_m, good_b = np.polyfit(time, data_good, deg=1) # type: ignore
    bad_m, bad_b = np.polyfit(time, data_bad, deg=1) # type: ignore

    good_l = np.array([(x*good_m)+good_b for x in time])
    bad_l = np.array([(x*bad_m)+bad_b for x in time])

    err_good = np.abs(good_l - data_good).mean()
    err_bad = np.abs(bad_l - data_bad).mean()
    print(f'bad error: {err_bad}, good error: {err_good}')

    fig, axes = plt.subplots(2,1)
    axes[0].plot(time,good_l, label='linear fit - good data')
    axes[0].plot(time,data_good, label='orignal data - good data')
    axes[0].legend()
    axes[1].plot(time,bad_l, label='linear fit - bad data')
    axes[1].plot(time,data_bad, label='orignal data - bad data')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig('linearity.png')

def get_individual_sampling_rate(folder: Path, crop:bool=True):
    ts_file = list(folder.glob('*.mrk'))[0]
    timestamps = parse_ts(ts_file)
    bs_start = timestamps[251][0][0] - 5

    nir_file = list(folder.glob('*.nir'))[0]
    if crop:
        _, _, _, org_data = parse_nir(nir_file, bs_start)
        time = org_data.get_column('time').to_numpy()
    else:
        _, _, _, org_data = parse_nir(nir_file)
        time = org_data.get_column('time').to_numpy()
    samples = []
    count = 0
    while count+1 < len(time):
        samples.append(1 / (time[count+1] - time[count]))
        count += 1
    return pl.DataFrame({'subject':[folder.stem]*len(samples), 'rate (hz)':samples})


def get_avg_sampling_rate():
    results = []
    for subject in TARGET.iterdir():
        rate = get_individual_sampling_rate(subject)
        # print(f'Subject {subject.stem} avg sampling rate: {rate:.2f} hz')
        results.append(rate)

    out_path = Path('variability')
    out_path.mkdir(parents=True, exist_ok=True)
    for subject in results:
        fig, axes = plt.subplots(1,1)
        axes.plot(list(range(len(subject.rows()))), subject['rate (hz)'])
        name = f"{list(set(subject['subject']))[0]}_variability.png"
        fig.savefig(str(out_path / name))
    report = pl.concat(results)
    print(report.describe())


def get_snirfs():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for subject in TARGET.iterdir():
        snirf_path = list(subject.glob('*.snirf'))[0]
        copy(snirf_path, OUTPUT)


def rm_snirfs():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for subject in TARGET.iterdir():
        if list(subject.glob('*.snirf')):
            snirf_path = list(subject.glob('*.snirf'))[0]
            os.remove(str(snirf_path))


if __name__ =="__main__":
    # analyze_linearity(Path('fnirs-pilot-data/allison_no_eyetracking'))
    # get_avg_sampling_rate()
    # analyze_interpolation(Path('fnirs-pilot-data/alex_no_eyetracking'))
    rm_snirfs()
    bulk_convert()
    get_snirfs()
    # test()