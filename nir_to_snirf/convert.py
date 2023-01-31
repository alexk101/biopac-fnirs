from pathlib import Path
from typing import List, Dict, Tuple, Any
from datetime import datetime
import numpy as np
import polars as pl
from snirf import Snirf
import os
import h5py

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
os.chdir(dir_path)

TEST = Path()
FILENAME = 'test.snirf'

TS_MARKERS = {
    252: 251,
    2: 1,
    4: 3
}
TS_START = list(TS_MARKERS.values())
TARGET = Path('sample_data').resolve()


def parse_header(data: List[str]):
    output = {}
    data_cleaned = [x.strip() for x in data]
    output['device'] = data_cleaned[0].split(' ')[0]
    time = ' '.join(data_cleaned[1].split(' ')[1:]).split('\t')[1].split(':')
    temp = time[2].split('.')
    temp2 = temp[1].split(' ')
    time[2] = '.'.join([temp[0], temp2[0].ljust(6,'0')+' '+temp2[1]])
    time = ':'.join(time[:2] + ['0'+time[2]])
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


def parse_nir(file: Path) -> Tuple[Dict, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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


def get_posititions():
    data_path = Path('biopac_2000_coords.csv').resolve()
    data = pl.read_csv(data_path)
    data = data.with_columns(
        [
            pl.col('x') / 100,
            pl.col('y') / 100,
            pl.col('z') / 100,
        ]
    )
    detectors = data.filter(pl.col('label').str.contains('detector')).drop('label').to_numpy()
    sources = data.filter(pl.col('label').str.contains('source')).drop('label').to_numpy()
    return detectors, sources


def convert(folder: Path):
    nir_file = list(folder.glob('*.nir'))[0]
    header, baseline, baseline_avgs, org_data = parse_nir(nir_file)
    ts_file = list(folder.glob('*.mrk'))[0]
    timestamps = parse_ts(ts_file)

    amb_col = [f'Optode_{x}_amb' for x in range(1, ((org_data.shape[1]-1)//3)+1)]
    ambient = org_data.select(amb_col)
    times = org_data.get_column('time').to_numpy()
    org_data = org_data.drop(amb_col+['time'])

    output_file = folder / f'{folder.stem}.snirf'

    with h5py.File(output_file, 'w') as fp:
        fp.create_dataset('formatVersion', data='1.1')
        nirs = fp.create_group("/nirs")
        
        # Initialize Metadata
        meta = nirs.create_group("metaDataTags")
        time = header.pop('start_time')
        meta.create_dataset("FrequencyUnit", data = 'Hz')
        meta.create_dataset("LengthUnit", data = 'm')
        meta.create_dataset("TimeUnit", data = 's')
        meta.create_dataset("SubjectID", data = 'temp')
        meta.create_dataset('MeasurementDate', data=datetime.strftime(time, '%Y-%m-%d'))
        meta.create_dataset('MeasurementTime', data=datetime.strftime(time, '%X'))
        for key, val in header.items():
            if val is not None:
                meta.create_dataset(key, data=val)

        # Initialize Data
        data = nirs.create_group('data')
        data.create_dataset('time', data=times)
        data.create_dataset('dataTimeSeries', data=org_data.to_numpy())
        start = 1
        count = 1
        # Initialize Source -> Detector for optodes in MeasurementList
        for source in range(1,5):
            for detector in range(start,4+start):
                for wavelength in range(1,3):
                    mEle = data.create_group(f"measurementList{count}")
                    mEle.create_dataset('dataType', data=1, dtype=np.int32)
                    mEle.create_dataset('dataTypeIndex', data=1, dtype=np.int32)
                    mEle.create_dataset('wavelengthIndex', data=wavelength, dtype=np.int32)
                    mEle.create_dataset('detectorIndex', data=detector, dtype=np.int32)
                    mEle.create_dataset('sourceIndex', data=source, dtype=np.int32)
                    count += 1
                start += 2

        # Add Extra Reference Optodes
        sources = [1,4]
        for detector in range(11,13):
            for x in range(1,2):
                for wavelength in range(1,3):
                    mEle = data.create_group(f"measurementList{count}")
                    mEle.create_dataset('dataType', data=1, dtype=np.int32)
                    mEle.create_dataset('dataTypeIndex', data=1, dtype=np.int32)
                    mEle.create_dataset('wavelengthIndex', data=wavelength, dtype=np.int32)
                    mEle.create_dataset('detectorIndex', data=detector, dtype=np.int32)
                    mEle.create_dataset('sourceIndex', data=sources[detector-11], dtype=np.int32)
                    count += 1

        # Initialize Probe
        probe = nirs.create_group('probe')
        probe.create_dataset('detectorLabels', data=[f'D{x}' for x in range(1,11)])
        probe.create_dataset('sourceLabels', data=[f'S{x}' for x in range(1,5)])
        probe.create_dataset('wavelengths', data=np.array([730,850]).astype(float))
        detectors, sources = get_posititions()
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


def bulk_convert():
    total_subs = len(list(TARGET.iterdir()))
    print(f'Converting {total_subs} subjects to the snirf format...')
    count = 1
    for subject in TARGET.iterdir():
        convert(subject)
        print(f'Converted subject {subject.stem} {count}/{total_subs}')
        count += 1

if __name__ =="__main__":
    bulk_convert()