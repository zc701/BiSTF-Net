import scipy.io as io
import numpy as np
import os
import mne
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm


def step_1_mat_to_npz(root_path):
    print("\n===== Step 1: Converting MAT to NPZ =====")
    mat_root_path = root_path
    eeg_folder = os.path.join(mat_root_path, 'EEG_01-26_MATLAB')
    fnirs_folder = os.path.join(mat_root_path, 'NIRS_01-26_MATLAB')
    save_dir = os.path.join(mat_root_path, 'mat2array')
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isdir(eeg_folder) or not os.path.isdir(fnirs_folder):
        print(f"Error: Source MAT folders not found. Please check paths: '{eeg_folder}' and '{fnirs_folder}'.")
        return

    subject_list = [d.split('-')[0] for d in os.listdir(eeg_folder) if os.path.isdir(os.path.join(eeg_folder, d))]
    subject_list = sorted(list(set(subject_list)))
    print(f"Found {len(subject_list)} subjects.")

    for name in tqdm(subject_list, desc="Converting"):
        try:
            eeg_data_path = os.path.join(eeg_folder, f'{name}-EEG', 'cnt_wg.mat')
            eeg_mrk_path = os.path.join(eeg_folder, f'{name}-EEG', 'mrk_wg.mat')
            fnirs_data_path = os.path.join(fnirs_folder, f'{name}-NIRS', 'cnt_wg.mat')
            fnirs_mrk_path = os.path.join(fnirs_folder, f'{name}-NIRS', 'mrk_wg.mat')

            eeg_data = io.loadmat(eeg_data_path)
            eeg_mrk_data = io.loadmat(eeg_mrk_path)
            fnirs_data = io.loadmat(fnirs_data_path)
            fnirs_mrk_data = io.loadmat(fnirs_mrk_path)

            eeg = eeg_data['cnt_wg'][0, 0][3].T
            eeg_time = eeg_mrk_data['mrk_wg'][0, 0][0]
            hbo = fnirs_data['cnt_wg']['oxy'][0, 0][0, 0][5].T
            hbr = fnirs_data['cnt_wg']['deoxy'][0, 0][0, 0][5].T
            fnirs_time = fnirs_mrk_data['mrk_wg'][0, 0][0]
            label = eeg_mrk_data['mrk_wg'][0, 0][1]

            save_dict = {
                'eeg': eeg, 'eeg_time': eeg_time,
                'hbo': hbo, 'hbr': hbr,
                'fnirs_time': fnirs_time, 'label': label
            }
            np.savez(os.path.join(save_dir, name), **save_dict)
        except Exception as e:
            tqdm.write(f"\nError processing {name}: {e}")  


def step_2_preprocess_continuous(root_path):
    print("\n===== Step 2: Preprocessing Continuous Signals =====")
    input_dir = os.path.join(root_path, 'mat2array')
    save_dir = os.path.join(root_path, 'preprocessed')
    os.makedirs(save_dir, exist_ok=True)

    subject_list = [f.split('.')[0] for f in os.listdir(input_dir) if f.endswith('.npz')]

    eeg_chn_names = ['Fp1', 'AFF5h', 'AFz', 'F1', 'FC5', 'FC1', 'T7', 'C3', 'Cz', 'CP5', 'CP1', 'P7', 'P3', 'Pz', 'POz',
                     'O1', 'Fp2',
                     'AFF6h', 'F2', 'FC2', 'FC6', 'C4', 'T8', 'CP2', 'CP6', 'P4', 'P8', 'O2']
    eeg_info = mne.create_info(ch_names=eeg_chn_names, sfreq=200, ch_types='eeg', verbose=False)
    eeg_info.set_montage('standard_1005', on_missing='ignore')

    fnirs_chn_names = ['AF7', 'AFF5', 'AFp7', 'AF5h', 'AFp3', 'AFF3h', 'AF1', 'AFFz', 'AFpz', 'AF2', 'AFp4', 'FCC3',
                       'C3h', 'C5h', 'CCP3', 'CPP3', 'P3h', 'P5h', 'PPO3', 'AFF4h', 'AF6h', 'AFF6', 'AFp8', 'AF8',
                       'FCC4', 'C6h', 'C4h', 'CCP4', 'CPP4', 'P6h', 'P4h', 'PPO4', 'PPOz', 'PO1', 'PO2', 'POOz']
    fnirs_info = mne.create_info(ch_names=fnirs_chn_names, sfreq=10, ch_types='hbo', verbose=False)
    fnirs_info.set_montage('standard_1005', on_missing='ignore')

    for subject_no in tqdm(subject_list, desc="Preprocessing"):
        with np.load(os.path.join(input_dir, f'{subject_no}.npz')) as data:
            eeg, eeg_time, hbo, hbr, fnirs_time, label = data['eeg'], data['eeg_time'], data['hbo'], data['hbr'], data[
                'fnirs_time'], data['label']

        raw_eeg = mne.io.RawArray(data=eeg[:-2, :], info=eeg_info, verbose=False)
        raw_eeg.notch_filter(np.arange(50, 100, 50), verbose=False)
        raw_eeg.filter(0.5, 50., method='iir', iir_params=dict(order=6, ftype='butter'), verbose=False)
        raw_eeg.set_eeg_reference(ref_channels="average", projection=True, verbose=False)
        raw_eeg.apply_proj(verbose=False)
        raw_eeg.load_data(verbose=False)

        filt_ica_raw = raw_eeg.copy().filter(l_freq=1., h_freq=None, verbose=False)
        ica = mne.preprocessing.ICA(n_components=20, random_state=42)
        ica.fit(filt_ica_raw)

        tqdm.write(f"\nVisualizing ICA for {subject_no}. Please check the plot and enter components to exclude.")
        ica.plot_sources(raw_eeg, title=f'ICA Sources for {subject_no}')
        plt.show()

        input_str = input(f'Enter components to exclude for {subject_no} (e.g., "0 1 5"): ')
        try:
            exclude_list = [int(i) for i in input_str.split()]
            ica.exclude = exclude_list
            tqdm.write(f"Excluding components: {ica.exclude}")
            raw_icaed = ica.apply(raw_eeg.copy(), verbose=False)
            eeg_processed = raw_icaed.get_data()
        except ValueError:
            tqdm.write("Invalid input. No components excluded.")
            eeg_processed = raw_eeg.get_data()

        hbo_raw = mne.io.RawArray(data=hbo, info=fnirs_info.copy().set_channel_types('hbo'), verbose=False)
        hbr_raw = mne.io.RawArray(data=hbr, info=fnirs_info.copy().set_channel_types('hbr'), verbose=False)
        hbo_filtered = hbo_raw.filter(0.01, 0.1, method='iir', iir_params=dict(order=6, ftype='butter'), verbose=False)
        hbr_filtered = hbr_raw.filter(0.01, 0.1, method='iir', iir_params=dict(order=6, ftype='butter'), verbose=False)
        hbo_processed, hbr_processed = hbo_filtered.get_data(), hbr_filtered.get_data()

        save_dict = {'eeg': eeg_processed, 'eeg_time': eeg_time, 'hbo': hbo_processed, 'hbr': hbr_processed,
                     'fnirs_time': fnirs_time, 'label': label}
        np.savez(os.path.join(save_dir, subject_no), **save_dict)


def step_3_epoching(root_path):
    print("\n===== Step 3: Epoching and Baseline Correction =====")
    input_dir = os.path.join(root_path, 'preprocessed')
    save_dir = os.path.join(root_path, 'epoch')
    os.makedirs(save_dir, exist_ok=True)

    task_period = 10;
    eeg_sample_rate = 200;
    eeg_pre_onset = 5;
    eeg_post_onset = task_period
    fnirs_sample_rate = 10;
    fnirs_pre_onset = 5;
    fnirs_post_onset = task_period + 12

    fnirs_chn_names = ['AF7', 'AFF5', 'AFp7', 'AF5h', 'AFp3', 'AFF3h', 'AF1', 'AFFz', 'AFpz', 'AF2', 'AFp4', 'FCC3',
                       'C3h', 'C5h', 'CCP3', 'CPP3', 'P3h', 'P5h', 'PPO3', 'AFF4h', 'AF6h', 'AFF6', 'AFp8', 'AF8',
                       'FCC4', 'C6h', 'C4h', 'CCP4', 'CPP4', 'P6h', 'P4h', 'PPO4', 'PPOz', 'PO1', 'PO2', 'POOz']
    fnirs_info_for_bc = mne.create_info(ch_names=fnirs_chn_names, sfreq=10, ch_types='eeg', verbose=False)
    fnirs_info_for_bc.set_montage('standard_1005', on_missing='ignore')

    subject_list = [f.split('.')[0] for f in os.listdir(input_dir) if f.endswith('.npz')]
    for subject in tqdm(subject_list, desc="Epoching"):
        with np.load(os.path.join(input_dir, f'{subject}.npz')) as data:
            eeg, eeg_time, hbo, hbr, fnirs_time, label = data['eeg'], data['eeg_time'], data['hbo'], data['hbr'], data[
                'fnirs_time'], data['label']

        n_trials = eeg_time.shape[1]
        eeg_epoch_len = int((eeg_pre_onset + eeg_post_onset) * eeg_sample_rate)
        fnirs_epoch_len = int((fnirs_pre_onset + fnirs_post_onset) * fnirs_sample_rate)
        eeg_epoch = np.zeros((n_trials, eeg.shape[0], eeg_epoch_len))
        hbo_epoch = np.zeros((n_trials, hbo.shape[0], fnirs_epoch_len))
        hbr_epoch = np.zeros((n_trials, hbr.shape[0], fnirs_epoch_len))

        for t in range(n_trials):
            eeg_start_idx = int((eeg_time[0, t] / 1000. - eeg_pre_onset) * eeg_sample_rate)
            eeg_epoch[t,] = eeg[:, eeg_start_idx: eeg_start_idx + eeg_epoch_len]

            fnirs_start_idx = int((fnirs_time[0, t] / 1000. - fnirs_pre_onset) * fnirs_sample_rate)
            hbo_epoch[t,] = hbo[:, fnirs_start_idx: fnirs_start_idx + fnirs_epoch_len]
            hbr_epoch[t,] = hbr[:, fnirs_start_idx: fnirs_start_idx + fnirs_epoch_len]

        hbo_epochs_mne = mne.EpochsArray(hbo_epoch, fnirs_info_for_bc, tmin=-fnirs_pre_onset, verbose=False)
        hbr_epochs_mne = mne.EpochsArray(hbr_epoch, fnirs_info_for_bc, tmin=-fnirs_pre_onset, verbose=False)
        hbo_epochs_mne.apply_baseline(baseline=(-5., -2.), verbose=False)
        hbr_epochs_mne.apply_baseline(baseline=(-5., -2.), verbose=False)
        hbo_epoch_bc = hbo_epochs_mne.get_data(copy=True)
        hbr_epoch_bc = hbr_epochs_mne.get_data(copy=True)

        save_dict = {'eeg': eeg_epoch, 'hbo': hbo_epoch_bc, 'hbr': hbr_epoch_bc, 'label': label}
        np.savez(os.path.join(save_dir, subject), **save_dict)


def step_4_spatial_mapping(root_path):
    print("\n===== Step 4: Spatial Mapping (Channel -> Grid) =====")
    input_dir = os.path.join(root_path, 'epoch')
    save_dir = os.path.join(root_path, 'd3')
    os.makedirs(save_dir, exist_ok=True)

    x, y = np.arange(16), np.arange(16)
    xx, yy = np.meshgrid(x, y)
    all_points = np.column_stack((xx.ravel(), yy.ravel()))

    known_eeg_point_coords = np.array(
        [[0., 6.], [2., 5.], [2., 8.], [3., 7.], [5., 2.], [5., 6.], [7., 1.], [7., 4.], [7., 8.], [9., 2.], [9., 6.],
         [11., 2.], [11., 5.], [11., 8.], [13., 8.], [14., 6.], [0., 10.], [2., 11.], [3., 9.], [5., 10.], [5., 14.],
         [7., 12.], [7., 15.], [9., 10.], [9., 14.], [11., 11.], [11., 14.], [14., 10.]])
    unknown_eeg_point_coords = np.array(
        [p for p in all_points if p.tolist() not in known_eeg_point_coords.tolist()]).astype(float)

    known_fnirs_point_coords = np.array(
        [[2., 4.], [3., 4.], [1., 5.], [2., 5.], [1., 7.], [3., 6.], [2., 7.], [3., 8.], [1., 8.], [2., 9.], [1., 9.],
         [6., 4.], [7., 5.], [7., 3.], [8., 4.], [10., 5.], [11., 6.], [11., 4.], [12., 5.], [3., 10.], [2., 11.],
         [3., 12.], [1., 11.], [2., 12.], [6., 12.], [7., 13.], [7., 11.], [8., 12.], [10., 11.], [11., 12.],
         [11., 10.], [12., 11.], [12., 8.], [13., 7.], [13., 9.], [14., 8.]])
    unknown_fnirs_point_coords = np.array(
        [p for p in all_points if p.tolist() not in known_fnirs_point_coords.tolist()]).astype(float)

    subject_list = [f.split('.')[0] for f in os.listdir(input_dir) if f.endswith('.npz')]
    for subject in tqdm(subject_list, desc="Spatial Mapping"):
        with np.load(os.path.join(input_dir, f'{subject}.npz')) as data:
            eeg, hbo, hbr, label = data['eeg'], data['hbo'], data['hbr'], data['label']

        eeg_3d, hbo_3d, hbr_3d = np.zeros((eeg.shape[0], 16, 16, eeg.shape[2])), np.zeros(
            (hbo.shape[0], 16, 16, hbo.shape[2])), np.zeros((hbr.shape[0], 16, 16, hbr.shape[2]))

        for e in range(eeg.shape[0]):
            for t in range(eeg.shape[2]):
                eeg_2d = np.ones((16, 16))
                vals, coords = eeg[e, :, t], known_eeg_point_coords
                interp_vals = griddata(coords, vals, unknown_eeg_point_coords, method='cubic')
                for k, (y_coord, x_coord) in enumerate(coords): eeg_2d[int(y_coord), int(x_coord)] = vals[k]
                for u, (y_coord, x_coord) in enumerate(unknown_eeg_point_coords): eeg_2d[int(y_coord), int(x_coord)] = \
                interp_vals[u]

                nan_mask = np.isnan(eeg_2d)
                if np.any(nan_mask):
                    known_points, known_values, nan_points = np.argwhere(~nan_mask), eeg_2d[~nan_mask], np.argwhere(
                        nan_mask)
                    eeg_2d[nan_mask] = griddata(known_points, known_values, nan_points, method='nearest')
                eeg_3d[e, :, :, t] = eeg_2d

            for ft in range(hbo.shape[2]):
                hbo_2d, hbr_2d = np.ones((16, 16)), np.ones((16, 16))
                vals_hbo, vals_hbr, coords = hbo[e, :, ft], hbr[e, :, ft], known_fnirs_point_coords
                interp_hbo, interp_hbr = griddata(coords, vals_hbo, unknown_fnirs_point_coords,
                                                  method='cubic'), griddata(coords, vals_hbr,
                                                                            unknown_fnirs_point_coords, method='cubic')

                for k, (y_coord, x_coord) in enumerate(coords):
                    hbo_2d[int(y_coord), int(x_coord)] = vals_hbo[k]
                    hbr_2d[int(y_coord), int(x_coord)] = vals_hbr[k]
                for u, (y_coord, x_coord) in enumerate(unknown_fnirs_point_coords):
                    hbo_2d[int(y_coord), int(x_coord)] = interp_hbo[u]
                    hbr_2d[int(y_coord), int(x_coord)] = interp_hbr[u]

                for img in [hbo_2d, hbr_2d]:
                    nan_mask = np.isnan(img)
                    if np.any(nan_mask):
                        known_points, known_values, nan_points = np.argwhere(~nan_mask), img[~nan_mask], np.argwhere(
                            nan_mask)
                        img[nan_mask] = griddata(known_points, known_values, nan_points, method='nearest')

                hbo_3d[e, :, :, ft], hbr_3d[e, :, :, ft] = hbo_2d, hbr_2d

        save_dict = {'eeg': eeg_3d, 'hbo': hbo_3d, 'hbr': hbr_3d, 'label': label}
        np.savez(os.path.join(save_dir, subject), **save_dict)


def step_5_windowing_and_finalizing(root_path):
    print("\n===== Step 5: Windowing and Finalizing Model Input =====")
    input_dir = os.path.join(root_path, 'd3')
    save_dir = os.path.join(root_path, 'model_input')
    os.makedirs(save_dir, exist_ok=True)

    win_length, eeg_segments, fnirs_segments, fnirs_lag_length = 3, 10, 22, 11
    eeg_srate, fnirs_srate = 200, 10

    subject_list = [f.split('.')[0] for f in os.listdir(input_dir) if f.endswith('.npz')]
    for subject in tqdm(subject_list, desc="Windowing"):
        with np.load(os.path.join(input_dir, f'{subject}.npz')) as data:
            eeg, hbo, hbr, label = data['eeg'], data['hbo'], data['hbr'], data['label']

        n_trials = eeg.shape[0]
        eeg_win_len, fnirs_win_len = int(win_length * eeg_srate), int(win_length * fnirs_srate)
        time_offset_sec = 3
        eeg_start_offset, fnirs_start_offset = int(time_offset_sec * eeg_srate), int(time_offset_sec * fnirs_srate)

        eeg_window = np.zeros((n_trials, eeg_segments, 16, 16, eeg_win_len))
        hbo_window = np.zeros((n_trials, fnirs_segments, 16, 16, fnirs_win_len))
        hbr_window = np.zeros((n_trials, fnirs_segments, 16, 16, fnirs_win_len))

        for e in range(n_trials):
            for w in range(eeg_segments):
                start = eeg_start_offset + w * eeg_srate
                end = start + eeg_win_len
                eeg_window[e, w] = eeg[e, :, :, start:end]

            for fw in range(fnirs_segments):
                start = fnirs_start_offset + fw * fnirs_srate
                end = start + fnirs_win_len
                hbo_window[e, fw] = hbo[e, :, :, start:end]
                hbr_window[e, fw] = hbr[e, :, :, start:end]

        fnirs_session_dataset = np.zeros((n_trials, eeg_segments, fnirs_lag_length, 16, 16, fnirs_win_len, 2))
        for e in range(n_trials):
            for w in range(eeg_segments):
                hbo_sample = hbo_window[e, w: w + fnirs_lag_length]
                hbr_sample = hbr_window[e, w: w + fnirs_lag_length]
                fnirs_session_dataset[e, w, :, :, :, :, 0] = hbo_sample
                fnirs_session_dataset[e, w, :, :, :, :, 1] = hbr_sample

        eeg_input = np.expand_dims(eeg_window, axis=-1).reshape(-1, 16, 16, eeg_win_len, 1)
        fnirs_input = fnirs_session_dataset.reshape(-1, fnirs_lag_length, 16, 16, fnirs_win_len, 2)
        label_input = np.repeat(label.T, repeats=eeg_segments, axis=0)

        save_dict = {'eeg': eeg_input, 'fnirs': fnirs_input, 'label': label_input}
        np.savez(os.path.join(save_dir, subject), **save_dict)


if __name__ == '__main__':
    
    DATASET_ROOT_PATH = r'D:\new_dataset'

    mne.set_log_level('WARNING')  

    step_1_mat_to_npz(DATASET_ROOT_PATH)
    step_2_preprocess_continuous(DATASET_ROOT_PATH)
    step_3_epoching(DATASET_ROOT_PATH)
    step_4_spatial_mapping(DATASET_ROOT_PATH)
    step_5_windowing_and_finalizing(DATASET_ROOT_PATH)

    print("\n" + "=" * 20 + " All preprocessing steps completed! " + "=" * 20)
    print(f"Final model input files have been saved to: {os.path.join(DATASET_ROOT_PATH, 'model_input')}")
