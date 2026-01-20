import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import mne
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
from model import BiSTFNet

def build_model_input_from_interpolated(
        eeg_interp, hbo_interp, hbr_interp, label_interpolated,
        eeg_fs, fnirs_fs,
        win_len_sec=3,
        time_offset_sec=3,
        step_sec=1,
        n_windows_per_trial=10,
        fnirs_lag=11
):
    n_trials = eeg_interp.shape[0]
    eeg_win_len = int(win_len_sec * eeg_fs)
    fnirs_win_len = int(win_len_sec * fnirs_fs)

    eeg_offset = int(time_offset_sec * eeg_fs)
    fnirs_offset = int(time_offset_sec * fnirs_fs)

    eeg_step = int(step_sec * eeg_fs)
    fnirs_step = int(step_sec * fnirs_fs)

    eeg_windows_list = []
    fnirs_windows_list = []
    labels_list = []

    for trial_idx in range(n_trials):
        current_eeg_trial = eeg_interp[trial_idx]
        current_hbo_trial = hbo_interp[trial_idx]
        current_hbr_trial = hbr_interp[trial_idx]
        current_label = label_interpolated[:, trial_idx]

        for i in range(n_windows_per_trial):
            eeg_start = eeg_offset + i * eeg_step
            eeg_end = eeg_start + eeg_win_len

            fnirs_start_first_lag = fnirs_offset + i * fnirs_step
            fnirs_end_last_lag = fnirs_start_first_lag + (fnirs_lag - 1) * fnirs_step + fnirs_win_len

            if eeg_end > current_eeg_trial.shape[-1] or fnirs_end_last_lag > current_hbo_trial.shape[-1]:
                break

            eeg_window = current_eeg_trial[:, :, eeg_start:eeg_end]

            fnirs_lag_set_hbo = np.zeros((fnirs_lag, 16, 16, fnirs_win_len), dtype=hbo_interp.dtype)
            fnirs_lag_set_hbr = np.zeros_like(fnirs_lag_set_hbo)

            for lag_idx in range(fnirs_lag):
                s = fnirs_start_first_lag + lag_idx * fnirs_step
                e = s + fnirs_win_len
                fnirs_lag_set_hbo[lag_idx] = current_hbo_trial[:, :, s:e]
                fnirs_lag_set_hbr[lag_idx] = current_hbr_trial[:, :, s:e]

            eeg_windows_list.append(eeg_window)
            fnirs_window_combined = np.stack([fnirs_lag_set_hbo, fnirs_lag_set_hbr], axis=-1)
            fnirs_windows_list.append(fnirs_window_combined)
            labels_list.append(current_label)

    if not eeg_windows_list:
        eeg_shape = (0, 16, 16, eeg_win_len, 1)
        fnirs_shape = (0, fnirs_lag, 16, 16, fnirs_win_len, 2)
        labels_shape = (0, label_interpolated.shape[0])
        return np.empty(eeg_shape), np.empty(fnirs_shape), np.empty(labels_shape)

    eeg_out = np.expand_dims(np.array(eeg_windows_list), axis=-1)
    fnirs_out = np.array(fnirs_windows_list)
    labels_out = np.array(labels_list)

    return eeg_out, fnirs_out, labels_out


class EEGfNIRSDataset(Dataset):
    def __init__(self, eeg, fnirs, labels):
        self.eeg = eeg
        self.fnirs = fnirs
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            self.labels = labels.argmax(axis=1)
        else:
            self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'eeg_input': torch.FloatTensor(self.eeg[idx]),
            'fnirs_input': torch.FloatTensor(self.fnirs[idx])
        }, {
            'class_output': torch.LongTensor([self.labels[idx]]).squeeze(),
            'eeg_output': torch.LongTensor([self.labels[idx]]).squeeze()
        }


class Trainer:
    def __init__(self, model, total_epochs, device=device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(device)

        self.optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_epochs, eta_min=1e-6)

        self.w_class = 2.0
        self.w_eeg_aux = 1.0
        self.w_bicmg = 0.1
        self.w_ata = 0.2
        self.w_ef_max = 1.0
        self.ef_warmup_epochs = 50

    def train_epoch(self, dataloader, current_epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, leave=False)

        for batch in pbar:
            inputs, targets = batch
            eeg = inputs['eeg_input'].to(self.device, non_blocking=True)
            fnirs = inputs['fnirs_input'].to(self.device, non_blocking=True)
            class_target = targets['class_output'].to(self.device, non_blocking=True)
            eeg_target = targets['eeg_output'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(eeg, fnirs)

            internal_losses = outputs['losses']
            class_loss = self.criterion(outputs['class_output'], class_target)
            eeg_loss = self.criterion(outputs['eeg_output'], eeg_target)

            w_ef_current = self.w_ef_max * min(1.0, current_epoch / self.ef_warmup_epochs)

            total_batch_loss = (
                    self.w_class * class_loss +
                    self.w_eeg_aux * eeg_loss +
                    self.w_bicmg * (internal_losses.get('bicmg1_loss', 0) + internal_losses.get('bicmg2_loss', 0)) +
                    w_ef_current * internal_losses.get('ef_loss', 0) +
                    self.w_ata * internal_losses.get('ata_loss', 0)
            )

            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()
            pbar.set_postfix(
                {'loss': f'{total_batch_loss.item():.4f}', 'lr': f'{self.optimizer.param_groups[0]["lr"]:.1e}'})

        self.scheduler.step()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_class_preds, all_class_targets = [], []
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                eeg = inputs['eeg_input'].to(self.device, non_blocking=True)
                fnirs = inputs['fnirs_input'].to(self.device, non_blocking=True)
                class_target = targets['class_output'].to(self.device, non_blocking=True)
                outputs = self.model(eeg, fnirs)
                class_loss = self.criterion(outputs['class_output'], class_target)
                total_loss += class_loss.item()
                _, class_pred = torch.max(outputs['class_output'], 1)
                all_class_preds.extend(class_pred.cpu().numpy())
                all_class_targets.extend(class_target.cpu().numpy())
        if not all_class_targets:
            return {'loss': 0, 'class_acc': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'kappa': 0}
        unique_labels_true = np.unique(all_class_targets)
        if len(unique_labels_true) < 2:
            accuracy = np.mean(np.array(all_class_preds) == np.array(all_class_targets))
            return {'loss': total_loss / len(dataloader), 'class_acc': accuracy, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'kappa': 0.0}
        known_labels = [0, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(all_class_targets, all_class_preds, average='macro', zero_division=0, labels=known_labels)
        kappa = cohen_kappa_score(all_class_targets, all_class_preds, labels=known_labels)
        accuracy = np.mean(np.array(all_class_preds) == np.array(all_class_targets))
        return {'loss': total_loss / len(dataloader), 'class_acc': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1, 'kappa': kappa}


def save_results(results, filename='results.xlsx'):
    all_folds_data = []
    metrics_to_process = ['class_acc', 'precision', 'recall', 'f1_score', 'kappa']
    for subject, fold_results in results.items():
        summary_stats = {}
        for metric in metrics_to_process:
            metric_list = [fold[metric] for fold in fold_results.values() if metric in fold]
            summary_stats[f'avg_{metric}'] = np.mean(metric_list) if metric_list else 0
            summary_stats[f'std_{metric}'] = np.std(metric_list) if metric_list else 0
        for fold_name, fold_metrics in fold_results.items():
            row = {'Subject': subject, 'Fold': fold_name}
            row.update(fold_metrics)
            all_folds_data.append(row)
        avg_row = {'Subject': subject, 'Fold': 'Average'}
        for metric in metrics_to_process:
            avg_row[metric] = summary_stats[f'avg_{metric}']
            avg_row[f'{metric}_std'] = summary_stats[f'std_{metric}']
        all_folds_data.append(avg_row)
    if all_folds_data:
        final_summary_stats = {}
        for metric in metrics_to_process:
            all_metric_values = [fold[metric] for res in results.values() for fold in res.values() if metric in fold]
            final_summary_stats[f'final_avg_{metric}'] = np.mean(all_metric_values) if all_metric_values else 0
            final_summary_stats[f'final_std_{metric}'] = np.std(all_metric_values) if all_metric_values else 0
        all_folds_data.append({})
        final_avg_row = {'Subject': 'OVERALL', 'Fold': 'Final Average'}
        for metric in metrics_to_process:
            final_avg_row[metric] = final_summary_stats[f'final_avg_{metric}']
            final_avg_row[f'{metric}_std'] = final_summary_stats[f'final_std_{metric}']
        all_folds_data.append(final_avg_row)
    df = pd.DataFrame(all_folds_data)
    cols = ['Subject', 'Fold', 'class_acc', 'precision', 'recall', 'f1_score', 'kappa', 'loss', 'time']
    for metric in metrics_to_process:
        if f'{metric}_std' in df.columns:
            cols.append(f'{metric}_std')
    existing_cols = [col for col in cols if col in df.columns]
    df = df[existing_cols]
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
            worksheet = writer.sheets['Results']
            for col in worksheet.columns:
                max_length = 0
                column = col[0].column_letter
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column].width = adjusted_width if adjusted_width < 50 else 50
    except Exception as e:
        pass


def main():
    program_start = time.time()

    SEED = 42
    N_SPLITS = 10
    TOTAL_EPOCHS = 200
    PATIENCE = 30
    BATCH_SIZE = 64
    DATASET_PATH = r"D:\dataset\data_MI_MA_interpolated_v1\MI"
    VIS_DIR = r"D:\dataset\dataset\结果"

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    subject_list = os.listdir(DATASET_PATH)
    all_results = {}
    os.makedirs(VIS_DIR, exist_ok=True)

    for subject in subject_list:
        data_path = os.path.join(DATASET_PATH, subject)
        if not os.path.isfile(data_path) or not data_path.endswith('.npz'):
            continue

        with np.load(data_path) as data:
            eeg_interp, hbo_interp, hbr_interp, labels_onehot = \
                data['eeg_interpolated'], data['hbo_interpolated'], data['hbr_interpolated'], data['label_interpolated']
            eeg_fs, fnirs_fs = data['eeg_fs'].item(), data['fnirs_fs'].item()

        labels_for_stratify = labels_onehot.argmax(axis=0)
        kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

        fold_results = {}
        n_trials = eeg_interp.shape[0]

        for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_trials), labels_for_stratify)):
            fold_start_time = time.time()

            eeg_train, fnirs_train, label_train = build_model_input_from_interpolated(
                eeg_interp[train_idx], hbo_interp[train_idx], hbr_interp[train_idx],
                labels_onehot[:, train_idx], eeg_fs, fnirs_fs)

            eeg_test, fnirs_test, label_test = build_model_input_from_interpolated(
                eeg_interp[test_idx], hbo_interp[test_idx], hbr_interp[test_idx],
                labels_onehot[:, test_idx], eeg_fs, fnirs_fs)

            if eeg_train.shape[0] == 0 or eeg_test.shape[0] == 0:
                continue

            fnirs_train *= 1e6
            fnirs_test *= 1e6

            train_dataset = EEGfNIRSDataset(eeg_train, fnirs_train, label_train)
            test_dataset = EEGfNIRSDataset(eeg_test, fnirs_test, label_test)

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                      pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

            is_first_run = (subject == subject_list[0] and fold == 0)
            model = BiSTFNet(verbose=is_first_run).to(device)

            trainer = Trainer(model, total_epochs=TOTAL_EPOCHS)

            best_acc = 0.0
            patience_counter = 0
            best_model_dir = os.path.join(VIS_DIR, 'best_models')
            os.makedirs(best_model_dir, exist_ok=True)
            best_model_path = os.path.join(best_model_dir, f'best_model_{subject}_fold_{fold + 1}.pt')

            for epoch in range(TOTAL_EPOCHS):
                train_loss = trainer.train_epoch(train_loader, epoch)
                eval_result = trainer.evaluate(test_loader)

                if eval_result['class_acc'] > best_acc:
                    best_acc = eval_result['class_acc']
                    torch.save(model.state_dict(), best_model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        break

                if eval_result['class_acc'] == 1.0:
                    break

            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path, map_location=device))

            final_result = trainer.evaluate(test_loader)

            fold_results[f'fold_{fold + 1}'] = final_result
            fold_results[f'fold_{fold + 1}']['time'] = time.time() - fold_start_time

            del model, trainer, train_loader, test_loader
            torch.cuda.empty_cache()

        all_results[subject] = fold_results
        save_results(all_results, os.path.join(VIS_DIR, 'intermediate_results.xlsx'))

    save_results(all_results, os.path.join(VIS_DIR, 'final_results.xlsx'))


if __name__ == '__main__':
    main()
