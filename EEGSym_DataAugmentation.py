import tensorflow as tf
import numpy as np


def preprocessing_function(augmentation=True):
    """Custom Data Augmentation for EEGSym.

        Parameters
        ----------
        augmentation : Bool
            If the augmentation is performed to the input.

        Returns
        -------
        data_augmentation : function
            Data augmentation performed to each trial
    """

    def data_augmentation(trial):
        """Custom Data Augmentation for EEGSym.

            Parameters
            ----------
            trial : tf.tensor
                Input of the

            Returns
            -------
            data_augmentation : keras.models.Model
                Data augmentation performed to each trial
        """

        samples, ncha, _ = trial.shape

        augmentations = dict()
        augmentations["patch_perturbation"] = 0
        augmentations["random_shift"] = 0
        augmentations["hemisphere_perturbation"] = 0
        augmentations["no_augmentation"] = 0

        selectionables = ["patch_perturbation", "random_shift",
                          "hemisphere_perturbation", "no_augmentation"]
        probabilities = None

        if augmentation:
            selection = np.random.choice(selectionables, p=probabilities)
            augmentations[selection] = 1

            method = np.random.choice((0, 2))
            std = 'self'
            # elif data_augmentation == 1:  # Random shift
            for _ in range(augmentations["random_shift"]):  # Random shift
                # Select position where to erase that timeframe
                position = 0
                if position == 0:
                    samples_shifted = np.random.randint(low=1, high=int(
                        samples * 0.5 / 3))
                else:
                    samples_shifted = np.random.randint(low=1, high=int(
                        samples * 0.1 / 3))

                if method == 0:
                    shifted_samples = np.zeros((samples_shifted, ncha, 1))
                else:
                    if std == 'self':
                        std_applied = np.std(trial)
                    else:
                        std_applied = std
                    center = 0
                    shifted_samples = np.random.normal(center, std_applied,
                                                       (samples_shifted, ncha,
                                                        1))
                if position == 0:
                    trial = np.concatenate((shifted_samples, trial),
                                           axis=0)[:samples]
                else:
                    trial = np.concatenate((trial, shifted_samples),
                                           axis=0)[samples_shifted:]

            for _ in range(
                    augmentations["patch_perturbation"]):  # Patch perturbation
                channels_affected = np.random.randint(low=1, high=ncha - 1)
                pct_max = 1
                pct_min = 0.2
                pct_erased = np.random.uniform(low=pct_min, high=pct_max)
                # Select time to be erased acording to pct_erased
                # samples_erased = np.min((int(samples*ncha*pct_erased//channels_affected), samples))#np.random.randint(low=1, high=samples//3)
                samples_erased = int(samples * pct_erased)
                # Select position where to erase that timeframe
                if samples_erased != samples:
                    samples_idx = np.arange(samples_erased) + np.random.randint(
                        samples - samples_erased)
                else:
                    samples_idx = np.arange(samples_erased)
                # Select indexes to erase (always keep at least a channel)
                channel_idx = np.random.permutation(np.arange(ncha))[
                              :channels_affected]
                channel_idx.sort()
                for channel in channel_idx:
                    if method == 0:
                        trial[samples_idx, channel] = 0
                    else:
                        if std == 'self':
                            std_applied = np.std(trial[:, channel]) \
                                          * np.random.uniform(low=0.01, high=2)
                        else:
                            std_applied = std
                        center = 0
                        trial[samples_idx, channel] += \
                            np.random.normal(center, std_applied,
                                             trial[samples_idx, channel,
                                             :].shape)
                        # Standarize the channel again after the change
                        temp_trial_ch_mean = np.mean(trial[:, channel], axis=0)
                        temp_trial_ch_std = np.std(trial[:, channel], axis=0)
                        trial[:, channel] = (trial[:,
                                             channel] - temp_trial_ch_mean) / temp_trial_ch_std

            for _ in range(augmentations["hemisphere_perturbation"]):
                # Select side to mix/change for noise
                left_right = np.random.choice((0, 1))
                if method == 0:
                    if left_right == 1:
                        channel_idx = np.arange(ncha)[:int((ncha / 2) - 1)]
                        channel_mix = np.random.permutation(channel_idx.copy())
                    else:
                        channel_idx = np.arange(ncha)[-int((ncha / 2) - 1):]
                        channel_mix = np.random.permutation(channel_idx.copy())
                    temp_trial = trial.copy()
                    for channel, channel_mixed in zip(channel_idx, channel_mix):
                        temp_trial[:, channel] = trial[:, channel_mixed]
                    trial = temp_trial
                else:
                    if left_right == 1:
                        channel_idx = np.arange(ncha)[:int((ncha / 2) - 1)]
                    else:
                        channel_idx = np.arange(ncha)[-int((ncha / 2) - 1):]
                    for channel in channel_idx:
                        trial[:, channel] = np.random.normal(0, 1,
                                                             trial[:,
                                                             channel].shape)

        return trial

    return data_augmentation


def trial_iterator(X, y, batch_size=32, shuffle=True, augmentation=True):
    """Custom trial iterator to pretrain EEGSym.

        Parameters
        ----------
        X : tf.tensor
            Input tensor of  EEG features.
        y : tf.tensor
            Input tensor of  labels.
        batch_size : int
            Number of features in each batch.
        shuffle : Bool
            If the features are shuffled at each training epoch.
        augmentation : Bool
            If the augmentation is performed to the input.

        Returns
        -------
        trial_iterator : tf.keras.preprocessing.image.NumpyArrayIterator
            Iterator used to train the model.
    """

    trial_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocessing_function(
            augmentation=augmentation))

    trial_iterator = tf.keras.preprocessing.image.NumpyArrayIterator(
        X, y, trial_data_generator, batch_size=batch_size, shuffle=shuffle,
        sample_weight=None,
        seed=None, data_format=None, save_to_dir=None, save_prefix='',
        save_format='png', subset=None, dtype=None
    )
    return trial_iterator
