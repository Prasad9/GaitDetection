import os
import glob
import random
import numpy as np
import cv2

from KeypointsForVideo import KeypointsForVideo


class DataGeneration:
    def __init__(self, params):
        self._folder_path = os.path.abspath(params['VIDEO_FOLDER_PATH'])
        self._data_save_path = os.path.abspath(params['DATA_SAVE_PATH'])
        self._input_size = params['INPUT_SIZE']

        self._keypoints_for_video = KeypointsForVideo(params)

    def create_data(self):
        video_file_paths = glob.glob(self._folder_path + '/*.mp4')
        os.makedirs(self._data_save_path, exist_ok=True)

        for label_index, video_file_path in enumerate(video_file_paths):
            file_index = 0
            frames = self._get_frames_in_video(video_file_path)
            keypoints_with_scores, frames = self._extract_time_series_data(frames)
            # keypoints = np.reshape(keypoints[:, :, :2], [-1, 17 * 2])
            keypoints = keypoints_with_scores[:, :, :2]
            print(keypoints[0])
            self._save_data(keypoints, label_index, file_index)
            print('Base len:', len(keypoints), keypoints.shape)

            # Fast videos
            for counter in range(random.randint(1, 4)):
                file_index += 1
                fast_keypoints = self._faster_video(keypoints)
                self._save_data(fast_keypoints, label_index, file_index)

            # Slow videos
            keypoints_1d = np.reshape(keypoints, [-1, 17 * 2])
            for counter in range(random.randint(1, 4)):
                file_index += 1
                slow_keypoints = self._slower_video(keypoints_1d)
                self._save_data(slow_keypoints, label_index, file_index)

            # Left hand stagnant videos
            left_positions_at = [5, 7, 9]
            for counter in range(random.randint(1, 2)):
                file_index += 1
                left_stag_keypoints = self._hand_stagnant(keypoints, left_positions_at)
                self._save_data(left_stag_keypoints, label_index, file_index)

            # Right hand stagnant videos
            right_positions_at = [6, 8, 10]
            for counter in range(random.randint(1, 2)):
                file_index += 1
                right_stag_keypoints = self._hand_stagnant(keypoints, right_positions_at)
                self._save_data(right_stag_keypoints, label_index, file_index)

            # Both hand stagnant videos
            both_positions_at = [5, 6, 7, 8, 9, 10]
            for counter in range(random.randint(1, 2)):
                file_index += 1
                both_stag_keypoints = self._hand_stagnant(keypoints, both_positions_at)
                self._save_data(both_stag_keypoints, label_index, file_index)

            # Top posture videos
            for counter in range(random.randint(2, 5)):
                file_index += 1
                top_posture_keypoints = self._top_posture_only(keypoints)
                self._save_data(top_posture_keypoints, label_index, file_index)

            # Random noise videos
            for counter in range(random.randint(1, 5)):
                file_index += 1
                random_noise_keypoints = self._random_noise_positions(keypoints)
                self._save_data(random_noise_keypoints, label_index, file_index)

            # Stop walking videos
            keypoints_1d = np.reshape(keypoints, [-1, 17 * 2])
            for counter in range(random.randint(1, 5)):
                file_index += 1
                stop_walking_keypoints = self._stop_walking_video(keypoints_1d)
                self._save_data(stop_walking_keypoints, label_index, file_index)

            # Walking left videos
            for counter in range(random.randint(1, 2)):
                file_index += 1
                walk_left_keypoints = self._walk_left_video(keypoints)
                self._save_data(walk_left_keypoints, label_index, file_index)

            # Walking right videos
            for counter in range(random.randint(1, 2)):
                file_index += 1
                walk_right_keypoints = self._walk_right_video(keypoints)
                self._save_data(walk_right_keypoints, label_index, file_index)

    def _save_data(self, keypoints, label_index, file_index):
        file_save_path = os.path.join(self._data_save_path, '{}_{}.npy'.format(label_index, file_index))
        if len(keypoints.shape) == 3:
            keypoints = np.reshape(keypoints, [-1, 17 * 2])
        np.save(file_save_path, keypoints)

    def _faster_video(self, keypoints):
        n_keypoints = len(keypoints)
        n_samples = random.randint(n_keypoints // 3, (3 * n_keypoints) // 4)
        samples = list(range(len(keypoints)))
        indexes = random.sample(samples, n_samples)
        indexes.sort()
        new_keypoints = keypoints[indexes]
        print('Fast len:', len(indexes), new_keypoints.shape)
        return new_keypoints

    def _slower_video(self, keypoints, min_interval=1, max_interval=4):
        n_keypoints = len(keypoints)
        interpolate_at = []
        for index in range(n_keypoints):
            intervals = [index + random.random() for _ in range(random.randint(min_interval, max_interval))]
            intervals.sort()
            interpolate_at.extend(intervals)

        x_val = np.arange(n_keypoints)
        interp_vals_arr = []
        for col in range(keypoints.shape[1]):
            col_keypoints = keypoints[:, col]
            interp_vals = np.interp(interpolate_at, x_val, col_keypoints)
            interp_vals = np.expand_dims(interp_vals, axis=1)
            interp_vals_arr.append(interp_vals)
        interp_vals_arr = np.concatenate(interp_vals_arr, axis=1)
        # print('Slow len:', len(interpolate_at), interp_vals_arr.shape)
        return interp_vals_arr

    def _hand_stagnant(self, keypoints, positions):
        keypoints_copy = keypoints.copy()
        for position in positions:
            keypoints_copy = self._create_stagnant_distribution(keypoints_copy, position)
        return keypoints_copy

    def _create_stagnant_distribution(self, keypoints, index):
        x_keypoints = self._create_1d_stagnant_distribution(keypoints[:, index, 1])
        y_keypoints = self._create_1d_stagnant_distribution(keypoints[:, index, 0])
        keypoints[:, index, 1] = x_keypoints
        keypoints[:, index, 0] = y_keypoints
        return keypoints

    def _create_1d_stagnant_distribution(self, keypoints):
        new_keypoints = []
        batch = 10
        for offset in range(0, len(keypoints), batch):
            keypoints_batch = keypoints[offset: offset + batch]
            k_mean = np.mean(keypoints_batch)
            k_std = np.std(keypoints_batch)

            min_range = k_mean - 0.2 * k_std
            max_range = k_mean + 0.2 * k_std
            new_keypoints_batch = [random.uniform(min_range, max_range) for _ in range(batch)]
            new_keypoints.extend(new_keypoints_batch)
        return new_keypoints

    def _top_posture_only(self, keypoints):
        keypoints_copy = keypoints.copy()
        min_x = max(0.55, np.min(keypoints_copy[:, 11:, 0]))
        var_to_add = 1.0 - min_x - random.uniform(-0.05, 0.05)
        keypoints_copy[:, :, 0] = np.minimum(1.0, var_to_add + keypoints_copy[:, :, 0])
        return keypoints_copy

    def _random_noise_positions(self, keypoints):
        keypoints_copy = keypoints.copy()
        keypoint_shape = keypoints[0].shape
        for index in range(len(keypoints)):
            if random.choice([True, False, False, False]):
                noise = 0.25 * np.random.randn(*keypoint_shape)
                buffer = keypoints_copy[index] + noise
                buffer = np.minimum(1.0, np.maximum(0.0, buffer))
                keypoints_copy[index] = buffer
        return keypoints_copy

    def _stop_walking_video(self, keypoints):
        stop_times = random.randint(3, 5)
        stop_intervals = []
        batch_size = len(keypoints) // stop_times
        for offset in range(0, len(keypoints), batch_size):
            stop_at = random.randint(offset + 1, offset + batch_size - 1)
            if stop_at < len(keypoints):
                stop_intervals.append(stop_at)

        new_keypoints = np.empty((1,) + keypoints[0].shape)
        prev_at = 0
        for stop_at in stop_intervals:
            new_keypoints = np.concatenate((new_keypoints, keypoints[prev_at: stop_at]))
            stop_walk_keypoints = self._slower_video(keypoints[stop_at: stop_at + 2], min_interval=8, max_interval=20)
            new_keypoints = np.concatenate((new_keypoints, stop_walk_keypoints))
            prev_at = stop_at
        new_keypoints = np.concatenate((new_keypoints, keypoints[prev_at:]))
        print('Stop walking video', new_keypoints.shape)
        return new_keypoints

    def _walk_left_video(self, keypoints):
        new_keypoints = keypoints.copy()
        start_y = random.uniform(0.75, 0.90)
        end_y = random.uniform(0.10, 0.25)
        diff_at = (start_y - end_y) / len(keypoints)
        current_y = start_y
        for index in range(len(keypoints)):
            k_mean = np.mean(keypoints[index, :, 1])
            new_keypoints[index, :, 1] = new_keypoints[index, :, 1] - k_mean + current_y
            current_y -= diff_at
        print('Walk left shape:', new_keypoints.shape)
        return new_keypoints

    def _walk_right_video(self, keypoints):
        new_keypoints = keypoints.copy()
        start_y = random.uniform(0.10, 0.25)
        end_y = random.uniform(0.75, 0.90)
        diff_at = (end_y - start_y) / len(keypoints)
        current_y = start_y
        for index in range(len(keypoints)):
            k_mean = np.mean(keypoints[index, :, 1])
            new_keypoints[index, :, 1] = new_keypoints[index, :, 1] - k_mean + current_y
            current_y += diff_at
        print('Walk right shape:', new_keypoints.shape)
        return new_keypoints

    def _get_frames_in_video(self, file_path):
        vidcap = cv2.VideoCapture(file_path)
        img_arr = []
        success, image = vidcap.read()
        while success:
            success, image = vidcap.read()
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
                img_arr.append(image)
        return img_arr

    def _extract_time_series_data(self, frames):
        num_frames = len(frames)
        # Load the input image.
        image_height, image_width, _ = frames[0].shape
        crop_region = self._keypoints_for_video.init_crop_region(image_height, image_width)

        output_images = []
        frames_to_delete = []
        for frame_idx in range(num_frames):
            if frames[frame_idx] is not None:
                keypoints_with_scores = self._keypoints_for_video.run_inference(frames[frame_idx],
                                                                                crop_region,
                                                                                crop_size=[self._input_size,
                                                                                           self._input_size])
                diff_x = np.max(keypoints_with_scores[:, :, 0]) - np.min(keypoints_with_scores[:, :, 0])
                diff_y = np.max(keypoints_with_scores[:, :, 1]) - np.min(keypoints_with_scores[:, :, 1])
                if diff_x < 0.02 or diff_y < 0.02:
                    frames_to_delete.append(frame_idx)
                    continue
                output_images.append(keypoints_with_scores)

        output_images = np.concatenate(output_images, axis=0)
        output_images = np.squeeze(output_images)
        print('Totally skipped frames:', len(frames_to_delete))
        frames_to_delete.reverse()
        for frame_idx in frames_to_delete:
            del frames[frame_idx]
        return output_images, frames


if __name__ == '__main__':
    params = {
        'INPUT_SIZE': 256,                                    # Crop size for TFLite model to work on
        'TFLITE_MODEL_PATH': './Model/model.tflite',          # File path of Human Pose Estimation model
        'VIDEO_FOLDER_PATH': './Data/Videos',                 # Folder where videos are present
        'DATA_SAVE_PATH': './Data/TimeSeriesData'             # Folder where time series data needs to be saved
    }
    data_generation = DataGeneration(params)
    data_generation.create_data()
