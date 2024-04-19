import os
import numpy as np
from numpy.lib import recfunctions as rfn
import cv2
import numba as nb
import torch
from torch.nn.functional import pad
from src.io.psee_loader import PSEELoader
import math


class Prophesee:
    def __init__(self, root, height, width, stride=10000, transform=None, mode="training"):
        """
        Creates an iterator over the Prophesee object recognition dataset.

        :param root: path to dataset root
        :param height: height of dataset image
        :param width: width of dataset image
        :stride: stride of events
        :param mode: "training", "testing" or "validation"
        """
        if mode == "training":
            mode = "train"
        elif mode == "validation":
            mode = "val"
        elif mode == "testing":
            mode = "test"

        self.root = root
        self.mode = mode
        self.width = width
        self.height = height
        self.stride = stride
        self.transform = transform
        self.max_nr_bbox = 60

        filelist_path = os.path.join(self.root, self.mode)

        self.event_files, self.label_files = self.load_data_files(filelist_path, self.root, self.mode)

        assert len(self.event_files) == len(self.label_files)
        self.num_events = [PSEELoader(f).event_count() for f in self.event_files] # number of events in each file
        self.num_sample = [math.ceil(n / self.stride) for n in self.num_events] # number of samples in each file
        
        self.event_files_idx = np.concatenate([np.repeat(i, int(n)) for i, n in enumerate(self.num_sample)])

        self.nr_samples = len(self.event_files)


    def __len__(self):
        return len(self.event_files_idx)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from split .npy files
        :param idx:
        :return: events: (x, y, t, p)
                 boxes: (N, 4), which is consist of (x_min, y_min, x_max, y_max)
        """
        # load events and label file
        file_idx = self.event_files_idx[idx]
        bbox_file = os.path.join(self.label_files[file_idx])
        event_file = os.path.join(self.event_files[file_idx])

        labels_np = np.load(bbox_file)
        video = PSEELoader(event_file)
        
        sample_idx = sum(self.event_files_idx[:idx]==file_idx)
        video._file.seek(video._start+self.stride*sample_idx*video._ev_size)
        events_np = video.load_n_events(self.stride)  
        
        # get up and low bound of timestamp
        t_wall = [events_np['t'][0], events_np['t'][-1]]

        mask = (labels_np['ts'] >= t_wall[0]) * (labels_np['ts'] <= t_wall[1])  # filter boxes which are out of bounds
        labels = rfn.structured_to_unstructured(labels_np[mask])[:, [1, 2, 3, 4, 5]] # (x, y, w, h, class_id)

        mask = (events_np['x'] < 1280) * (events_np['y'] < 720)  # filter events which are out of bounds
        events = events_np[mask] if self.transform is None else self.transform(events_np[mask])
        events['x'] = events['x']/1280*512
        events['y'] = events['y']/1280*512
        # events = rfn.structured_to_unstructured(events_np[mask])[:, [1, 2, 0, 3]]  # (x, y, t, p)
        # const_size_events = np.ones([self.stride, 4]).astype(np.float32) * -1
        # const_size_events[:events.shape[0], :] = events
        
        # downsample and resolution=1280x720 -> resolution=512x512
        # events = self.downsample_event_stream(events)
        
        labels = self.cropToFrame(labels)
        labels = self.filter_boxes(labels, 60, 20)  # filter small boxes
        labels[:, 2] += labels[:, 0]
        labels[:, 3] += labels[:, 1]  # [x1, y1, x2, y2, class]
        # labels[:, 0] /= 1280
        # labels[:, 1] /= 720
        # labels[:, 2] /= 1280
        # labels[:, 3] /= 720

        # labels[:, :4] *= 512
        labels[:, 2] -= labels[:, 0]
        labels[:, 3] -= labels[:, 1]

        labels[:, 2:-1] += labels[:, :2]  # [x_min, y_min, x_max, y_max, class_id]
        boxes = labels.astype(np.float32)
        const_size_box = np.ones([self.max_nr_bbox, 5]).astype(np.float32) * -1
        const_size_box[:boxes.shape[0], :] = boxes

        return events, const_size_box

    def downsample_event_stream(self, events):
        events[:, 0] = events[:, 0] / 1280 * 512  # x
        events[:, 1] = events[:, 1] / 720 * 512  # y
        delta_t = events[-1, 2] - events[0, 2]
        events[:, 2] = 4 * (events[:, 2] - events[0, 2]) / delta_t

        _, ev_idx = np.unique(events[:, :2], axis=0, return_index=True)
        downsample_events = events[ev_idx]
        ev = downsample_events[np.argsort(downsample_events[:, 2])]
        return ev

    def cropToFrame(self, np_bbox):
        """Checks if bounding boxes are inside frame. If not crop to border"""
        boxes = []
        for box in np_bbox:
            # if box[2] > 1280 or box[3] > 800:  # filter error label
            if box[2] > 1280:
                continue

            if box[0] < 0:  # x < 0 & w > 0
                box[2] += box[0]
                box[0] = 0
            if box[1] < 0:  # y < 0 & h > 0
                box[3] += box[1]
                box[1] = 0
            if box[0] + box[2] > self.width:  # x+w>1280
                box[2] = self.width - box[0]
            if box[1] + box[3] > self.height:  # y+h>720
                box[3] = self.height - box[1]

            if box[2] > 0 and box[3] > 0 and box[0] < self.width and box[1] <= self.height:
                boxes.append(box)
        boxes = np.array(boxes).reshape(-1, 5)
        return boxes

    def filter_boxes(self, boxes, min_box_diag=60, min_box_side=20):
        """Filters boxes according to the paper rule.
        To note: the default represents our threshold when evaluating GEN4 resolution (1280x720)
        To note: we assume the initial time of the video is always 0
        :param boxes: (np.ndarray)
                     structured box array with fields ['t','x','y','w','h','class_id','track_id','class_confidence']
                     (example BBOX_DTYPE is provided in src/box_loading.py)
        Returns:
            boxes: filtered boxes
        """
        width = boxes[:, 2]
        height = boxes[:, 3]
        diag_square = width ** 2 + height ** 2
        mask = (diag_square >= min_box_diag ** 2) * (width >= min_box_side) * (height >= min_box_side)
        return boxes[mask]

    @staticmethod
    def load_data_files(filelist_path, root, mode):
        event_files = [os.path.join(root, mode, fn) for fn in sorted(os.listdir(filelist_path)) if fn.endswith('dat')]
        label_files = [os.path.join(root, mode, fn) for fn in sorted(os.listdir(filelist_path)) if fn.endswith('npy')]

        return event_files, label_files



if __name__ == "__main__":
    root = "/beegfs/ws/0/scma493c-dvs/Mini_Automotive_Detection_Dataset"
    height = 720
    width = 1280
    # resize = 512
    stride = 100000
    import tonic
    transform = tonic.transforms.Compose([
		tonic.transforms.DropEvent(p=0.1),
	])
    dataset = Prophesee(
        root,
        height,
        width,
        stride,
        transform,
        mode="training",
    )
    first = dataset[0]
    print(first)
    print("A")
