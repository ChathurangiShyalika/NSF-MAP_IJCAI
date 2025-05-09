from pypylon import pylon as py
import cv2
import numpy as np


class PylonCameras:
    def __init__(self, num_devices: int):
        tlFactory = py.TlFactory.GetInstance()
        self.devices = tlFactory.EnumerateDevices()
        if len(self.devices) == 0:
            raise py.RuntimeException("No camera present.")
        if len(self.devices) < num_devices:
            raise py.RuntimeException("Not enough cameras present for the number specified.")

        self.cameras = py.InstantCameraArray(num_devices)
        self.converters = []  # List to store converters
        for i, cam in enumerate(self.cameras):
            cam.Attach(tlFactory.CreateDevice(self.devices[i]))
            cv2.namedWindow(f'Cam{i}', cv2.WINDOW_NORMAL)
            # print("Using device ", cam.GetDeviceInfo().GetModelName())

            # Initialize a converter for each camera
            converter = py.ImageFormatConverter()
            converter.OutputPixelFormat = py.PixelType_BGR8packed
            converter.OutputBitAlignment = py.OutputBitAlignment_MsbAligned
            self.converters.append(converter)

    def grab(self, strategy: str):  # Options are: 'OneByOne', 'LatestOnly', 'Upcoming'
        try:
            if strategy == 'OneByOne':
                self.cameras.StartGrabbing(py.GrabStrategy_OneByOne)
            elif strategy == 'LatestOnly':
                self.cameras.StartGrabbing(py.GrabStrategy_LatestImageOnly)
            elif strategy == 'Upcoming':
                self.cameras.StartGrabbing(py.GrabStrategy_UpcomingImage)
        except py.RuntimeException:
            print("Strategy not recognized")

    def grabCount(self, count, cam_index):
        self.cameras[cam_index].StartGrabbingMax(count)

    @staticmethod
    def get_img_size(res):
        return res.GetWidth(), res.GetHeight()

    @staticmethod
    def set_img_size(img, size):
        return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def adjust_white_balance(im):
        avgR = np.average(im[:, :, 2])
        avgG = np.average(im[:, :, 1])
        avgB = np.average(im[:, :, 0])
        scaleR = avgG / avgR
        scaleB = avgG / avgB
        im[:, :, 2] = np.clip(im[:, :, 2] * scaleR, 0, 255).astype(np.uint8)
        im[:, :, 0] = np.clip(im[:, :, 0] * scaleB, 0, 255).astype(np.uint8)
        return im

    def get_image_device(self, res):
        return res.GetCameraContext(), self.cameras[res.GetCameraContext()].GetDeviceInfo().GetModelName()

    @staticmethod
    def display_img(img, idx: int, key: int):
        cv2.imshow(f'Cam{idx}', img)
        cv2.waitKey(key)
