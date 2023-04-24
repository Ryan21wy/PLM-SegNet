import torch
import numpy as np
import os
import cv2
import time

from Architecture.PLM_SegNet_Architecture import UNet, UNet1, UNet_4l

from Main.utils import compute_gaussian

import PIL.Image as Image


def byj_position_detection(src):
    '''
    this function is used for preprocessing the palm-leaf manuscript images.
    the position of the manuscript is firstly detected, and the image of 
    palm-leaf manuscript is then cropped to balance the ratio of the subject,
    namely the palm-leaf manuscript and the background.

    Parameters
    ----------
    src : ndarray
        a 3-D array indicating the image of palm-leaf manuscript.
        the former two dimension jointly indicates the size of the image.
        the last one dimension indicates the channels of the image, for a
        digital image, it is 3 channels that the last one dimension is, which 
        are respectively red channel, green channel and blue channel.

    Returns
    -------
    [row_min, row_max, col_min, col_max] : list
        row_min : the minimum coordinate value of row
        row_max : the maximum coordinate value of row
        col_min : the minimum coordinate value of cloumn
        col_max : the maximum coordinate value of cloumn

    '''
    # Canny Edge Extraction Parameters
    init_threhold = 80
    canny_up = 130
    canny_down = 80
    kernel_size = (6, 4)
    
    # Open the pic and make it a matrix
    # src = np.array(Image.open(file_path).convert('RGB'))
    if src.shape[0] > src.shape[1]:
        src = np.transpose(src,(1,0,2))
    
    length = src.shape[1] #big
    width = src.shape[0]
    # Make use of OpenCV to transform RGB mode to HSV mode
    hsv_ori = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
    
    # blank = np.zeros_like(ori, ori.dtype)
    b, g, r = cv2.split(hsv_ori)
    r[r < init_threhold] = 255
    r = cv2.GaussianBlur(r, (5, 5), 0)
    
    # -- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(r, canny_down, canny_up)
    KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    edges = cv2.dilate(edges, KERNEL, iterations=1)
    edges = cv2.erode(edges, KERNEL, iterations=1)
    
    # -- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((c, cv2.isContourConvex(c), cv2.contourArea(c)))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0][0]
    
    # Get the maximun coordinate of the contour
    max_contour_coordinate = max_contour[:, 0, :]
    col_max = np.max(max_contour_coordinate[:, 0]) + 100
    col_min = np.min(max_contour_coordinate[:, 0]) - 100
    row_max = np.max(max_contour_coordinate[:, 1]) + 100
    row_min = np.min(max_contour_coordinate[:, 1]) - 100
    
    # Threhold Setting to avoid being out of range
    if col_max > length:
        col_max = length - 1
    if row_max > width:
        row_max = width - 1
    if col_min < 0:
        col_min = 0
    if row_min < 0:
        row_min = 0
    
    return [row_min, row_max, col_min, col_max]


def prediction(file_path, model_path, save_path, tta=False, edge_treatment=False):
    '''
        this function is used for segment the palm-leaf manuscript.

        Parameters
        ----------
        file_path :
            the path of palm-leaf manuscript images.
        model_path :
            the path of PLM-SegNet model.
        save_path :
            the path for saving the segmented palm-leaf manuscript images.
        tta : bool
            Test-Time Augmentation, using it may gives more accurate results, but it also takes more time.
        edge_treatment : bool
            treat the inaccurate predicted edge of image patch,
            using it may gives more accurate results, but it also takes more time.
        '''
    print("Model Performance Testing...")
    print("Model loading...")
    device = torch.device("cuda:0")  # if using cpu, change 'cuda:0' to 'cpu'
    model = UNet(3, 2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loading success!")
    print("Prediction start...")

    if edge_treatment:
        g_map = compute_gaussian((512, 400), 1./4)

    t1 = time.time()
    dirs = os.listdir(file_path)
    for ind, dir_ in enumerate(dirs):
        # open an image
        print("Background removal begins! Now :{}/{}".format(ind, len(dirs)))
        print("Trying to open an image and express it in form of ndarray...")
        data_x1 = cv2.imdecode(np.fromfile(file_path + "\\" + dir_ + "\\img.png", dtype=np.uint8), -1)
        data_x1 = cv2.cvtColor(data_x1, cv2.COLOR_BGR2RGB)

        print("Image open success! Now confirm the coordinate of image...")
        [row_min, row_max, col_min, col_max] = byj_position_detection(data_x1)

        print("Image preprocessing...")
        data_x = np.transpose(data_x1, (2, 0, 1))
        if data_x.shape[1] > data_x.shape[2]:
            data_x = np.transpose(data_x, (0, 2, 1))

        length = data_x.shape[2]
        width = data_x.shape[1]

        max_col_factor = length // 400 + 1

        # after position detection, the distance of the position coordinate
        row_length = row_max - row_min
        col_length = col_max - col_min

        # height/width is multiples of 512/400
        row_factor = row_length // 512 + 1
        col_factor = col_length // 400 + 1
        row = row_factor * 512
        col = col_factor * 400

        # supply
        supplyment_row = row - row_length
        supplyment_col = col - col_length

        nrow_min = row_min - supplyment_row // 2
        nrow_max = row_max + supplyment_row - supplyment_row // 2
        ncol_min = col_min - supplyment_col // 2
        ncol_max = col_max + supplyment_col - supplyment_col // 2

        if col_factor == max_col_factor:
            ncol_min = 0
            ncol_max = length
        elif ncol_min < 0:
            ncol_min = 0
            ncol_max = 400 * col_factor
        elif ncol_max > length:
            ncol_max = length
            ncol_min = length - 400 * col_factor

        data_x = data_x.astype(np.float32)

        data_xs = data_x[:, nrow_min: nrow_max, ncol_min: ncol_max]
        blank = np.zeros((2, data_xs.shape[1], data_xs.shape[2]))

        bar = 0
        if edge_treatment:
            row_factor = row_factor * 2 - 1
            col_factor = col_factor * 2 - 1
            s = 2
        else:
            s = 1
        total = row_factor * col_factor
        for i in range(row_factor):
            for ii in range(col_factor):
                bar += 1
                test_x = data_xs[:, (512 // s * i): (512 // s * (i + s)), (400 // s * ii): (400 // s * (ii + s))]
                model.eval()
                with torch.no_grad():
                    if tta:
                        test_x1 = test_x.reshape((1, 3, 512, 400))
                        test_x2 = np.flip(test_x, 1).reshape((1, 3, 512, 400))
                        test_x3 = np.flip(test_x, 2).reshape((1, 3, 512, 400))
                        test_x4 = np.flip(test_x, (1, 2)).reshape((1, 3, 512, 400))
                        test_ = np.concatenate([test_x1, test_x2, test_x3, test_x4], axis=0)
                    else:
                        test_ = test_x.reshape((1, 3, 512, 400))
                    # test_ = test_ / test_.max()
                    test_ = torch.from_numpy(test_)
                    test_ = test_.to(device)
                    pred_y = model(test_)

                    py = pred_y.cpu().numpy()
                    if tta:
                        py1 = py[0].reshape(2, 512, 400)
                        py2 = np.flip(py[1].reshape(2, 512, 400), 1)
                        py3 = np.flip(py[2].reshape(2, 512, 400), 2)
                        py4 = np.flip(py[3].reshape(2, 512, 400), (1, 2))
                        py = py1 + py2 + py3 + py4
                    else:
                        py = py.reshape(2, 512, 400)

                    if edge_treatment:
                        py = py * g_map
                    blank[:, (512 // s * i): (512 // s * (i + s)), (400 // s * ii): (400 // s * (ii + s))] += py

                    per = bar / total
                    print("\rCurrent prediction progress： %.f%s" % (per * 100, '%'), flush=True, end='')

        blank = np.argmax(blank, axis=0).astype(np.uint8)
        contours, _ = cv2.findContours(blank, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour[:,0,:]) < 150:
                c_edges = contour[:,0,:]
                cmin = np.min(c_edges[:,0]) - 1
                cmax = np.max(c_edges[:,0]) + 1
                rmin = np.min(c_edges[:,1]) - 1
                rmax = np.max(c_edges[:,1]) + 1
                if cmax >= data_xs.shape[2]:
                    cmax = data_xs.shape[2] - 1
                if rmax >= data_xs.shape[1]:
                    rmax = data_xs.shape[1] - 1
                if cmin < 0:
                    cmin = 0
                if rmin < 0:
                    rmin = 0

                byj_or_br = [blank[rmin, cmin], blank[rmin, cmax], blank[rmax, cmin], blank[rmax, cmax]]
                if len(np.where(np.array(byj_or_br) == 1)[0]) >= 2:
                    blank[rmin: rmax, cmin: cmax] = 1
                else:
                    blank[rmin: rmax, cmin: cmax] = 0
        t2 = time.time()
        print("Background removal is done!")
        print("Total time cost: %.4f s" % (t2 - t1))

        print("Image saving...")
        if data_x1.shape[0] > data_x1.shape[1]:
            data_x1 = np.transpose(data_x1, (1, 0, 2))
        new_blank = np.zeros((data_x1.shape[0], data_x1.shape[1]))
        new_blank[nrow_min: nrow_max, ncol_min: ncol_max] = blank
        data_x1[new_blank == 0] = 255  # white background
        new_result = data_x1.astype('uint8')
        new_result = cv2.cvtColor(new_result, cv2.COLOR_RGB2BGR)
        os.chdir(save_path)
        cv2.imwrite("{}_filename.png".format(ind), new_result)
        print("ok!")


if __name__ == '__main__':
    image_path = "PLM images path"
    model_path = "PLM-SegNet model path"
    save_path = "Segmented images save path"
    prediction(image_path,
               model_path,
               save_path,
               tta=False,
               edge_treatment=False)