import os
import cv2
import numpy as np
import tensorflow as tf

from config import config
from logger import Logger
#from models import Simple, NASNET, Inception, GAP, YOLO
from models import Inception
from utils import annotator, change_channel, gray_normalizer
from information import Information


def load_model(session, m_type, m_name, logger):
    # load the weights based on best loss
    best_dir = "best_loss"

    # check model dir
    model_path = "models/" + m_name
    path = os.path.join(model_path, best_dir)
    if not os.path.exists(path):
        raise FileNotFoundError

    if m_type == "simple":
        model = Simple(m_name, config, logger)
    elif m_type == "YOLO":
        model = YOLO(m_name, config, logger)
    elif m_type == "GAP":
        model = GAP(m_name, config, logger)
    elif m_type == "NAS":
        model = NASNET(m_name, config, logger)
    elif m_type == "INC":
        model = Inception(m_name, config, logger)
    else:
        raise ValueError

    # load the best saved weights
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.log('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        raise ValueError('There is no best model with given model')

    return model


def rescale(image):
    scale_side = max(image.shape)
    # image width and height are equal to 192
    scale_value = config["input_width"] / scale_side

    # scale down or up the input image
    scaled_image = cv2.resize(image, dsize=None, fx=scale_value, fy=scale_value)

    # convert to numpy array
    scaled_image = np.asarray(scaled_image, dtype=np.uint8)

    # one of pad should be zero
    w_pad = int((config["input_width"] - scaled_image.shape[1]) / 2)
    h_pad = int((config["input_width"] - scaled_image.shape[0]) / 2)

    # create a new image with size of: (config["image_width"], config["image_height"])
    new_image = np.ones((config["input_width"], config["input_height"]), dtype=np.uint8) * 250

    # put the scaled image in the middle of new image
    new_image[h_pad:h_pad + scaled_image.shape[0], w_pad:w_pad + scaled_image.shape[1]] = scaled_image

    return new_image


def upscale_preds(_preds, _shapes):
    # we need to calculate the pads to remove them from predicted labels
    pad_side = np.max(_shapes)
    # image width and height are equal to 384
    downscale_value = config["input_width"] / pad_side

    scaled_height = _shapes[0] * downscale_value
    scaled_width = _shapes[1] * downscale_value

    # one of pad should be zero
    w_pad = (config["input_width"] - scaled_width) / 2
    h_pad = (config["input_width"] - scaled_height) / 2

    # remove the pas from predicted label
    x = _preds[0] - w_pad
    y = _preds[1] - h_pad
    w = _preds[2]

    # calculate the upscale value
    upscale_value = pad_side / config["input_height"]

    # upscale preds
    x = x * upscale_value
    y = y * upscale_value
    w = w * upscale_value

    return x, y, w


def pre_process(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (9, 9), 3)
    median = cv2.medianBlur(gaussian, 3)

    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(median, kernel=kernel, iterations=1)
    return cv2.dilate(erode, kernel=kernel, iterations=1)


def add_label(file, information):
    with open(file, 'a', newline='') as file_writer:
        file_writer.write('{}\n'.format(information))
        file_writer.close()


def run(model, sess, file_in, information, name_exam):
    # load the video or camera
    cap = cv2.VideoCapture(file_in)
    fps = cap.get(cv2.CAP_PROP_FPS)

    path_exams = '/media/marcos/Dados/Projects/Results/Qualificacao/DeepRN/'
    # path_frames = '/media/marcos/Dados/Projects/Results/Qualificacao/DeepRN/Frames/{}'.format(name_exam)

    # try:
    #     os.mkdir(path_frames)
    # except FileExistsError:
    #     pass

    title_label = 'patient,param,frame,center_x,center_y,radius,flash_algorithm,flash_information,color_flash,eye_size,img_mean,img_std,img_median'
    file_label = '/media/marcos/Dados/Projects/Results/Qualificacao/DeepRN/{}_label.csv'.format(name_exam)
    add_label(file_label, title_label)

    file_information = '/media/marcos/Dados/Projects/Datasets/Exams/Information_Exams/{}.log'.format(name_exam)
    if os.path.exists(file_information):
        patient_exam, param_exam = information.get_information_exam(file_information, fps)
    else:
        patient_exam, param_exam = '', ''

    counter, frames, preds = 0, [], []

    counter = 0
    ret, frame = cap.read()
    while ret:
        frames.append(frame)

        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('{}/gray_{}.png'.format(path_frames, counter), frame)

        f_shape = frame.shape
        if frame.shape[0] != 192:
            frame = rescale(frame)

        image = gray_normalizer(frame)
        # cv2.imwrite('{}/gray_normalizer_{}.png'.format(path_frames, counter), image)

        image = change_channel(image, config["input_channel"])
        # cv2.imwrite('{}/change_channel_{}.png'.format(path_frames, counter), image)

        [p] = model.predict(sess, [image])
        x, y, w = upscale_preds(p, f_shape)

        preds.append([x, y, w])

        if os.path.exists(file_information):
            flash_information, color_information = information.get_information_params(counter)
        else:
            flash_information, color_information = '', ''

        inf = '{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(patient_exam, param_exam, counter, x, y, w, 0,
                                                              flash_information, color_information, 0,
                                                              image.mean(), image.std(), np.median(image))

        add_label(file_label, inf)

        ret, frame = cap.read()
        counter += 1

    if len(frames) > 0:
        video_size = frames[0].shape[0:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        video = cv2.VideoWriter('{}/{}.avi'.format(path_exams, name_exam), cv2.VideoWriter_fourcc(*"XVID"), fps, (video_size[1], video_size[0]))

        for i, img in enumerate(frames):
            # labeled_img = annotator((0, 250, 0), img, *preds[i])
            x, y, w = preds[i]
            color = (0, 250, 0)
            labeled_img = cv2.circle(img, (int(x), int(y)), int(w/2), color, 2)
            video.write(labeled_img)

        video.release()

    print("Done {}...".format(name_exam))


# load a the model with the best saved state from file and predict the pupil location
# on the input video. finaly save the video with the predicted pupil on disk
def main(m_type, m_name, logger, information, video_path=None):
    with tf.Session() as sess:

        model = load_model(sess, m_type, m_name, logger)

        work_files = []
        with open('files.txt') as file_read:
            for line in file_read.readlines():
                work_files.append(line.replace('\n', '').replace('.avi', '.mp4'))

        path_exams = '/media/marcos/Dados/Projects/Results/PupilLocator/Exams'
        files_exists = [x.replace('.avi', '.mp4') for x in os.listdir(path_exams)]

        if os.path.exists(video_path):
            files = [x for x in os.listdir(video_path) if ('.mp4' in x) and (x in work_files) and (x not in files_exists)]

        files = ['benchmark_final.avi']

        for file in files:
            file_in = '{}/{}'.format(video_path, file)
            name_exam = file.replace('.avi', '')
            run(model, sess, file_in, information, name_exam)


if __name__ == "__main__":
    model_name = "3A4Bh-Ref25"
    model_type = "INC"
    video_path = '/media/marcos/Dados/Projects/Datasets/Exams'

    # initial a logger
    logger = Logger(model_type, model_name, "", config, dir="models/")
    logger.log("Start inferring model...")

    information = Information()

    main(model_type, model_name, logger, information, video_path)
