from __future__ import absolute_import
from Code.faster_rcnn.model.utils.net_utils import vis_detections
from Code.faster_rcnn.model.faster_rcnn.resnet import resnet
from Code.faster_rcnn.model.utils.blob import im_list_to_blob
from Code.faster_rcnn.model.roi_layers import nms
from Code.faster_rcnn.model.rpn.bbox_transform import clip_boxes, bbox_transform_inv
from Code.faster_rcnn.model.utils.config import cfg, cfg_from_file, cfg_from_list
from imageio.v2 import imread
from torch.autograd import Variable
from PIL import Image
import json
import argparse
import numpy as np
import cv2
import os
import base64
import torch
import sys
sys.path.append('../../')


def process(source_path, save_path, path):
    data = []

    with open(source_path, 'r', encoding='utf-8') as file:
        next(file)
        for line in file:
            subdata = {}
            line = line.strip('\n').split('\t')
            subdata['image'] = '../../data/' + path + line[2]
            subdata['target'] = line[4]
            subdata['text'] = line[4] + " [SEP] " + \
                line[3] + "[SEP]"  # line[3] + " [SEP] " +
            #subdata['caption'] = subdata['caption'].replace("$T$", line[4])
            subdata['label'] = line[1]
            data.append(subdata.copy())

    with open(save_path, 'w+', encoding='utf-8') as f:
        json.dump(data, f)


def process_target(source_path, save_path, path):
    data = []
    classes, fasterRCNN = load_model()

    with open(source_path, 'r', encoding='utf-8') as file:
        next(file)
        for line in file:
            subdata = {}
            line = line.strip('\n').split('\t')
            subdata['image'] = '../../data/' + path + line[2]
            subdata['max_conf'], subdata['feature'] = get_detections_from_im(
                fasterRCNN, classes, subdata['image'])
            subdata['target'] = line[4]
            subdata['text'] = line[3]  # line[3] + " [SEP] " +
            #subdata['caption'] = subdata['caption'].replace("$T$", line[4])
            subdata['label'] = line[1]
            data.append(subdata.copy())

    with open(save_path, 'w+', encoding='utf-8') as f:
        json.dump(data, f)


def process_feature(source_path, save_path):
    classes, fasterRCNN = load_model()

    for _, _, file in os.walk(source_path):
        for image in file:
            save_path = os.path.join(save_path, image.replace('.jpg', '.json'))
            if os.path.exists(save_path):
                continue

            subdata = {}
            image_path = os.path.join(source_path, image)
            subdata['max_conf'], subdata['feature'] = get_detections_from_im(
                fasterRCNN, classes, image_path, source_path)

            with open(save_path, 'w+', encoding='utf-8') as f:
                json.dump(subdata.copy(), f)


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def get_detections_from_im(fasterRCNN, classes, im_file, source_path=None, conf_thresh=0.0):
    """obtain the image_info for each image,
    im_file: the path of the image

    return: dict of {'image_id', 'image_h', 'image_w', 'num_boxes', 'boxes', 'features'}
    boxes: the coordinate of each box
    """
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

    # make variable
    with torch.no_grad():
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    cfg.CUDA = True

    fasterRCNN.cuda()

    fasterRCNN.eval()

    # load images
    # im = cv2.imread(im_file)
    try:
        im_in = np.array(Image.open(im_file).convert('RGB'))
    except:
        im_file = os.path.join(source_path, '17_06_4705.jpg')
        im_in = np.array(Image.open(im_file).convert('RGB'))
    # print(im_in.shape)
    if len(im_in.shape) == 2:
        im_in = im_in[:, :, np.newaxis]
        im_in = np.concatenate((im_in, im_in, im_in), axis=2)
    # rgb -> bgr
    im = im_in[:, :, ::-1]

    vis = True

    blobs, im_scales = _get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()
    # pdb.set_trace()

    # the region features[box_num * 2048] are required.
    rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, pooled_feat = fasterRCNN(
            im_data, im_info, gt_boxes, num_boxes, pool_feat=True)

    #print(pooled_feat.shape, type(pooled_feat), flush=True)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev

            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()

            box_deltas = box_deltas.view(1, -1, 4 * len(classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    max_conf = torch.zeros((pred_boxes.shape[0]))
    max_conf = max_conf.cuda()

    if vis:
        im2show = np.copy(im)
    for j in range(1, len(classes)):
        inds = torch.nonzero(scores[:, j] > conf_thresh).view(-1)
        #print('inds:', inds, flush=True)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            index = inds[order[keep]]
            max_conf[index] = torch.where(
                scores[index, j] > max_conf[index], scores[index, j], max_conf[index])
            if vis:
                im2show = vis_detections(
                    im2show, classes[j], cls_dets.cpu().numpy(), 0.5)

    return str(base64.b64encode((max_conf.cpu()).detach().numpy()), encoding='utf-8'), str(base64.b64encode((pooled_feat.cpu()).detach().numpy()), encoding='utf-8')


def load_model():
    # set cfg according to the dataset used to train the pre-trained model
    set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                'ANCHOR_RATIOS', '[0.5,1,2]']

    cfg_from_file('./cfgs/res101.yml')
    cfg_from_list(set_cfgs)

    cfg.USE_GPU_NMS = True
    np.random.seed(cfg.RNG_SEED)

    # Load classes
    classes = ['__background__']
    with open(os.path.join('./data/genome/1600-400-20', 'objects_vocab.txt')) as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())

    load_name = os.path.join('./models/faster_rcnn_res101_vg.pth')

    # initilize the network here. the network used to train the pre-trained model
    fasterRCNN = resnet(classes, 101, pretrained=False)

    fasterRCNN.create_architecture()
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    return classes, fasterRCNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', default='../../data/Twitter15/images')
    parser.add_argument(
        '--save_path', default='../../data/Twitter15/faster_features')
    args = parser.parse_args()
    process_feature(args.source_path, args.save_path)
