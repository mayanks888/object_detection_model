import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2
import model
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from decimal import Decimal as D
from dataloader_visualise_riginal import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer


assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', default='csv',help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes',default="berkely_class.csv", help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', default="temp_2.csv", help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--model',default='/home/teai/Desktop/mayank/pytorch-retinanet-master (4)/checkpoint/old_datasets_with_missing_images_annotations/retina_fpn_11', help='Path to model (.pt) file.')
    parser.add_argument('--input_path', help="Input Folder", default='./input_images')
    parser.add_argument('--output_path', help="Output folder", default='./output_images')

    parser = parser.parse_args(args)

    # ____________________________________________________________________

    input_folder = parser.input_path
    output_folder = parser.output_path

    if not os.path.exists(input_folder):
        print("Input folder not found")
        return 1

    if not os.path.exists(output_folder):
        print("Output folder not present. Creating New folder...")
        os.makedirs(output_folder)
    file_path = []
    for root, _, filenames in os.walk(input_folder):
        if (len(filenames) == 0):
            print("Input folder is empty")
            return 1
        time_start = time.time()
        print("Creating object detection.. ")

        for filename in filenames:
            file_path.append(os.path.join(root, filename))
        np.savetxt("temp.csv", file_path, delimiter=",", fmt='%s')
    # __________________________________________________________________________-

    if parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'csv':
        dataset_val = CSVDataset(train_file="temp_2.csv", class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
    ###################################################333

    # one way to load model
    retinanet = torch.load(parser.model)
    #the other way to load model

    # net = RetinaNet()
    # retinanet.load_state_dict(torch.load('/home/teai/Desktop/mayank/pytorch-retinanet-master (4)/checkpoint/retina_fpn_1'))
    # retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    #
    #
    #
    # # checkpoint = torch.load('./checkpoint/saved_with_epochs/retina_fpn_1')['net']
    # retinanet=load_state_dict(torch.load('./checkpoint/saved_with_epochs/retina_fpn_1')['net'])
    # # retinanet=torch.load[('./checkpoint/saved_with_epochs/retina_fpn_1')]
    # # retinanet.load_state_dict(checkpoint['net'])
    ##################################################3
    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    retinanet.eval()

    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption,score):
        # score=round(score, 2)
        # score=truncate(score, 2)
        score=format(score, '.2f')
        score=caption + " "+str(score)
        b = np.array(box).astype(int)
        cv2.putText(image, score, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, score, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    st = time.time()
    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():
            # st = time.time()
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(data['img'].float())
            # print('Elapsed time: {}'.format(time.time()-st))
            idxs = np.where(scores>0.5)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img<0] = 0
            img[img>255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                score = float(scores[idxs[0][j]])
                # score=str(score)
                # print(score)
                draw_caption(img, (x1, y1, x2, y2), label_name,score)


                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                # print(label_name)

            # cv2.imshow('img', img)
            # cv2.waitKey(500)
            # cv2.destroyAllWindows()

            # cv2.imshow('img', img)
            # k = cv2.waitKey(0)

            image_nm="image_" + str(idx) + '.png'
            save_path = os.path.join(output_folder,image_nm)
            cv2.imwrite(save_path, img)
            cv2.destroyAllWindows()
            # break
    print('Elapsed time: {}'.format(time.time() - st))

    # if k == 27:  # wait for ESC key to exit
    #     cv2.destroyAllWindows()
    # elif k == ord('s'):  # wait for 's' key to save and exit
    #     save_path=os.path.join(output_folder,"messigray"+idx+'.png')
    #     cv2.imwrite('messigray.png', img)
    #     cv2.destroyAllWindows()

if __name__ == '__main__':
    main()