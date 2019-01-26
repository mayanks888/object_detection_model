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
import model_with_two_more_classification as model
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from decimal import Decimal as D
from dataloader_visualise import CocoDataset, CSVDataset, collater, Resizer,Resi_mayank, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer


assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', default='csv',help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes',default="berkely_class.csv", help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', default="/home/teai/Desktop/mayank/pytorch-retinanet_special_for_visualise/aptive images/video_to_images", help='Path to file containing validation annotations (optional, see readme)')
    # parser.add_argument('--csv_val', default="/home/teai/Desktop/mayank/pytorch-retinanet_special_for_visualise/aptive images/FCA_L19DOD02T_20170624_RWUP_Kokomo_RW_JR_092835_001", help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--model',default='./checkpoint/retina_fpn_4_daytime', help='Path to model (.pt) file.')
    # parser.add_argument('--input_path', help="Input Folder", default='./aptive images/FCA_L19DOD02T_20170624_RWUP_Kokomo_RW_JR_092835_001')
    # parser.add_argument('--input_path', help="Input Folder", default='./input_images')
    parser.add_argument('--input_path', help="Input Folder", default='/home/teai/working_directory/datasets/Berkely_DeepDrive/bdd100k/images/10k/test')
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
    for root, _, filenames in sorted(os.walk(input_folder)):
        # for root, _, filenames in sorted(os.listdir(input_folder)):
        # root=input_folder

        # for filenames in os.listdir(input_folder):
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
        # dataset_val = CSVDataset(train_file="temp.csv", class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resi_mayank()]))
        dataset_val = CSVDataset(train_file="temp.csv", class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
        # dataset_val = CSVDataset(train_file="temp.csv", class_list=parser.csv_classes, transform=transforms.Compose([Normalizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    # dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=sampler_val,shuffle=False)
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
    loop=1
    datasets=dataset_val
    # data= datasets.data_input(1)
    # for idx, data in enumerate(dataloader_val):

    # for idx, data in enumerate(datasets.data_input[loop]):

    for loop in range(1,1086):
        data,file_name = datasets.data_input(loop)
        print("processing frame {fn} :    image count {ct}" .format(fn=file_name,ct=loop))
        try:
            # loop+=1
            with torch.no_grad():
                # st = time.time()
                if torch.cuda.is_available():
                    scores, classification, transformed_anchors, weather_prob,daytime_prob = retinanet(data['img'].cuda().float())
                else:
                    scores, classification, transformed_anchors,weather_prob,daytime_prob = retinanet(data['img'].float())

                # ______________________________________________________________________________________________-
                # this is what I am doing for weather text
                weather_classes = ['clear', 'overcast', 'partly_cloudy', 'rainy', 'snowy']
                daytime_classes=['dawn_or_dusk', 'night', 'daytime']
                # weather_prob=float(weather_prob)
                # print(weather_prob.data[0])
                weather_prob=weather_prob.data[0].tolist()
                print(weather_prob)
                # data = [0.93, 0.21, 0.56, 0.48, 0.58]
                max_index = (weather_prob.index(max(weather_prob)))
                weather_info = (weather_classes[max_index])

                #######################333333
                daytime_prob = daytime_prob.data[0].tolist()
                print(daytime_prob)
                # data = [0.93, 0.21, 0.56, 0.48, 0.58]
                max_index = (daytime_prob.index(max(daytime_prob)))
                daytime_info = (daytime_classes[max_index])
                #############################

                # cv2.putText(image, score, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                # cv2.putText(img, weather_info, (230, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (50, 36, 174), 2, cv2.LINE_AA)
                # ___________________________________________________________________________________________________
                # print('Elapsed time: {}'.format(time.time()-st))
                idxs = np.where(scores>0.5)
                img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

                img[img<0] = 0
                img[img>255] = 255

                img = np.transpose(img, (1, 2, 0))

                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                #cv2.putText(img, weather_info, (230, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (50, 36, 174), 2, cv2.LINE_AA)

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
                    #


                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                    # print(label_name)
                cv2.putText(img, weather_info, (230, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (50, 36, 174), 2, cv2.LINE_AA)
                cv2.putText(img, daytime_info, (550, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (50, 36, 174), 2, cv2.LINE_AA)
                # cv2.imshow('img', img)
                # cv2.waitKey(10000)
                # cv2.destroyAllWindows()

                # cv2.imshow('img', img)
                # k = cv2.waitKey(0)
                # ___________________________________________________
                image_nm=file_name.split('/')[-1]
                # image_nm="frame_" + str(loop) + '.jpg'

                save_path = os.path.join(output_folder,image_nm)
                cv2.imwrite(save_path, img)
                cv2.destroyAllWindows()
        except IOError:
            continue
            print("Existing Object Detection...")
            # break
    print('Elapsed time: {}'.format(time.time() - st))
    ## ___________________________________________________

    # if k == 27:  # wait for ESC key to exit
    #     cv2.destroyAllWindows()
    # elif k == ord('s'):  # wait for 's' key to save and exit
    #     save_path=os.path.join(output_folder,"messigray"+idx+'.png')
    #     cv2.imwrite('messigray.png', img)
    #     cv2.destroyAllWindows()

if __name__ == '__main__':
    main()