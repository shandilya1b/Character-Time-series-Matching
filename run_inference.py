from detection.DETECTION import Detection
import os
import argparse
from tqdm import tqdm
import cv2
import shutil
from tqdm import tqdm
import process_plate
import imutils
import math
import numpy as np

def suppressAllWarnings(params={}):
    '''
        LP: suppress all warnings
        params = { regex: [ String ] }
    '''
    options = {
        "level": "ignore",
        "regex": [ r'All-NaN (slice|axis) encountered' ]
    }
    for attr in params:
        options[attr]=params[attr]
    import warnings
    # system-wide warning
    def fxn():
        warnings.warn("deprecated", DeprecationWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()
        # LP custom regex warning
        for regex in options['regex']:
            warnings.filterwarnings('ignore', regex)
    # by module warnings
    try:
        # numpy warnings
        import numpy as np
        np.seterr(all=options['level'])
    except:
        pass

suppressAllWarnings()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object-weights', nargs='+', type=str, default='object.pt', help='model path or triton URL')
    parser.add_argument('--char-weights', nargs='+', type=str, default='char.pt', help='model path or triton URL')
    parser.add_argument('--out-dir', default='out', help='path to output folder')
    parser.add_argument('--dataset-dir', help='path to dataset to run inference over')
    parser.add_argument('--object-imgsz', '--object-img', '--object-img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--char-imgsz', '--char-img', '--char-img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--object-conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--object-iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--char-conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--char-iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.object_imgsz *= 2 if len(opt.object_imgsz) == 1 else 1  # expand
    opt.char_imgsz *= 2 if len(opt.char_imgsz) == 1 else 1  # expand
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    dataset_dir = opt.dataset_dir
    print('Loading data from: ', dataset_dir)
    dest_path=opt.out_dir
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)    
    os.mkdir(dest_path)
    
    object_model = Detection(size=opt.object_imgsz,
                         weights_path=opt.object_weights,
                         device=opt.device,
                         iou_thres=opt.object_iou_thres,
                         conf_thres=opt.object_conf_thres)

    char_model = Detection(size=opt.char_imgsz,
                         weights_path=opt.char_weights,
                         device=opt.device,
                         iou_thres=opt.char_iou_thres,
                         conf_thres=opt.char_conf_thres)
    labels = os.listdir(dataset_dir)
    label_count = 0
    acc = 0
    case_fail = []
    labels.sort()
    for item in labels:
        if os.path.isdir(os.path.join(dataset_dir, item)):
            label_dir = os.path.join(dataset_dir, item)
            out_dir = os.path.join(dest_path, item)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.mkdir(out_dir)
            plates_out_dir = os.path.join(out_dir, 'plates')
            if os.path.exists(plates_out_dir):
                shutil.rmtree(plates_out_dir)
            os.mkdir(plates_out_dir)

            track_plates = []
            for image_name in tqdm(os.listdir(label_dir)):
                # detected_plates = []
                image_path = os.path.join(label_dir, image_name)
                img = cv2.imread(image_path)
                # print('opening image: ', image_path)
                det_results, resized_img = object_model.detect(img.copy())
                copy_img = resized_img.copy()
                for name,conf,box in det_results:
                    resized_img=cv2.putText(resized_img, "{}".format(name), (int(box[0]), int(box[1])-3),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 0, 255), 3)
                    resized_img = cv2.rectangle(resized_img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 2)
                    if 'license plate' in name:
                        # detected_plates.append(copy_img[int(box[1]): int(box[3]), int(box[0]): int(box[2])].copy())
                        track_plates.append(copy_img[int(box[1]): int(box[3]), int(box[0]): int(box[2])].copy())
                cv2.imwrite(os.path.join(out_dir, image_name), resized_img)


            # Assuming only a single number plate currently.
            # Require Tracking for multi plate recognition
            alpha=0
            results=""
            track_boxs=[]
            Ws=[]
            Hs=[]
            if len(track_plates) == 0:
                continue
        
            for image in tqdm(track_plates):
                h,w,_=image.shape
                Ws.append(w)
                Hs.append(h)
                image = imutils.rotate(image, math.degrees(alpha))
                detections,image=char_model.detect(image)
                # For recognising plates and annotating images.
                # This will be moved into the tracked recognition code
                resized_recog_img = image.copy()
                for name,conf,box in detections:
                    resized_recog_img=cv2.putText(resized_recog_img, "{}".format(name), (int(box[0]), int(box[1])-3),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                                            (255, 0, 255), 2)
                    resized_recog_img = cv2.rectangle(resized_recog_img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 2)
                cv2.imwrite(os.path.join(plates_out_dir, image_name), resized_recog_img)
                if len(detections)==0:
                    continue
                
                dets=process_plate.merge_box(detections)
                track_box=[]

                for label, confidence, box in dets:
                    track_box.append([(int(round(box[0] - box[2]/2))), (int(round(box[1] - box[3]/2))), (int(round(box[0] + box[2]/2))), (box[1] + box[3]/2),
                                           [[float(c)] for c in confidence.split("-")],[[l] for l in label.split("-")]])
                track_box=np.array(track_box,dtype=object)
                center_x = (track_box[:, 0] + track_box[:, 2]) / 2
                center_y = (track_box[:, 1] + track_box[:, 3]) / 2
                track_boxs.append(track_box)
                
                center = np.vstack((center_x, center_y)).T

                degree=process_plate.find_angle(center_x,center_y)

                if 3<abs(math.degrees(degree))<25:
                    alpha-=degree

            old_char = np.zeros((0, 0))
            for track_box in track_boxs:
                arr_track=process_plate.matching_char(old_char,track_box)
                old_char=arr_track
            Hm=np.mean(np.array(Hs)) if len(Hs)>0 else 0
            Wm=np.mean(np.array(Ws)) if len(Ws)>0 else 0

            if arr_track.shape[0]>7:
                arr_track=process_plate.merge_box_arr_track(arr_track)
            arr_track=sorted(arr_track, key=lambda x: float(x[0]))
            re=""
            arr_track=np.array([arr_ for arr_ in arr_track if len(arr_[5])>=1/2*(len(track_plates))],dtype=object)
            for arr_ in arr_track:

                clss=max(arr_[5],key=arr_[5].count)
                clss=process_plate.get_maximum_conf_char(arr_)   
                re+=clss         
            if Hm*2>Wm:
                center_x = (arr_track[:, 0] + arr_track[:, 2]) / 2
                center_y = (arr_track[:, 1] + arr_track[:, 3]) / 2
                chars = ["{}".format(process_plate.get_maximum_conf_char(track_box_)) for track_box_ in arr_track]
                _,re=process_plate.find_chars_plate(center_x,center_y,chars)
            re=re.replace("-","")
            label=labels[label_count].replace("-","")
            re=re[0:3].replace("0","O").replace("1","I")+re[3:]
            # print(fd,re)
            # TODO implement character level accuracy computation
            if re == label:
                acc+=1
            else:
                case_fail.append(item.split('.')[0])
            label_count += 1

    print(acc/len(labels))
    print("case fail",case_fail)
