import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.hubconf import custom
from utils.plots import plot_one_box
import cv2
import easyocr


# DeepSort
tracker = DeepSort(max_age=5)

confidence = 0.2
class_labels = open('class.txt').read().splitlines()
class_selected_id = class_labels.index('car')
# print('class_selected_id: ', class_selected_id)

model = custom(path_or_model='yolov7.pt')
np_model = custom(path_or_model='best.pt')

cap = cv2.VideoCapture('Videos/t1.mp4')
while True:
    success, img = cap.read()
    if not success:
        break

    # bbox_list = []
    track_box_list = []
    # current_no_class = []
    
    results = model(img)
    # Bounding Box
    box = results.pandas().xyxy[0]
    # print(box.index)
    class_list = box['class'].to_list()
    for i in box.index:
        xmin, ymin, xmax, ymax, conf, class_id = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
            int(box['ymax'][i]), box['confidence'][0], box['class'][i]
        if conf > confidence:
            # if class_id == class_selected_id:
            if class_id == 2 or class_id == 3 or class_id == 5 or class_id == 7:
                boxes = [xmin, ymin, int(xmax-xmin), int(ymax-ymin)]
                bbs = (boxes, conf, class_labels[class_id])
                track_box_list.append(bbs)

    if len(track_box_list)>0:
        tracks = tracker.update_tracks(track_box_list, frame=img)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()

            print('track_id: ', track_id)
            print('ltrb: ', ltrb)

            bbox = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]
            plot_one_box(bbox, img, (0, 150, 0), str(track_id), 2)
            
            # Number Plate Model
            img_roi = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            np_result = np_model(img_roi)
            # Bounding Box
            np_box = np_result.pandas().xyxy[0]
            # print(box.index)
            np_class_list = np_box['class'].to_list()
            for i in np_box.index:
                xmin, ymin, xmax, ymax, conf, class_id = int(np_box['xmin'][i]), int(np_box['ymin'][i]), int(np_box['xmax'][i]), \
                    int(np_box['ymax'][i]), np_box['confidence'][0], np_box['class'][i]

                np_bbox = [xmin, ymin, xmax, ymax]

                # Pre-Processing
                img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                reader = easyocr.Reader(['en'])
                result_ocr = reader.readtext(img_roi_gray)

                print(result_ocr)

                for i in result_ocr:
                    # cv2.rectangle(img_roi, i[0][0], i[0][2], (0, 255, 0), 2)
                    # cv2.putText(img_roi, f'{i[1]}', i[0][0], cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2) 

                # print('np_bbox:\n', np_bbox)
                    plot_one_box(np_bbox, img_roi, (150, 150, 0), f'{i[1]}', 1)
            
            cv2.imwrite(f"Images/{len(os.listdir('Images'))}.jpg", img_roi)

    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
    # cv2.imwrite(f"Images/{len(os.listdir('Images'))}.jpg", img_roi)
cap.release()
