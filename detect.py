import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.hubconf import custom
from utils.plots import plot_one_box
import cv2
import easyocr
import datetime


def getAttendance(name, number, img_roi):
    with open('Attendance.csv', 'r+') as f:
        data = f.readlines()
        name_list = []
        for line in data:
            entry = line.split(',')
            name_list.append(entry[0])

        # if name NOT present in the list, it will add
        if name not in name_list:
            c_time = datetime.datetime.now()
            date_str = c_time.strftime("%d/%m/%Y %H:%M:%S")
            f.writelines(f'{name}, {number}, {date_str}\n')

            # Save Number Plate
            cv2.imwrite(f"Images/{len(os.listdir('Images'))}.jpg", img_roi)


save = True
confidence = 0.5
class_labels = 'no_plate'
os.makedirs('Images', exist_ok=True)

# DeepSort
tracker = DeepSort(max_age=5)

# YOLOv7
model = custom(path_or_model='best.pt', gpu=True)

cap = cv2.VideoCapture('test.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if save:
    out_vid = cv2.VideoWriter('output.mp4', 
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (w, h))

while True:
    success, img = cap.read()
    if not success:
        break

    track_box_list = []
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img)

    # Bounding Box
    box = results.pandas().xyxy[0]
    class_list = box['class'].to_list()
    for i in box.index:
        xmin, ymin, xmax, ymax, conf, class_id = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
            int(box['ymax'][i]), box['confidence'][i], box['class'][i]
        if conf > confidence:
            boxes = [xmin, ymin, int(xmax-xmin), int(ymax-ymin)]
            bbs = (boxes, conf, class_labels[class_id])
            track_box_list.append(bbs)
            # plot_one_box([xmin, ymin, xmax, ymax], img, (0, 150, 0), class_labels, 2)
    
    if len(track_box_list)>0:
        tracks = tracker.update_tracks(track_box_list, frame=img)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()

            # print('track_id: ', track_id)
            # print('ltrb: ', ltrb)

            bbox = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]

            # Pre-Processing
            img_roi = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            reader = easyocr.Reader(['en'])
            result_ocr = reader.readtext(img_roi_gray)

            print('result_ocr:\n', result_ocr)

            if len(result_ocr)>0:
                plate_no = result_ocr[0][1].upper()
                plot_one_box(bbox, img, (0, 150, 0), f'{plate_no}', 4)
                if result_ocr[0][2] > 0.5:                    
                    getAttendance(track_id, plate_no, img_roi)
            
            # Save Number Plate Image
            cv2.imshow('Video roi', img_roi)
            
    if save:
        out_vid.write(img)

    img = cv2.resize(img, (940, 550))
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

if save:
    out_vid.release()
cap.release()
