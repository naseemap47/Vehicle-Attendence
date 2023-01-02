import os
from utils.hubconf import custom
from utils.plots import plot_one_box
import cv2
import easyocr
import datetime


def getAttendance(number, img_roi):
    with open('Attendance.csv', 'r+') as f:
        data = f.readlines()
        name_list = []
        for line in data:
            entry = line.split(',')
            name_list.append(entry[0])

        # if name NOT present in the list, it will add
        if number not in name_list:
            c_time = datetime.datetime.now()
            date_str = c_time.strftime("%d/%m/%Y %H:%M:%S")
            f.writelines(f'{number}, {date_str}\n')

            # Save Number Plate
            cv2.imwrite(f"Images/{len(os.listdir('Images'))}.jpg", img_roi)


save = False
confidence = 0.5
ocr_conf = 0.5
class_labels = 'no_plate'
os.makedirs('Images', exist_ok=True)

# YOLOv7
model = custom(path_or_model='best.pt', gpu=True)

cap = cv2.VideoCapture('t1.mp4')
# cap = cv2.VideoCapture(2)

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

    results = model(img)

    # Bounding Box
    box = results.pandas().xyxy[0]
    class_list = box['class'].to_list()
    for i in box.index:
        xmin, ymin, xmax, ymax, conf, class_id = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
            int(box['ymax'][i]), box['confidence'][i], box['class'][i]
        if conf > confidence:
            # plot_one_box([xmin, ymin, xmax, ymax], img, (0, 150, 0), class_labels, 2)
            img_roi = img[ymin:ymax, xmin:xmax]
            img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            reader = easyocr.Reader(['en'])
            result_ocr = reader.readtext(img_roi_gray)

            print('result_ocr:\n', result_ocr)

            if len(result_ocr)>0:
                plate_no = ''
                for ocr in result_ocr:
                    plate_no = ocr[1].upper() + plate_no
                    if ocr[2] > ocr_conf:                   
                        getAttendance(plate_no, img_roi)
                plot_one_box([xmin, ymin, xmax, ymax], img, (0, 150, 0), f'{plate_no}', 4)
            
            # Plate Image (ROI)
            cv2.imshow('Video roi', img_roi)
            
    if save:
        out_vid.write(img)

    # img = cv2.resize(img, (940, 550))
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

if save:
    out_vid.release()
cap.release()
