from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from utils.hubconf import custom
from utils.plots import plot_one_box
import cv2
import datetime
import os


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


confidence = 0.5
class_labels = 'no_plate'
os.makedirs('Images', exist_ok=True)

# YOLOv7
model = custom(path_or_model='best.pt', gpu=True)

# TrOCR
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model_ocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

img = cv2.imread('test_img/3.png')
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

        ## TrOCR setup
        img_roi_rgb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
        pixel_values = processor(img_roi_rgb, return_tensors="pt").pixel_values
        generated_ids = model_ocr.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print('result_ocr: ', generated_text)
        
        plate_no = generated_text.upper()
        getAttendance(plate_no, img_roi)
        plot_one_box([xmin, ymin, xmax, ymax], img, (0, 150, 0), f'{plate_no}', 2)
        
        # Plate Image (ROI)
        # cv2.imshow('Video roi', img_roi)
        
# img = cv2.resize(img, (940, 550))
cv2.imshow('Video', img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
