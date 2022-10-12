from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.hubconf import custom
from utils.plots import plot_one_box
import cv2


# DeepSort
tracker = DeepSort(max_age=5)

confidence = 0.6
class_labels = open('class.txt').read().splitlines()
class_selected_id = class_labels.index('car')

model = custom(path_or_model='yolov7.pt')
cap = cv2.VideoCapture('Videos/s1.mp4')
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
            if class_id == class_selected_id:
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
            # current_no_class.append([class_labels[id]])

    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
cap.release()
