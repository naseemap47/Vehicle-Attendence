import glob
import os
import shutil
import argparse


ap = argparse.ArgumentParser()              
ap.add_argument("-i", "--data", type=str, required=True,
                help="path to coco dir")
ap.add_argument("-s", '--save', type=str, required=True,
                help="path to coco update")

args = vars(ap.parse_args())
coco_dir = args["data"]
save_dir = args['save']

txt_files = glob.glob(f'{coco_dir}/labels/val2017/*.txt')
for txt in txt_files:
    
    txt_name = os.path.split(txt)[1]
    img_name = os.path.splitext(txt_name)[0] + '.jpg'

    txt_f = open(txt)
    lines = txt_f.read().splitlines()
    line_list = []
    for line in lines:
        s_line = line.split()
        class_id = s_line[0]
        # if class_id == '2' or class_id == '3' or class_id == '5' or class_id == '7':
        if class_id == '73':
            print('[INFO] Class Matched...')
            line_list.append(line)
    
    if len(line_list)>0:
        # Labels
        updated_txt = open(f'{save_dir}/labels/val2017/{txt_name}', 'w+')
        for new_line in line_list:
            new_line_split = new_line.split()
            new_line_split[0] = '0'
            new_line_split = ' '.join(new_line_split)
            u_new_line = f'{new_line_split}\n'
            updated_txt.write(u_new_line)
        updated_txt.close()
        
        # Images
        src = f'{coco_dir}/images/val2017/{img_name}'
        dst = f'{save_dir}/images/val2017/{img_name}'
        shutil.copyfile(src, dst)
        print(f'[INFO] Successfully copied {img_name}')
