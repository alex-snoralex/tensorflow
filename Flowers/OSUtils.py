import os
import glob
import shutil

flower_classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']


def organize_photos(base_dir):
    print("Organizing training and validation sets...")
    for cl in flower_classes:
        img_path = os.path.join(base_dir, cl)
        images = glob.glob(img_path + '/*.jpg')
        print("{}: {} Images".format(cl, len(images)))
        train, val = images[:round(len(images) * 0.8)], images[round(len(images) * 0.8):]

        for t in train:
            if not os.path.exists(os.path.join(base_dir, 'train', cl)):
                os.makedirs(os.path.join(base_dir, 'train', cl))
            shutil.move(t, os.path.join(base_dir, 'train', cl))

        for v in val:
            if not os.path.exists(os.path.join(base_dir, 'val', cl)):
                os.makedirs(os.path.join(base_dir, 'val', cl))
            shutil.move(v, os.path.join(base_dir, 'val', cl))
