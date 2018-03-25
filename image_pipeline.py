import glob
from find_cars import *

test_images = list(map(lambda img: read_image(img), glob.glob('./test_images/*.jpg')))

model_file = './data/model.p'
print('Loading classifier model from file', model_file)
clf, scaler = load_model(model_file)
parameter = FeatureParameter()


box_imgs = []
for img in test_images:
    car_boxes = find_cars(img, clf, scaler, parameter)

    car_boxes_img = draw_cars(img, car_boxes)
    box_imgs.append(car_boxes_img)

    from heatmap import HeatMap

    heatmap = HeatMap(threshold=2)
    heatmap.add_heat(car_boxes)
    heatmap.apply_threshold()
    heatmap_img = heatmap.get_headmap()

    from scipy.ndimage.measurements import label
    labels = label(heatmap_img)

    box_imgs.append(heatmap_img)
    label_box_img = draw_labeled_bboxes(np.copy(img), labels)
    box_imgs.append(label_box_img)

plot_images(box_imgs)
