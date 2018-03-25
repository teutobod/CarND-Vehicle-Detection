from find_cars import *

model_file = './data/model.p'
print('Loading classifier model from file', model_file)
clf, scaler = load_model(model_file)
parameter = FeatureParameter()

def process_frame(img):
    car_boxes = find_cars(img, clf, scaler, parameter)

    from heatmap import HeatMap
    heatmap = HeatMap(threshold=3)
    heatmap.add_heat(car_boxes)
    heatmap.apply_threshold()

    from scipy.ndimage.measurements import label
    labels = label(heatmap.get_headmap())

    label_box_img = draw_labeled_bboxes(np.copy(img), labels)

    return label_box_img

from moviepy.editor import VideoFileClip

# video_output = './videos/test_video_ouput.mp4'
# clip = VideoFileClip("./videos/test_video.mp4")

video_output = './videos/project_video_ouput.mp4'
clip = VideoFileClip("./videos/project_video.mp4")

output_clip = clip.fl_image(process_frame)
output_clip.write_videofile(video_output, audio=False)
