from util import *
from moviepy.editor import VideoFileClip

if USE_SAMPLE == True:
    random_idxs = np.random.randint(0, len(cars), SAMPLE_SIZE)
    cars = np.array(cars)[random_idxs]
    notcars = np.array(notcars)[random_idxs]

car_features = extract_features(cars, COLOR_SPACE, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, HOG_CHANNEL)
notcar_features = extract_features(notcars, COLOR_SPACE, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, HOG_CHANNEL)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.2, random_state = rand_state)

# Parameter Tuning
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
svc.fit(X_train, y_train)
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


ystart = 350
ystop = None
heat_arr = deque(maxlen = 10)

def process_image(image):
    global heat_arr
    heatmap = find_cars(image, ystart, ystop, 1.5, svc, X_scaler)
    heatmap += find_cars(image, ystart, 500, 1, svc, X_scaler)
    heat_arr.append(heatmap)
    if len(heat_arr) == 10:
        avg_heat = sum(heat_arr)/len(heat_arr)
        heatmap = avg_heat
    heatmap = apply_threshold(heatmap, 2)
    heatmap = np.clip(heatmap, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img

# vid_output = 'project_attempt.mp4'
# clip1 = VideoFileClip("project_video.mp4")
# first_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# first_clip.write_videofile(vid_output, audio=False)
img = mpimg.imread('test_images/test4.jpg')
img = process_image(img)
plt.imshow(img)
plt.show()
