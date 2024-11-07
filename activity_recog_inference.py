import cv2
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from easy_ViTPose import VitInference
import os
from huggingface_hub import hf_hub_download
import pickle
from datetime import datetime 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_SIZE = 'b'
YOLO_SIZE = 's'
DATASET = 'ap10k'
ext = '.pth'
ext_yolo = '.pt'

MODEL_TYPE = "torch"
YOLO_TYPE = "torch"
REPO_ID = 'JunkyByte/easy_ViTPose'
FILENAME = os.path.join(MODEL_TYPE, f'{DATASET}/vitpose-' + MODEL_SIZE + f'-{DATASET}') + ext
FILENAME_YOLO = 'yolov8/yolov8' + YOLO_SIZE + ext_yolo

print(f'Downloading model {REPO_ID}/{FILENAME}')
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
yolo_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_YOLO)


pose_model = VitInference(model_path, yolo_path, MODEL_SIZE,
                     dataset=DATASET, yolo_size=320, is_video=False, det_class = "cow")

# Define model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

input_size = 51  
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

act_model = NeuralNet(input_size, 256, 4)
act_model.load_state_dict(torch.load('model.pth'))
act_model.eval()

# Load the video
video_path = 'example_1.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create a VideoWriter object to save the output video
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

df = pd.DataFrame(columns = ['time','Cows Standing','Cows Walking','Cows Eating','Cows Sitting'])
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    count_standing = 0
    count_walking = 0
    count_eating = 0
    count_sitting = 0

    timestamp = str(datetime.now())

    # Detect cow keypoints using your pose detection model
    frame_keypoints = pose_model.inference(frame)
    print("Detecting keypoints")

    for cow_id, keypoints in frame_keypoints.items():
        # Preprocess keypoints for classification model
        keypoints_array = keypoints.flatten()  # Flatten into a 1D array
        keypoints_tensor = torch.tensor(keypoints_array, dtype=torch.float32).unsqueeze(0)

        # Run classification model
        with torch.no_grad():
            outputs = act_model(keypoints_tensor)
            _, predicted = torch.max(outputs.data, 1)
            activity_label = label_encoder.inverse_transform(predicted)[0]

        # Draw keypoints and activity label on the frame
        for i, keypoint in enumerate(keypoints):
            y, x, confidence = keypoint
            if activity_label == "standing":
                cv2.putText(frame, f'{activity_label}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                count_standing = count_standing + 1
            elif activity_label == "walking":
                cv2.putText(frame, f'{activity_label}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                count_walking = count_walking + 1
            elif activity_label == "eating":
                cv2.putText(frame, f'{activity_label}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                count_eating = count_eating + 1 
            elif activity_label == "sitting":
                cv2.putText(frame, f'{activity_label}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                count_sitting = count_sitting + 1
            break

    frame_count = frame_count +1 

    print("Number of cattle standing in this frame:",count_standing )
    print("Number of cattle walking in this frame:",count_walking )
    print("Number of cattle eating in this frame:",count_eating )
    print("Number of cattle sitting in this frame:",count_sitting )

    df.loc[frame_count] = [timestamp,count_standing ,count_walking, count_eating, count_sitting] 

    # Write the frame to the output video
    out.write(frame)

# Release the video objects
cap.release()
out.release()

# Summing up the activities
activity_totals = df[['Cows Standing', 'Cows Walking', 'Cows Eating', 'Cows Sitting']].sum()

# Filter out activities with a count of 0
filtered_totals = activity_totals[activity_totals > 0]

# Define a color palette
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']

# Adjusted color palette based on the number of activities present
filtered_colors = [color for idx, color in enumerate(colors) if activity_totals.index[idx] in filtered_totals.index]

# Plotting the adjusted pie chart
plt.figure(figsize=(10, 6))
plt.pie(filtered_totals, colors=filtered_colors, autopct='%1.1f%%', startangle=90, labels=filtered_totals.index, shadow=True, wedgeprops=dict(width=0.3))
plt.title('A Pie Chart of Cattle Activities')
plt.savefig('pie_chart.png')

# Convert the 'timestamp' column to datetime format
df['timestamp'] = pd.to_datetime(df['time'])

# Define the list of activities
activities = ['Cows Standing', 'Cows Walking', 'Cows Eating', 'Cows Sitting']

# Check the span of the data
time_span = df['timestamp'].max() - df['timestamp'].min()

# Format the datetime values based on the span of the data
if time_span.days > 1:
    df['formatted_time'] = df['timestamp'].dt.strftime('%d %H:%M')
elif time_span.seconds >= 3600:
    df['formatted_time'] = df['timestamp'].dt.strftime('%H:%M')
elif time_span.seconds >= 300:
    df['formatted_time'] = df['timestamp'].dt.strftime('%M:%S')
else:
    df['formatted_time'] = df['timestamp'].dt.strftime('%S') + 's'

# Prepare data for heatmap
heatmap_data = df[['formatted_time'] + activities].set_index('formatted_time').T

# Binarize the heatmap data to represent whole numbers (0 or 1)
heatmap_data_binary = heatmap_data.applymap(lambda x: 1 if x > 0.5 else 0)

# Create a minty-green light color palette that starts from white and progresses to mediumseagreen
minty_green_palette = sns.light_palette("mediumseagreen", as_cmap=True)

# Selecting a subset of x-axis ticks for better readability and performance
num_ticks = 50  # Number of ticks to display on the x-axis
tick_indices = range(0, len(df['formatted_time']), len(df['formatted_time']) // num_ticks)

# Creating the heatmap visualization with the minty-green color palette
plt.figure(figsize=(20, 8))
sns.heatmap(heatmap_data_binary.iloc[:, tick_indices], cmap=minty_green_palette, cbar=False, linewidths=0.5)

plt.title('Heatmap of Cows Activity Over Time', fontsize=18)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Activity', fontsize=15)
plt.xticks(ticks=range(len(tick_indices)), labels=df['formatted_time'].iloc[tick_indices].values, rotation=45, fontsize=10)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('heat_map.png')
