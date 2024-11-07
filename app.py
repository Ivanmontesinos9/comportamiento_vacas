import streamlit as st
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
import time
import numpy as np
import tempfile


st.title('Automated Cattle Monitoring and Health Assessment')

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

if 'init' not in st.session_state:
    st.session_state.init = False
    st.session_state.frame_count = 0
    st.session_state.df = pd.DataFrame(columns=['time', 'Cows Standing', 'Cows Walking', 'Cows Eating', 'Cows Sitting'])
    st.session_state.processed_frames = []

if uploaded_file is not None and not st.session_state.init:
    MODEL_SIZE = 'b'
    YOLO_SIZE = 's'
    DATASET = 'ap10k'
    ext = '.pth'
    ext_yolo = '.pt'

    MODEL_TYPE = "torch"
    YOLO_TYPE = "torch"
    REPO_ID = 'JunkyByte/easy_ViTPose'
    
    # Nueva URL corregida para el modelo ViTPose
    FILENAME = 'torch/ap10k/vitpose-b-ap10k.pth'
    FILENAME_YOLO = 'yolov8/yolov8' + YOLO_SIZE + ext_yolo

    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    yolo_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_YOLO)

    pose_model = VitInference(model_path, yolo_path, MODEL_SIZE, dataset=DATASET, yolo_size=320, is_video=False, det_class="cow")
    st.session_state.pose_model = pose_model

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

    st.session_state.label_encoder = label_encoder
    act_model = NeuralNet(input_size, 256, 4)
    act_model.load_state_dict(torch.load('model.pth'))
    act_model.eval()

    st.session_state.act_model = act_model

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.session_state.cap = cv2.VideoCapture(tfile.name)
    st.session_state.init = True

if st.session_state.init:
    ret, frame = st.session_state.cap.read()
    if ret:
        count_standing = 0
        count_walking = 0
        count_eating = 0
        count_sitting = 0

        timestamp = str(datetime.now())

        # Detect cow keypoints using your pose detection model
        frame_keypoints = st.session_state.pose_model.inference(frame)

        for cow_id, keypoints in frame_keypoints.items():
            keypoints_array = keypoints.flatten()
            keypoints_tensor = torch.tensor(keypoints_array, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                outputs = st.session_state.act_model(keypoints_tensor)
                _, predicted = torch.max(outputs.data, 1)
                activity_label = st.session_state.label_encoder.inverse_transform(predicted)[0]

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

        st.session_state.frame_count += 1

        new_data = pd.DataFrame([{
            'time': timestamp,
            'Cows Standing': count_standing,
            'Cows Walking': count_walking,
            'Cows Eating': count_eating,
            'Cows Sitting': count_sitting
        }])

        st.session_state.df = pd.concat([st.session_state.df, new_data], ignore_index=True)

        # Display the frame
        st.image(frame, channels="BGR", caption="Processing Frames. Please be patient.", use_column_width=True)

        st.session_state.processed_frames.append(frame)

        time.sleep(0.1)

        st.rerun()

    else:
        st.session_state.cap.release()
        st.write("Processing complete.")

        # Convert list of frames into a video
        height, width, layers = st.session_state.processed_frames[0].shape
        size = (width, height)
        out = cv2.VideoWriter('final_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 15, size)

        for i in range(len(st.session_state.processed_frames)):
            out.write(st.session_state.processed_frames[i])
        out.release()

        os.system("ffmpeg -i final_output.avi -vcodec libx264 output.mp4")

        st.video('output.mp4')

        # Display completed graphs
        activity_totals = st.session_state.df[['Cows Standing', 'Cows Walking', 'Cows Eating', 'Cows Sitting']].sum()
        filtered_totals = activity_totals[activity_totals > 0]
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        filtered_colors = [color for idx, color in enumerate(colors) if activity_totals.index[idx] in filtered_totals.index]

        col1, col2 = st.columns(2)

        # Pie Chart in the first column
        with col1:
            plt.figure(figsize=(10, 6))
            plt.pie(filtered_totals, colors=filtered_colors, autopct='%1.1f%%', startangle=90, labels=filtered_totals.index, shadow=True, wedgeprops=dict(width=0.3))
            plt.title('A Pie Chart of Cattle Activities')
            st.pyplot(plt)

        # Heatmap in the second column
        with col2:
            st.session_state.df['timestamp'] = pd.to_datetime(st.session_state.df['time'])
            activities = ['Cows Standing', 'Cows Walking', 'Cows Eating', 'Cows Sitting']
            st.session_state.df['formatted_time'] = st.session_state.df['timestamp'].dt.strftime('%H:%M:%S')
            heatmap_data = st.session_state.df[['formatted_time'] + activities].set_index('formatted_time').T
            heatmap_data_binary = heatmap_data.applymap(lambda x: 1 if x > 0.5 else 0)
            minty_green_palette = sns.light_palette("mediumseagreen", as_cmap=True)

            plt.figure(figsize=(20, 8))
            sns.heatmap(heatmap_data_binary, cmap=minty_green_palette, cbar=False, linewidths=0.5)

            x_tick_labels = st.session_state.df['formatted_time'].values[::len(st.session_state.df['formatted_time']) // 10]
            x_tick_positions = np.arange(0, len(st.session_state.df['formatted_time']), len(st.session_state.df['formatted_time']) // 10)
            plt.xticks(ticks=x_tick_positions, labels=x_tick_labels, rotation=45, fontsize=10)

            plt.title('Heatmap of Cows Activity Over Time', fontsize=18)
            plt.xlabel('Time', fontsize=15)
            plt.ylabel('Activity', fontsize=15)
            plt.yticks(fontsize=12)
            st.pyplot(plt)

        st.markdown("""
            The videos and graphs presented above offer a comprehensive overview of cattle activities...
        """)
