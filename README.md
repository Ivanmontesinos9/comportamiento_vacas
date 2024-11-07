
# Automated Cattle Monitoring and Health Assessment with Streamlit

## Introduction

Harness the power of AI and computer vision for modern agriculture. This repository offers tools for cattle activity recognition integrated with Streamlit, enabling real-time insights and actionable data visualizations.

## Table of Contents

- [Quick Start](#quick-start)
- [Directory Structure and Descriptions](#directory-structure-and-descriptions)
- [Usage](#usage)
- [AI Insights for Cattle Care](#ai-insights-for-cattle-care)
- [Conclusion](#conclusion)

## Quick Start

1. Ensure prior implementation of cattle pose detection and activity recognition modules.
2. Familiarize yourself with the basics of Streamlit.
3. Clone the repository and navigate to the project directory.
4. Follow the Setting Up the Environment instructions in the article.

## Directory Structure and Descriptions

```
.
├── README.md                       - This README file.
├── app.py                          - Main Streamlit application for visualizing cattle activity and health insights.
├── activity_recog_inference.py     - Script for cattle activity recognition.
├── final_output.avi                - Sample final output video.
├── output.mp4                      - Sample processed video output.
├── label_encoder.pkl               - Pickled label encoder for activity recognition.
├── test_videos_and_outputs         - Directory containing test videos and corresponding output files.
├── easy_ViTPose & associated files - Original pose estimation and inference tools.
```

**easy_ViTPose & associated files** include:
- `.gitignore`
- `LICENSE`
- `README.md`
- `colab_demo.ipynb`                - Jupyter notebook for demonstration in Google Colab.
- `easy_ViTPose`                    - Core directory containing the model and utility scripts.
- `export.py`                       - Script for model exporting.
- `inference.py`                    - Script for running pose estimation inference.
- `model_split.py`                  - Utility script related to the model.
- `requirements.txt`                - Required packages for CPU.
- `requirements_gpu.txt`            - Required packages for GPU.
- `setup.py`                        - Setup script for the easy_ViTPose package.

## Usage

- Execute the main Streamlit app:

```bash
streamlit run app.py
```

- For cattle activity recognition:

```bash
python activity_recog_inference.py
```

## AI Insights for Cattle Care

- **Behavioral Insights**: Monitor and analyze cattle behavior, ensuring their well-being.
- **Reproductive Health Insights**: Optimize breeding strategies through behavior analysis.
- **Enhanced Veterinary Care**: Benefit from continuous monitoring for proactive veterinary care.

## Conclusion

By integrating AI-driven insights with traditional cattle care, we can enhance livestock management, ensuring both their productivity and well-being. This project bridges modern technology with agricultural practices, setting the foundation for future advancements in this domain.