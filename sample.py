from pgms.head_eye.head_eye import head_eye
from pgms.hands.hand_main.hand_detector_XGBoost import hand
from pgms.body.pose_detector_XGBoost import body
from pgms.smile.smile_detector_XGboost import smile_detector
from pgms.head_eye.head_pose import head_pose
import streamlit as st

loading_bar_hand = st.progress(0)
#hand(0, loading_bar_hand)