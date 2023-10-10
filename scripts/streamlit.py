import os
import glob

from PIL import Image
import numpy as np
import streamlit as st


def video_function(path):
    st.video(path, format="video/mp4")


def open_jpgs_to_nparray(jpgs):
    imgs = []
    for jpg in jpgs:
        img = Image.open(jpg).convert('RGB')
        imgs.append(np.array(img))
    return np.array(imgs)


def collect_imgs(video_num):
    jsons = glob.glob(f"test_data/{video_num}/*.json")
    jpgs = [json.replace(".json", ".jpg") for json in jsons]
    jpgs = sorted(jpgs, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    return open_jpgs_to_nparray(jpgs)


def collect_pred_with_view(view, video_num):
    img_jpgs = glob.glob(f"plot/*_{video_num}_{view}.jpg")
    img_jpgs = sorted(img_jpgs, key=lambda x: int(x.split("/")[-1].split("_")[0]))
    return open_jpgs_to_nparray(img_jpgs)


def main():
    _streamlit = "_streamlit"
    os.makedirs(_streamlit, exist_ok=True)
    
    st.image("images/prediction.png")

    video_list = os.listdir("test_data")
    video_num = st.selectbox("Select video", [None] + video_list)

    view_list = ["pred", "flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9"]
    view = st.selectbox("Mode", view_list)
    if video_num in video_list:
        imgs = collect_imgs(video_num)
        from utils import save_with_imageio
        
        video_path = f"{_streamlit}/test_{video_num}.mp4"
        if not os.path.exists(video_path):
            save_with_imageio(video_path, imgs)
        
        gt_imgs = collect_pred_with_view("gt", video_num)
        gt_video_path = f"{_streamlit}/gt_{video_num}.mp4"
        if not os.path.exists(gt_video_path):
            save_with_imageio(gt_video_path, gt_imgs)

        view_imgs = collect_pred_with_view(view, video_num)
        view_video_path = f"{_streamlit}/{view}_{video_num}.mp4"
        if not os.path.exists(view_video_path):
            save_with_imageio(view_video_path, view_imgs)
        
        st.text("video")
        st.video(video_path, format="video/mp4")
        st.text("gt")
        st.video(gt_video_path, format="video/mp4")
        st.text("pred")
        st.video(view_video_path, format="video/mp4")

        if video_num == "4":
            st.text("4번 비디오에서는, 2명의 겹치는 사람은 구분을 못하는듯.")

if __name__ == "__main__":
    main()
