import io
import os
import pandas as pd
from pyexpat import model
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms
import pickle
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset


def load_image1():
    uploaded_file1 = st.file_uploader(label='Pick an image to test', key='1')
    if uploaded_file1 is not None:
        image_data1 = uploaded_file1.getvalue()
        st.image(image_data1)
        return Image.open(io.BytesIO(image_data1))
    else:
        return None
    
def load_image2():
    uploaded_file2 = st.file_uploader(label='Pick an image to test', key='2')
    if uploaded_file2 is not None:
        image_data2 = uploaded_file2.getvalue()
        st.image(image_data2)
        return Image.open(io.BytesIO(image_data2))
    else:
        return None


def load_model():
    model = torch.load('/home/ikhsan/Documents/MBKM - Studi Independen 2022/Project akhir/Web_PA/web_app_ttd/trained-model.pt')
    return model


def load_labels():
    col_names = ['x0', 'x1', 'categories']
    data = pd.read_csv('/home/ikhsan/Documents/MBKM - Studi Independen 2022/Project akhir/Web_PA/web_app_ttd/sign_data/sign_data/test_data.csv',  names=col_names)
    data.drop('x0', inplace=True, axis=1)
    data.drop('x1', inplace=True, axis=1)
    data.to_csv('test.csv', index=False)
    labels_file = data
    
    labels_file = '/home/ikhsan/Documents/MBKM - Studi Independen 2022/Project akhir/Web_PA/web_app_ttd/streamlit/test.csv'

    with open(labels_file) as f:
        categories = [s.strip() for s in f.readlines()]
        if categories==torch.FloatTensor([[0]]):
            categories="Similiar Signature"
        else:
            categories="Signature Not Simmiliar"
        return categories
    
    
    


def predict(model, categories, image1, image2):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor1 = preprocess(image1)
    input_tensor2 = preprocess(image2)
    input_batch = input_tensor1.unsqueeze(0), input_tensor2.unsqueeze(0)

    #concat = torch.cat((image1,image2))
    model[input_tensor1, input_tensor2]


def main():
    st.title('Signature Equation Classification')
    model = load_model()
    categories = load_labels()
    image1 = load_image1()
    image2 = load_image2()
    result = st.button('Check similarity')
    if result:
        st.write('Calculating results...')
        predict(model, categories, image1, image2)


if __name__ == '__main__':
    main()