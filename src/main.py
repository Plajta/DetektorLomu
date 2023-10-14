from tkinter import filedialog as fd
from PIL import Image, ImageTk
import customtkinter as ctk
from model.process import Loader
import os
import numpy as np
import cv2
from model import inference
selected_img = None

ld = Loader()

open_button = None

ctk.set_appearance_mode("System")  # Modes: system (default), light, dark
ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

def show_page(page_name):
    for frame in frames.values():
        frame.grid_remove()
    frames[page_name].grid(row=0, column=0, padx=10, pady=10)



def process_and_open_page():
    # run inference on neural net
    match combobox.get():
        case "CNN":
            npimg = ld.resizing(selected_img)
            print(npimg.shape)
            out = inference.infer_CNN(npimg)
            answerLablou.configure(text=out)
            show_page("page2")
        case "KNNh":
            npimg = ld.resizing(selected_img)
            print(npimg.shape)
            out = inference.infer_KNN_hist(npimg)
            answerLablou.configure(text=out)
            show_page("page2")
        case "KNNr":
            npimg = ld.resizing(selected_img)
            print(npimg.shape)
            out = inference.infer_KNN_raw(npimg)
            answerLablou.configure(text=out)
            show_page("page2")
        case "SVM":
            npimg = ld.resizing(selected_img)
            print(npimg.shape)
            out = inference.infer_SVM(npimg)
            answerLablou.configure(text=out)
            show_page("page2")
        


root = ctk.CTk()
root.resizable(width=False, height=False)
root.title("Detektor lomů")
root.geometry("420x500")

frames = {
    "main_page": ctk.CTkFrame(root),
    "page1": ctk.CTkFrame(root),
    "page2": ctk.CTkFrame(root)
}

def next_button_command():
    show_page("page1")
    next_button.configure(state='normal')


next_button = ctk.CTkButton(
    frames["main_page"],
    text='Next',
    command=lambda: next_button_command(),
    state='disabled')

def select_file():
    filetypes = (
        ('Image files', '*.png *.jpg *.jpeg *.gif *.bmp'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open an image file',
        initialdir='~',
        filetypes=filetypes)
    
    if filename:
        print(filename)
        imgloaded = Image.open(filename)
        img = ctk.CTkImage(imgloaded, size=(400, 400)) #ImageTk.PhotoImage(resized_img)
        label.configure(image=img)
        label.image = img
        label2.configure(image=img)
        label2.image = img

        global selected_img
        selected_img = cv2.imread(filename)

        show_page("page1")


answerLablou = ctk.CTkLabel(frames['page2'], text = "Ktery lom to je", font=("Arial", 25),  padx=20)
# Main frame

emtrylabel = ctk.CTkLabel(frames["main_page"], text='')
emtrylabel.pack(pady=85)
ButtonFont = ctk.CTkFont(family='Arial')
open_button = ctk.CTkButton(
    frames["main_page"],
    text='Otevřít obrázek',
    command=select_file
)



open_button.pack(padx=130, pady=10)



def combobox_callback(choice):
    print("combobox dropdown clicked:", choice)

combobox = ctk.CTkComboBox(master=frames['main_page'],
                                     values=["CNN", "KNNh", "KNNr", "SVM"],
                                     command=combobox_callback)
combobox.pack(side="top")
combobox.set("CNN")  # set initial value


next_button.pack(pady=10)

emtrylabel = ctk.CTkLabel(frames["main_page"], text='')
emtrylabel.pack(pady=85)


# Img frame
label = ctk.CTkLabel(frames["page1"], text='')
label.pack()
label2 = ctk.CTkLabel(frames["page2"], text='')
label2.pack()
answerLablou.pack(side='right', pady = 20)
process_button = ctk.CTkButton(
    frames["page1"],
    text='Process',
    command=process_and_open_page
)

beck_button = ctk.CTkButton(
    frames["page1"],
    text='Back',
    command=lambda: show_page("main_page")
)
beck_button1 = ctk.CTkButton(
    frames["page2"],
    text='Back',
    command=lambda: show_page("main_page")
)
beck_button1.pack(side="left", padx=5, pady=10)
beck_button.pack(side="left", padx=5, pady=10)
process_button.pack(side="right", padx=5, pady=10)
show_page("main_page")

root.mainloop()