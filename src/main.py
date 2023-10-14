from tkinter import filedialog as fd
from PIL import Image, ImageTk
import customtkinter as ctk

import numpy as np
import cv2

from model import inference

selected_img = None

ctk.set_appearance_mode("System")  # Modes: system (default), light, dark
ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

def show_page(page_name):
    for frame in frames.values():
        frame.grid_remove()
    frames[page_name].grid(row=0, column=0, padx=10, pady=10)

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
        imgloaded = Image.open(filename)
        img = ctk.CTkImage(imgloaded, size=(400, 400)) #ImageTk.PhotoImage(resized_img)
        label.configure(image=img)
        label.image = img
        label2.configure(image=img)
        label2.image = img

        global selected_img
        selected_img = imgloaded

        show_page("page1")

def process_and_open_page():
    #run inference on neural net
    img = np.array(selected_img)

    img = np.resize(img, (480, 640, 3))
    print(img.shape)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    out = inference.infer_CNN(gray)


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

answerLablou = ctk.CTkLabel(frames['page2'], text = "Ktery lom to je", font=("Arial", 25))
# Main frame
ButtonFont = ctk.CTkFont(family='Arial')
open_button = ctk.CTkButton(
    frames["main_page"],
    text='Otevřít obrázek',
    command=select_file
)

open_button.pack(padx=160, pady=230)


# Img frame
label = ctk.CTkLabel(frames["page1"], text='')
label.pack()
label2 = ctk.CTkLabel(frames["page2"], text='')
label2.pack()
answerLablou.pack(pady = 20)
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
beck_button.pack(side="left", padx=5, pady=10)
process_button.pack(side="right", padx=5, pady=10)
show_page("main_page")

root.mainloop()