from tkinter import ttk
from tkinter import filedialog as fd
from PIL import Image, ImageTk
from tkinter import Tk
from time import sleep
import tkinter.font as font


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
        resized_img = imgloaded.resize((400, 400))
        img = ImageTk.PhotoImage(resized_img)
        label.config(image=img)
        label.image = img
        label2.config(image=img)
        label2.image = img
        show_page("page1")




def process_and_open_page2():
    answerLablou.config(text="neni to lom je to hovno!")
    show_page("page2")


root = Tk()
root.resizable(width=False, height=False)
root.title("Перехід між сторінками")
root.geometry("420x500")

frames = {
    "main_page": ttk.Frame(root),
    "page1": ttk.Frame(root),
    "page2": ttk.Frame(root)
}

answerLablou = ttk.Label(frames['page2'], text = "Ktery lom to je", font=("Arial", 25))
# Main frame
ButtonFont = font.Font(family='Arial')
open_button = ttk.Button(
    frames["main_page"],
    text='Open an Image',
    command=select_file
)

open_button.pack(padx=160, pady=230)


# Img frame
label = ttk.Label(frames["page1"])
label.pack()
label2 = ttk.Label(frames["page2"])
label2.pack()
answerLablou.pack(pady = 20)
process_button = ttk.Button(
    frames["page1"],
    text='Process',
    command=process_and_open_page2
)
process_button.pack(side="left", padx=5, pady=10)
beck_button = ttk.Button(
    frames["page1"],
    text='Back',
    command=lambda: show_page("main_page")
)
beck_button.pack(side="left", padx=5, pady=10)

show_page("main_page")



root.mainloop()
