import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
import tkinter.filedialog as tkfd
from pathlib import Path
import pandas as pd
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import sys
from PIL import Image
import cv2
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../../')
from lib.Allread_scanner import allread_scanner

class GUI:
    def __init__(self):
        self.color = 'WHITE'
        self.cvv = None
        #self.create_reference = CreateReference()
        #self.accumulate_intensity = AccumulateIntensity()
        self.root = tk.Tk()
        #cwd = os.path.dirname(os.path.abspath(__file__))
        #self.iconfile = os.path.join(cwd, 'icon/favicon.ico')
        #self.root.iconbitmap(default=self.iconfile)
        self.nb = ttk.Notebook(width=900, height=300)
        self.tab1 = tk.Frame(self.nb, bg = self.color)
        self.tab2 = tk.Frame(self.nb, bg = self.color)
        self.nb.add(self.tab1, text='光源の強度分布,イメージング')
        self.nb.add(self.tab2, text='分割測定時の画像再構成')
        self.nb.pack(expand=1, fill='both')
        self.root.title(u"For THz scanner by terasense")
        self.root.resizable(width=0, height=0)
        self.getfile()
        self.image_sec = 3000
        self.normflag = False
        self.save_flag = 0

    def getfile(self):
        fileFrame = tk.LabelFrame(self.tab1, bg=self.color, bd=2, relief="ridge", text="ファイルの選択")
        fileFrame.pack(anchor=tk.W, pady=5)
        saveFrame = tk.LabelFrame(self.tab1, bd=2, bg=self.color, relief="ridge", text="データ保存")
        saveFrame.pack(anchor=tk.W, pady=5)
        analysisFrame = tk.LabelFrame(self.tab1, bd=2, bg=self.color, relief="ridge", text="データ解析")
        analysisFrame.pack(anchor=tk.W, pady=5)
        pictureFrame = tk.LabelFrame(self.tab2, bd=2, bg=self.color, relief="ridge", text="分割画像再構成")
        pictureFrame.pack(anchor=tk.W, pady=5)

        lbl = tk.Label(fileFrame, text='ファイルを選択してください')
        lbl.grid(row=0, column=0)
        lbl = tk.Label(saveFrame, text='保存先を選択してください')
        lbl.grid(row=0, column=0)
        lbl = tk.Label(analysisFrame, text='解析方法を選択してください')
        lbl.grid(row=0, column=0)
        lbl = tk.Label(analysisFrame, text='グラフタイトルを入力してください(ビーム強度，イメージング両方用)')
        lbl.grid(row=1, column=0)
        lbl = tk.Label(analysisFrame, text='イメージング用強度最大値を入力してください')
        lbl.grid(row=2, column=0)
        lbl = tk.Label(analysisFrame, text='イメージング用強度最小値を入力してください')
        lbl.grid(row=3, column=0)
        lbl = tk.Label(pictureFrame, text='再構成用の画像を選択してください')
        lbl.grid(row=0, column=0)
        lbl = tk.Label(pictureFrame, text='画像1')
        lbl.grid(row=1, column=0)
        lbl = tk.Label(pictureFrame, text='画像2')
        lbl.grid(row=2, column=0)
        lbl = tk.Label(pictureFrame, text='保存先フォルダを指定してください')
        lbl.grid(row=3, column=0)
        lbl = tk.Label(pictureFrame, text='2枚の画像のシフト量を指定してください')
        lbl.grid(row=4, column=0)


        # ファイルのテキストボックスを出現させる
        fileEntry = tk.Entry(fileFrame, width=80)
        fileEntry.grid(row=0, column=1)
        # 保存先のテキストボックスを出現させる
        saveEntry = tk.Entry(saveFrame, width=80)  # widthプロパティで大きさを変える
        saveEntry.grid(row=0, column=1)
        # タイトルのテキストボックスを出現させる
        titleEntry = tk.Entry(analysisFrame, width=50)  # widthプロパティで大きさを変える
        titleEntry.insert(tk.END, 'Beam graph')
        titleEntry.grid(row=1, column=1)
        # 最大最小のテキストボックスを出現させる
        MaxEntry = tk.Entry(analysisFrame, width=10)  # widthプロパティで大きさを変える
        MaxEntry.insert(tk.END, u'0.0025')
        MaxEntry.grid(row=2, column=1, sticky=tk.W)
        MinEntry = tk.Entry(analysisFrame, width=10)  # widthプロパティで大きさを変える
        MinEntry.insert(tk.END, u'0')
        MinEntry.grid(row=3, column=1, sticky=tk.W)
        # ファイルのテキストボックスを出現させる
        fileEntry_2 = tk.Entry(pictureFrame, width=80)
        fileEntry_2.grid(row=1, column=1)
        fileEntry_3 = tk.Entry(pictureFrame, width=80)
        fileEntry_3.grid(row=2, column=1)
        # 保存先のテキストボックスを出現させる
        saveEntry_2 = tk.Entry(pictureFrame, width=80)  # widthプロパティで大きさを変える
        saveEntry_2.grid(row=3, column=1)
        # シフト量のテキストボックスを出現させる
        shiftEntry = tk.Entry(pictureFrame, width=10)  # widthプロパティで大きさを変える
        shiftEntry.insert(tk.END, u'-15')
        shiftEntry.grid(row=4, column=1, sticky=tk.W)

        ###関数のところ
        ####ファイル用関数
        def filefolder_button_clicked():
            fileEntry.delete(0, tk.END)
            file_path = tkfd.askopenfilename(title='ファイルを選択してください')
            fileEntry.insert(tk.END, file_path)
        def filefolder_button_clicked_2():
            fileEntry_2.delete(0, tk.END)
            file_path = tkfd.askopenfilename(title='ファイルを選択してください')
            fileEntry_2.insert(tk.END, file_path)
        def filefolder_button_clicked_3():
            fileEntry_3.delete(0, tk.END)
            file_path = tkfd.askopenfilename(title='ファイルを選択してください')
            fileEntry_3.insert(tk.END, file_path)

        ##ファイル削除用関数
        def filefolder_button_clear_clicked():
            fileEntry.delete(0, tk.END)
        def filefolder_button_clear_clicked_2():
            fileEntry_2.delete(0, tk.END)
        def filefolder_button_clear_clicked_3():
            fileEntry_3.delete(0, tk.END)

        ###セーブファイル用関数
        def savefolder_button_clicked():
            saveEntry.delete(0, tk.END)
            save_path = tkfd.askdirectory(title='セーブフォルダを選択してください')
            saveEntry.insert(tk.END, save_path)
        def savefolder_button_clicked_2():
            saveEntry_2.delete(0, tk.END)
            save_path = tkfd.askdirectory(title='セーブフォルダを選択してください')
            saveEntry_2.insert(tk.END, save_path)

        ### セーブファイル削除用関数
        def savefolder_button_clear_clicked():
            saveEntry.delete(0, tk.END)
        def savefolder_button_clear_clicked_2():
            saveEntry_2.delete(0, tk.END)

        def lightsource_beamshape_cilcked():
            if (len(fileEntry.get()) == 0) & (len(saveEntry.get()) == 0):
                messagebox.showerror('エラー', 'フォルダが選択されていません。')
            else:
                try:
                    if len(saveEntry.get()) == 0:
                        self.save_flag = 1
                    else:
                        pass
                    data = allread_scanner(fileEntry.get(), None, titleEntry.get(), self.save_flag, saveEntry.get()).lightsource_beamshape_smoothing()

                    root = tk.Tk()
                    root.title("graph")  # ウインドのタイトル
                    root.geometry("800x700")  # ウインドの大きさ
                    root.withdraw()
                    canvas = FigureCanvasTkAgg(data, master=root)
                    canvas.draw()
                    canvas.get_tk_widget().pack()
                    root.update()
                    root.deiconify()
                    root.after(self.image_sec, lambda: root.destroy())
                    root.mainloop()

                    #messagebox.showinfo('status', '解析，保存が完了しました')

                except FileNotFoundError as e:
                    print(e)

        def intensity_imaging_clicked():
            if (len(fileEntry.get()) == 0) & (len(saveEntry.get()) == 0):
                messagebox.showerror('エラー', 'フォルダが選択されていません。')
            else:
                try:
                    if len(saveEntry.get()) == 0:
                        self.save_flag = 1
                    else:
                        pass
                    data = allread_scanner(fileEntry.get(), None, titleEntry.get(), saveEntry.get(), self.save_flag).visuallization(MinEntry.get(), MaxEntry.get())
                    #messagebox.showinfo('status', '解析，保存が完了しました')

                    root = tk.Tk()
                    root.title("graph")  # ウインドのタイトル
                    root.geometry("800x700")  # ウインドの大きさ
                    root.withdraw()
                    canvas = FigureCanvasTkAgg(data, master=root)
                    canvas.draw()
                    canvas.get_tk_widget().pack()
                    root.update()
                    root.deiconify()
                    root.after(self.image_sec, lambda: root.destroy())
                    root.mainloop()

                except FileNotFoundError as e:
                    print(e)

        def picture_reconstruction_clicked():
            if ((len(fileEntry_2.get()) == 0) or (len(fileEntry_3.get()) == 0)) & (len(saveEntry_2.get()) == 0):
                messagebox.showerror('エラー', 'フォルダが選択されていません。')
            else:
                try:
                    if len(saveEntry_2.get()) == 0:
                        self.save_flag = 1
                    else:
                        pass
                    data = allread_scanner(fileEntry_2.get(), fileEntry_3.get(), titleEntry.get(), saveEntry_2.get(), self.save_flag).picture_reconstruction(shiftEntry.get())
                    # messagebox.showinfo('status', '解析，保存が完了しました')

                except FileNotFoundError as e:
                    print(e)

        ##タブ1用
        self.folderbutton = tk.Button(fileFrame, text='Select', command=filefolder_button_clicked)
        self.folderbutton.grid(row=0, column=2)
        self.folder_button_clear = tk.Button(fileFrame, text='Clear', command=filefolder_button_clear_clicked)
        self.folder_button_clear.grid(row=0, column=3)

        self.savefolderbutton = tk.Button(saveFrame, text='Select', command=savefolder_button_clicked)
        self.savefolderbutton.grid(row=0, column=2)
        self.savefolderbutton_clear = tk.Button(saveFrame, text='Clear', command=savefolder_button_clear_clicked)
        self.savefolderbutton_clear.grid(row=0, column=3)

        self.beamintensity_button = tk.Button(analysisFrame, text='ビームの強度分布グラフ', command=lightsource_beamshape_cilcked)
        self.beamintensity_button.grid(row=0, column=1)

        self.intensity_imaging_button = tk.Button(analysisFrame, text='THz強度イメージング', command=intensity_imaging_clicked)
        self.intensity_imaging_button.grid(row=0, column=2)

        ###タブ2用
        self.folderbutton_2 = tk.Button(pictureFrame, text='Select', command=filefolder_button_clicked_2)
        self.folderbutton_2.grid(row=1, column=2)
        self.folder_button_2_clear = tk.Button(pictureFrame, text='Clear', command=filefolder_button_clear_clicked_2)
        self.folder_button_2_clear.grid(row=1, column=3)
        self.folderbutton_3 = tk.Button(pictureFrame, text='Select', command=filefolder_button_clicked_3)
        self.folderbutton_3.grid(row=2, column=2)
        self.folder_button_3_clear = tk.Button(pictureFrame, text='Clear', command=filefolder_button_clear_clicked_3)
        self.folder_button_3_clear.grid(row=2, column=3)

        self.savefolderbutton_2 = tk.Button(pictureFrame, text='Select', command=savefolder_button_clicked_2)
        self.savefolderbutton_2.grid(row=3, column=2)
        self.savefolderbutton_2_clear = tk.Button(pictureFrame, text='Clear', command=savefolder_button_clear_clicked_2)
        self.savefolderbutton_2_clear.grid(row=3, column=3)

        self.intensity_imaging_button = tk.Button(pictureFrame, text='画像再構成開始', command=picture_reconstruction_clicked)
        self.intensity_imaging_button.grid(row=5, column=1)

class Main:
    def __init__(self):
        self.gui=GUI()
        self.gui.root.mainloop()


if __name__=="__main__":
    Main()
