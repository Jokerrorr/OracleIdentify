"""
后续用于的加载模型进行测试与GUI的实现
"""
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

from PIL import Image, ImageDraw, ImageTk
import os
from datetime import datetime
import tkinter.messagebox as messagebox
from package.predict import predict
import torch
from resnet18.net import net
from Res2Net.Res2Net import res2net50


class HandwritingBoard:
    def __init__(self, master):
        self.canvas_length = 600
        self.master = master
        self.master.title("手写板")
        self.canvas = tk.Canvas(self.master, width=self.canvas_length, height=self.canvas_length, bg="white")
        self.canvas.pack(side="left", fill=tk.BOTH, expand=True)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.save_button = tk.Button(self.master, text="保存绘画", command=self.save_image, )
        self.save_button.pack(expand=False)

        self.undo_button = tk.Button(self.master, text="撤销重绘", command=self.undo)
        self.undo_button.pack()

        self.submit_button = tk.Button(self.master, text="识别绘图", command=self.submit)
        self.submit_button.pack()

        self.choose_button = tk.Button(self.master, text="本地上传", command=self.choose_image)
        self.choose_button.pack()

        self.image = Image.new("RGB", (self.canvas_length, self.canvas_length), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None

        self.save_folder = "saved image"  # 保存文件夹名称

    def draw(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=5)
            self.draw.line([(self.last_x, self.last_y), (x, y)], fill="black", width=5)
        self.last_x = x
        self.last_y = y

    def reset(self, event):
        self.last_x = None
        self.last_y = None

    def undo(self):
        self.master.destroy()
        root = tk.Tk()
        app = HandwritingBoard(root)
        root.mainloop()

    def save_image(self):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file_name = f"{current_time}.png"
        file_path = os.path.join(self.save_folder, file_name)
        self.image.save(file_path)
        messagebox.showinfo("提示", f"图像已保存为: {file_path}")

    def submit(self):
        self.save_image()
        image_dir = "saved image\\" + os.listdir("saved image")[-1]
        topk = 10
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = res2net50()
        checkpoint_dir = "Res2Net/save_train_data/best.pth"
        checkpoint = torch.load(checkpoint_dir, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()

        class_names_pkl_dir = "Res2Net/save_train_data/class_names.pkl"
        pred = predict(image_dir, topk, model, class_names_pkl_dir, device)

        prediction_window = tk.Toplevel(self.master)
        prediction_window.title("Prediction Results")

        prediction_table = ttk.Treeview(prediction_window, columns=('Class', 'Probability'))
        prediction_table.heading('#0', text='Rank')
        prediction_table.heading('Class', text='Class')
        prediction_table.heading('Probability', text='Probability')
        prediction_table.pack(pady=10)

        for idx, (class_name, probability) in enumerate(pred, start=1):
            prediction_table.insert("", 'end', text=str(idx), values=(class_name, f"{probability:.2f}"))

    def choose_image(self):
        file_path = filedialog.askopenfilename(initialdir="/", title="选择图片文件", filetypes=(
            ("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")))
        if file_path:
            # 清除之前的绘画
            self.canvas.delete("all")
            # 加载用户选择的图片文件
            image = Image.open(file_path)
            # 在画布上显示图片
            self.canvas.image = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.canvas.image)
        image_dir = file_path
        topk = 10
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = res2net50()
        checkpoint_dir = "Res2Net/save_train_data/best.pth"
        checkpoint = torch.load(checkpoint_dir, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()

        class_names_pkl_dir = "Res2Net/save_train_data/class_names.pkl"
        pred = predict(image_dir, topk, model, class_names_pkl_dir, device)

        prediction_window = tk.Toplevel(self.master)
        prediction_window.title("Prediction Results")

        prediction_table = ttk.Treeview(prediction_window, columns=('Class', 'Probability'))
        prediction_table.heading('#0', text='Rank')
        prediction_table.heading('Class', text='Class')
        prediction_table.heading('Probability', text='Probability')
        prediction_table.pack(pady=10)

        for idx, (class_name, probability) in enumerate(pred, start=1):
            prediction_table.insert("", 'end', text=str(idx), values=(class_name, f"{probability:.2f}"))

def main():
    root = tk.Tk()
    app = HandwritingBoard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
