"""
后续用于的加载模型进行测试与GUI的实现
"""
import tkinter as tk
from PIL import Image, ImageDraw
import os
from datetime import datetime
import tkinter.messagebox as messagebox
from predict import predict


class HandwritingBoard:
    def __init__(self, master):
        self.canvas_length = 600
        self.master = master
        self.master.title("手写板")
        self.canvas = tk.Canvas(self.master, width=self.canvas_length, height=self.canvas_length, bg="light yellow")
        self.canvas.pack(side="left", fill=tk.BOTH, expand=True)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.save_button = tk.Button(self.master, text="保存绘画", command=self.save_image, )
        self.save_button.pack(expand=False)

        self.undo_button = tk.Button(self.master, text="撤销重绘", command=self.undo)
        self.undo_button.pack()

        self.submit_button = tk.Button(self.master, text="识别绘图", command=self.submit)
        self.submit_button.pack()

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
        image_path = "saved image\\" + os.listdir("saved image")[-1]
        messagebox.showinfo("结果", f"图像识别为: {predict(image_path)}")


def main():
    root = tk.Tk()
    app = HandwritingBoard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
