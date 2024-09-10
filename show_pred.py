import os
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from torchvision import transforms
import torch.nn as nn


# 定义 CNN 模型（与训练时的结构一致）
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 计算卷积层的输出大小
        self.fc1_input_features = self._get_conv_output_size()
        self.fc1 = nn.Linear(self.fc1_input_features, 128)
        self.fc2 = nn.Linear(128, 33)  # 33 classes: 0-9 and a-z excluding i, j, l

    def _get_conv_output_size(self):
        # 创建一个假输入，通过卷积层计算输出大小
        with torch.no_grad():
            x = torch.zeros(1, 1, 40, 30)  # 假设输入图像大小为 40x30
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            return x.numel()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 加载模型
model = CNN()
model.load_state_dict(torch.load('images.pth'))
model.eval()  # 设置模型为评估模式

# 定义图像预处理
transform = transforms.Compose([
    transforms.Grayscale(),  # 转换为灰度图像
    transforms.Resize((40, 30)),  # 调整图像大小为 40x30
    transforms.ToTensor()
])


# 定义 GUI
class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")

        self.folder_index = 0
        self.folders = []  # 保存所有文件夹的列表
        self.images = []  # 保存当前显示的图像
        self.image_labels = []  # 保存图像显示的标签
        self.prediction_labels = []  # 保存图像预测标签的文本

        # 创建布局框架
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(pady=10)

        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(pady=10)

        # 按钮和标签的布局
        self.label = tk.Label(self.top_frame, text="选择一个文件夹来开始", font=("Arial", 14))
        self.label.grid(row=0, column=0, columnspan=4)

        self.select_button = tk.Button(self.top_frame, text="选择文件夹", command=self.load_folder, font=("Arial", 12))
        self.select_button.grid(row=1, column=0, pady=10)

        self.next_button = tk.Button(self.top_frame, text="下一个文件夹", command=self.next_folder, font=("Arial", 12))
        self.next_button.grid(row=1, column=1, pady=10)

    def load_folder(self):
        # 选择文件夹
        self.directory = filedialog.askdirectory()
        if not self.directory:
            return

        # 获取所有子文件夹
        self.folders = [os.path.join(self.directory, f) for f in os.listdir(self.directory) if
                        os.path.isdir(os.path.join(self.directory, f))]
        self.folder_index = 0
        self.show_images()

    def show_images(self):
        if not self.folders:
            return

        # 清除之前的内容
        self.clear_canvas()

        # 获取当前文件夹
        folder = self.folders[self.folder_index]
        self.label.config(text=f"文件夹: {os.path.basename(folder)}")

        # 获取文件夹中的所有图片文件
        image_files = [os.path.join(folder, f) for f in os.listdir(folder) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.images.clear()
        self.image_labels.clear()
        self.prediction_labels.clear()

        # 展示最多四张图片
        for i in range(min(4, len(image_files))):
            image_path = image_files[i]
            image = Image.open(image_path)
            image = transform(image).unsqueeze(0)  # 添加批次维度

            # 预测图像
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)

            # 获取类别名称
            predicted_label = predicted.item()
            class_name = self.get_class_name(predicted_label)

            # 显示图片和预测结果
            img = Image.open(image_path).resize((100, 100))
            img_tk = ImageTk.PhotoImage(img)
            img_label = tk.Label(self.bottom_frame, image=img_tk)
            img_label.image = img_tk  # 保存引用以防止图片被垃圾回收
            img_label.grid(row=0, column=i, padx=10)
            self.image_labels.append(img_label)  # 保存标签以便清除

            pred_label = tk.Label(self.bottom_frame, text=f"预测: {class_name}", font=("Arial", 12))
            pred_label.grid(row=1, column=i, padx=10)
            self.prediction_labels.append(pred_label)

    def clear_canvas(self):
        # 清除所有显示的图片和预测标签
        for img_label in self.image_labels:
            img_label.destroy()
        for pred_label in self.prediction_labels:
            pred_label.destroy()

    def next_folder(self):
        if self.folders and self.folder_index < len(self.folders) - 1:
            self.folder_index += 1
            self.show_images()

    def get_class_name(self, index):
        # 定义类别名称映射
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'k', 'm', 'n',
                   'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        return classes[index] if 0 <= index < len(classes) else "未知"


# 创建 GUI 窗口
root = tk.Tk()
app = ImageClassifierApp(root)
root.mainloop()
