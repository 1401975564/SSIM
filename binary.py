import os
import cv2
import numpy as np
import torch
import time
import signal
import random
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

signal.signal(signal.SIGALRM, timeout_handler)

def crop_and_process_image(image_path, data_dir, output_dir, mask_value, max_crop_size=(32, 32), num_crops=4):
    print(f"开始处理图片: {image_path}")

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    
    if h < 50 or w < 50:
        print(f"图片尺寸小于最小裁剪尺寸，跳过处理: {image_path}")
        return

    crops = []
    for _ in range(num_crops):
        crop_height = random.randint(10, min(h, max_crop_size[0]))
        crop_width = random.randint(10, min(w, max_crop_size[1]))
        crop_size = (crop_height, crop_width)
      
        top = np.random.randint(0, h - crop_size[0] + 1) if h > crop_size[0] else 0
        left = np.random.randint(0, w - crop_size[1] + 1) if w > crop_size[1] else 0
        crop = image[top:top + crop_size[0], left:left + crop_size[1]]
        crops.append(crop)

    for i, crop in enumerate(crops):
        resized_crop = cv2.resize(crop, (244, 244), interpolation=cv2.INTER_CUBIC)
      
        threshold = 1
        closest_threshold = threshold
        closest_diff = float('inf')
        
        for threshold in range(1, 256):
            _, binary = cv2.threshold(resized_crop, threshold, 255, cv2.THRESH_BINARY)
            black_ratio = np.sum(binary == 0) / (binary.shape[0] * binary.shape[1])
            diff = abs(black_ratio - mask_value)
            
            if diff < closest_diff:
                closest_diff = diff
                closest_threshold = threshold
            
            if black_ratio == mask_value:
                break
        
        _, binary = cv2.threshold(resized_crop, closest_threshold, 255, cv2.THRESH_BINARY)
        black_ratio = np.sum(binary == 0) / (binary.shape[0] * binary.shape[1])

        while black_ratio > mask_value:
            black_pixels = np.column_stack(np.where(binary == 0))
            random_pixel = black_pixels[random.randint(0, len(black_pixels) - 1)]
            binary[random_pixel[0], random_pixel[1]] = 255
            black_ratio = np.sum(binary == 0) / (binary.shape[0] * binary.shape[1])

        relative_dir = os.path.relpath(os.path.dirname(image_path), data_dir)
        output_subdir = os.path.join(output_dir, relative_dir)
        os.makedirs(output_subdir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_subdir, f"{base_name}_crop{i}.bmp")
        cv2.imwrite(output_path, binary)
        print(f"保存裁剪后的图片: {output_path}")

    print(f"图片处理完成: {image_path}")


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        

def main(start_dir, end_dir):
    data_dir = '/root/dataset/train/'
    output_dir = '/root/output0.1'
    mask_value = 0.1
    processed_images_set = set()

    for subdir, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".jpg"):
                processed_image_path = os.path.join(subdir, file)
                original_image_path = processed_image_path.replace(output_dir, data_dir).replace('_crop', '')
                processed_images_set.add(original_image_path)

    total_images = sum([len(files) for r, d, files in os.walk(data_dir)])
    processed_images = 0
    skipped_images = 0

    total_start_time = time.time()
    for i, (root, dirs, files) in enumerate(os.walk(data_dir)):
        if i < start_dir or i >= end_dir:
            continue
        dir_start_time = time.time()
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                if image_path in processed_images_set:
                    print(f"图片已处理，跳过: {image_path}")
                    skipped_images += 1
                    continue
                try:
                    with time_limit(5):
                        crop_and_process_image(image_path, data_dir, output_dir, mask_value)
                    processed_images += 1
                except TimeoutException as e:
                    print(f"处理{image_path}超时，跳过该图像")
                    skipped_images += 1
                    continue
        dir_end_time = time.time()
        print(f"子文件夹处理完成，耗时：{dir_end_time - dir_start_time}秒")
    total_end_time = time.time()
    print(f"所有图片处理完成，总耗时：{total_end_time - total_start_time}秒")

    if processed_images + skipped_images == total_images:
        print("所有图片全部处理完毕，程序结束")


if __name__ == '__main__':
    main(1,4)
