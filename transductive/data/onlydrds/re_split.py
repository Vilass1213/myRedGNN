import random
import os


# 读取文件内容并返回列表
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


# 保存列表到文件
def save_to_file(data, folder, split_name):
    os.makedirs(folder, exist_ok=True)  # 创建文件夹（如果不存在）
    with open(os.path.join(folder, f'{split_name}.txt'), 'w', encoding='utf-8') as f:
        f.writelines(data)


# 合并文件
def merge_files(file_paths):
    data = []
    for file_path in file_paths:
        data.extend(read_file(file_path))
    return data


# 划分数据集
def split_dataset(data, facts_ratio=0.6, train_ratio=0.2, valid_ratio=0.1, test_ratio=0.1):
    random.shuffle(data)

    total = len(data)
    facts_size = int(facts_ratio * total)
    train_size = int(train_ratio * total)
    valid_size = int(valid_ratio * total)

    facts_data = data[:facts_size]
    train_data = data[facts_size:facts_size + train_size]
    valid_data = data[facts_size + train_size:facts_size + train_size + valid_size]
    test_data = data[facts_size + train_size + valid_size:]

    return facts_data, train_data, valid_data, test_data


# 主函数
def main():
    # 文件路径
    file_paths = ['facts.txt', 'train.txt', 'valid.txt', 'test.txt']

    # 合并所有文件
    all_data = merge_files(file_paths)

    # 获取上一级目录
    output_base_path = os.path.abspath('..')  # 获取当前目录的父级目录

    # 重新划分四次数据集
    for i in range(4):
        facts_data, train_data, valid_data, test_data = split_dataset(all_data)

        # 创建新的文件夹
        folder = os.path.join(output_base_path, f'split_{i + 1}')

        # 保存新的数据集到指定文件夹
        save_to_file(facts_data, folder, 'facts')
        save_to_file(train_data, folder, 'train')
        save_to_file(valid_data, folder, 'valid')
        save_to_file(test_data, folder, 'test')

    print(f"四次数据集划分并保存完成，所有数据保存在上一级目录下。")


# 运行主程序
if __name__ == '__main__':
    main()
