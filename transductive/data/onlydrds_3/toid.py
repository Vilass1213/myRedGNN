import os

# 读取文件并返回每一行
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

# 写入文件
def write_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(lines) + '\n')

# 给实体和关系分配ID
def assign_ids(items):
    return {item: idx for idx, item in enumerate(items)}

# 转换三元组为ID形式，输出顺序为(e1, e2, rel)
def convert_triples_to_ids(triples, entity2id, relation2id):
    converted_triples = []
    for line_num, line in enumerate(triples, start=1):  # 添加行号方便调试
        parts = line.strip().split(' ')  # 修改为按空格分隔
        if len(parts) != 3:  # 检查是否为合法的三元组
            print(f"警告：第 {line_num} 行格式不正确，跳过该行：{line}")
            continue
        h, r, t = parts
        if h not in entity2id or t not in entity2id or r not in relation2id:
            print(f"警告：第 {line_num} 行包含未知的实体或关系，跳过该行：{line}")
            continue
        converted_triples.append(f"{entity2id[h]}\t{entity2id[t]}\t{relation2id[r]}")  # 调整为(e1, e2, rel)
    return converted_triples

def main():
    # 输入文件路径
    files = {
        "entities": "entities.txt",
        "relations": "relations.txt",
        "facts": "facts.txt",
        "train": "train.txt",
        "valid": "valid.txt",
        "test": "test.txt"
    }

    # 输出文件存储目录（指定路径）
    output_dir = r"D:\github\OpenKE-OpenKE-PyTorch\benchmarks\onlydrds_3"
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

    # 读取实体和关系
    entities = read_file(files["entities"])
    relations = read_file(files["relations"])

    # 分配ID并保存到entity2id和relation2id
    entity2id = assign_ids(entities)
    relation2id = assign_ids(relations)

    write_file(
        os.path.join(output_dir, "entity2id.txt"),
        [f"{len(entities)}"] + [f"{entity}\t{id}" for entity, id in entity2id.items()]
    )
    write_file(
        os.path.join(output_dir, "relation2id.txt"),
        [f"{len(relations)}"] + [f"{relation}\t{id}" for relation, id in relation2id.items()]
    )

    # 读取并转换三元组
    facts = read_file(files["facts"])
    train = read_file(files["train"])
    valid = read_file(files["valid"])
    test = read_file(files["test"])

    # 转换并合并facts和train为train2id
    train2id = convert_triples_to_ids(facts + train, entity2id, relation2id)
    write_file(os.path.join(output_dir, "train2id.txt"), [f"{len(train2id)}"] + train2id)

    # 转换valid和test
    valid2id = convert_triples_to_ids(valid, entity2id, relation2id)
    write_file(os.path.join(output_dir, "valid2id.txt"), [f"{len(valid2id)}"] + valid2id)

    test2id = convert_triples_to_ids(test, entity2id, relation2id)
    write_file(os.path.join(output_dir, "test2id.txt"), [f"{len(test2id)}"] + test2id)

    print(f"所有文件已成功生成并存储到目录：{output_dir}")

if __name__ == "__main__":
    main()
