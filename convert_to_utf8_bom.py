import os

# 指定你要处理的根目录
target_dir = r"thirdparties/SDL"

# 要处理的文件扩展名
extensions = {'.cpp', '.hpp', '.h', '.c', '.cc', '.hh'}

def convert_file_to_utf8_bom(filepath):
    try:
        # 以二进制方式读取原始内容（尝试自动解码）
        with open(filepath, 'rb') as f:
            raw = f.read()

        # 尝试用多种编码方式读取原始文件
        for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'utf-16', 'latin1']:
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"❌ 无法识别编码: {filepath}")
            return

        # 写入 UTF-8 with BOM 编码
        with open(filepath, 'w', encoding='utf-8-sig') as f:
            f.write(text)
        print(f"✅ 已转换: {filepath}")
    except Exception as e:
        print(f"❌ 处理失败: {filepath}, 错误: {e}")

# 遍历目录并处理文件
for root, dirs, files in os.walk(target_dir):
    for filename in files:
        if os.path.splitext(filename)[1].lower() in extensions:
            full_path = os.path.join(root, filename)
            convert_file_to_utf8_bom(full_path)
