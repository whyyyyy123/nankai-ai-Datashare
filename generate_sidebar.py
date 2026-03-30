import os
import urllib.parse

# 配置区
GITHUB_RAW_BASE = "https://nkai-share.github.io/nankai-ai-Datashare/"
EXCLUDE_DIRS = ['.git', '.github', 'docs', 'assets', '.vscode', 'node_modules']
EXCLUDE_FILES = ['README.md', 'SUMMARY.md', '_sidebar.md', '_navbar.md', 'generate_sidebar.py', '.nojekyll']
SIDEBAR_PATH = '_sidebar.md'

def generate_sidebar():
    sidebar_content = [
        "* [🏠 首页](/) \n",
    ]
    
    for root, dirs, files in os.walk('.', topdown=True):
        dirs[:] = sorted([d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')])
        rel_path = os.path.relpath(root, '.')
        if rel_path == '.': continue
        level = rel_path.count(os.sep)
        indent = "  " * level
        folder_name = os.path.basename(root)
        sidebar_content.append(f"{indent}* **{folder_name}**\n")
        
        for file in sorted(files):
            if file in EXCLUDE_FILES or file.startswith('.'): continue
            ext = os.path.splitext(file)[1].lower()
            raw_path = os.path.join(rel_path, file).replace('\\', '/')
            url_path = urllib.parse.quote(raw_path)
            
            if ext != '.md':
                sidebar_content.append(f"{indent}  * [{file}]({url_path} ':ignore')\n")
            else:
                sidebar_content.append(f"{indent}  * [{file}]({url_path})\n")

    with open(SIDEBAR_PATH, 'w', encoding='utf-8') as f:
        f.writelines(sidebar_content)
    print(f"✅ 侧边栏已更新。")

if __name__ == "__main__":
    generate_sidebar()