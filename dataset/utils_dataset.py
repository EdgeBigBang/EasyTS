import logging
import os
import pathlib
import sys
from urllib import request
logging.basicConfig(level=logging.DEBUG)

def download(url: str, file_path: str) -> None:
    """
    Download a file to the given path.

    :param url: URL to download
    :param file_path: Where to download the content.
    """

    # 打印下载进度
    def progress(count, block_size, total_size):
        progress_pct = min(float(count * block_size) / float(total_size) * 100.0,100.0)
        # 打印到当前的窗口中，这个函数即是print的实现
        sys.stdout.write('\rDownloading {} to {} {:.1f}%'.format(url, file_path, progress_pct))
        # 将缓冲区的内容输出，这里有一个有意思的地方当写入时出现/n 即换行 则会自动将缓冲区的内容输出
        # 这里主要实现一个实时打印进度的效果
        sys.stdout.flush()

    # 检查是否存在文件
    if not os.path.isfile(file_path):
        opener = request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        request.install_opener(opener)
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        f, _ = request.urlretrieve(url, file_path, progress)
        sys.stdout.write('\n')
        sys.stdout.flush()
        file_info = os.stat(f)
        logging.info(f'Successfully downloaded {os.path.basename(file_path)} {file_info.st_size} bytes.')
    else:
        file_info = os.stat(file_path)
        logging.info(f'File already exists: {file_path} {file_info.st_size} bytes.')

def file_name(url: str) -> str:
    """
    Extract file name from url.

    :param url: URL to extract file name from.
    :return: File name.
    """
    return url.split('/')[-1] if len(url) > 0 else ''

if __name__ == '__main__':
    url = "http://gitlab.fei8s.com/tianchengZhang/dastaset-for-timeseries/-/raw/main/MD/swymz.csv"
    file_path = "../MD/PHAR1.csv"
    download(url,file_path)