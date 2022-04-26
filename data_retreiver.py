from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import os
import time

url_list_file = open("config/url_list.txt", 'r')
url_list = []
while True:
    url_line = url_list_file.readline()
    if not url_line:
        break
    url_list.append(url_line.strip())

hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}

for url in url_list:
    req = Request(url, headers=hdr)
    sticker_pack_name = url.split("/")[-1]
    sticker_pack_dir = "outputs/" + sticker_pack_name + "/"
    if not os.path.isdir(sticker_pack_dir):
        os.mkdir(sticker_pack_dir)
    url_html = urlopen(req).read()
    soup = BeautifulSoup(url_html, features="html.parser")
    # sticker_divs = soup.findAll("div", {"class": "md-layout-item md-size-20"})
    sticker_divs = soup.findAll("div", {"class": "pa-2 position-relative col-sm-3 col-md-2 col-3"})
    for sticker_div in sticker_divs:
        sticker_src = sticker_div.find('img').get('src')
        sticker_name = sticker_div.find('img').get('alt').replace(" ","")
        print(sticker_src)
        print("  ", sticker_name)
        sticker_req = Request(sticker_src, headers=hdr)
        sticker_response = urlopen(sticker_req)
        with open(sticker_pack_dir + sticker_name + '.webp', 'wb') as out_file:
            sticker_data = sticker_response.read()
            out_file.write(sticker_data)
    time.sleep(6)