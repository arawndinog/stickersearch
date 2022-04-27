from bs4 import BeautifulSoup
from urllib.request import Request, urlopen, build_opener
import os
import time

hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}

def parse_stick_cloud(url_list_path: str = "config/url_list.txt", output_dir: str = "outputs/") -> None:
    url_list_file = open(url_list_path, 'r')
    url_list = url_list_file.readlines()
    url_list_file.close()

    for url in url_list:
        sticker_pack_name = url.split("/")[-1]
        sticker_pack_dir = output_dir + sticker_pack_name + "/"
        if not os.path.isdir(sticker_pack_dir):
            os.mkdir(sticker_pack_dir)

        # opener = build_opener()
        # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        # url_html = opener.open(url).read()
        req = Request(url, headers=hdr)
        url_html = urlopen(req).read()
        soup = BeautifulSoup(url_html, features="html.parser")
        sticker_shared_class = "pa-2 position-relative col-sm-3 col-md-2 col-3"     #"md-layout-item md-size-20"
        sticker_divs = soup.find_all("div", {"class": sticker_shared_class})
        for sticker_div in sticker_divs:
            print(sticker_div)
            sticker_src = ""
            sticker_loaded_div = sticker_div.find('img')
            if sticker_loaded_div:
                sticker_src = sticker_loaded_div.get('src')
                sticker_name = sticker_loaded_div.get('alt').replace(" ","")
            else:
                # doesn't work yet since content is loaded moments after the page finish loading
                sticker_unloaded_class = "v-image__image v-image__image--cover"
                sticker_unloaded_div = sticker_div.find_all("div", {"class": sticker_unloaded_class})
                print(sticker_unloaded_class, sticker_unloaded_div)
            if sticker_src:
                print(sticker_name, sticker_src)
                # sticker_req = Request(sticker_src, headers=hdr)
                # sticker_response = urlopen(sticker_req)
                # with open(sticker_pack_dir + sticker_name + '.webp', 'wb') as out_file:
                #     sticker_data = sticker_response.read()
                #     out_file.write(sticker_data)
        time.sleep(5)


parse_stick_cloud()