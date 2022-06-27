# rsync -a . adrianwong@10.6.64.155:~/Projects/stickersearch/
rsync -a --progress . adrianwong@10.6.126.122:~/Projects/stickersearch/ --exclude "outputs/*"
# rsync -a --progress . adrianwong@10.6.126.122:~/Projects/stickersearch2/ --exclude "outputs/*"