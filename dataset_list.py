import os

def get_list(datasets):
    for dataset in datasets:
        imgs_dir = 'data/'+ dataset + "/Imgs/"
        imgs = os.listdir(imgs_dir)
        txt_dir = os.path.dirname(os.path.dirname(imgs_dir)) + "/" + mode+".txt"
        with open(txt_dir,"w") as f:
            for img in imgs:
                f.write(img.split('.')[0] + "\n")

if __name__ == '__main__':
    datasets = ['CAMO','NC4K','COD10K']
    mode = "test"
    get_list(datasets)
