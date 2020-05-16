import os, glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base_path", default="E:/MassimoMartini/Dataset PoinClouds/Dataset totale con 3d features/preprocessed/Trompone_TEST_VAL_TRAIN/", help="path for the dataset")
args = parser.parse_args()

base_path = args.base_path
data_path = "data/"
out_base_path = data_path + "dataset/"
meta_path = data_path + "meta/"

#classes = ["arc", "column", "moulding", "floor", "door-window", "wall", "stairs", "vault", "roof", "other"]
with open(base_path+"class_names.txt", "r") as fr:
    classes=[]
    for l in fr:
        classes.append(l.strip())
print("Classses:",classes)

with open(base_path+"structure.txt", "r") as fr:
    structure={}
    for l in fr:
        split = l.strip().split(" ")
        for i,s in enumerate(split):
            if s in ["x","y","z","r","g","b","label"]:
                structure[s]=i
        break
print("Structure:",structure)

os.makedirs(out_base_path,exist_ok=True)

files = glob.glob(base_path+"scena*")
anno_paths=[]
for i,f in enumerate(files):
    #x y z r g b f1 f2 f3 f4 f5 f6 label nx ny nz
    print(os.path.basename(f))
    with open(f,"r") as fr:
        anno_path="Area_{}/room_1/Annotations".format(i+1)
        anno_paths.append(anno_path)
        out_path = out_base_path + anno_path
        os.makedirs(out_path, exist_ok=True)
        files=[]
        for j in range(len(classes)):
            file = out_path + "/"+classes[j]+"_labels.txt"
            files.append(open(file,"w"))
        k=0
        for l in fr:
            if k%100000==0: print(k)
            k+=1
            feats = l.strip().split(" ")
            x = feats[structure["x"]]
            y = feats[structure["y"]]
            z = feats[structure["z"]]
            r = feats[structure["r"]]
            g = feats[structure["g"]]
            b = feats[structure["b"]]
            label = int(float(feats[structure["label"]]))
            new_line = "{} {} {} {} {} {}\n".format(x,y,z,r,g,b)
            files[label].write(new_line)
        for j in range(len(classes)):
            files[j].close()

os.makedirs(meta_path,exist_ok=True)
#Creo class_names.txt
with open(meta_path+"class_names.txt", "w") as fw:
    for c in classes[:-1]:
        fw.write(c+"\n")
    fw.write(classes[-1])

#Creo anno_paths.txt
with open(meta_path+"anno_paths.txt", "w") as fw:
    for a in anno_paths[:-1]:
        fw.write(a+"\n")
    fw.write(anno_paths[-1])

print("Done!!")
