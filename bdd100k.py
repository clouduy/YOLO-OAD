import json
import os


# 第42行改为 filepath = os.path.join(readpath, file)
# 第10行改为 write = open(writepath + os.sep + "%s.txt" % info["name"], 'w')
def bdd2yolo5(categorys, jsonFile, writepath):
    strs = ""
    f = open(jsonFile)
    info = json.load(f)
    write = open(writepath + os.sep + "%s.txt" % info["name"], 'w')
    # print(len(info))
    # print(info["name"])
    write = open(writepath + "%s.txt" % info["name"], 'w')
    for obj in info["frames"]:
        # print(obj["objects"])
        for objects in obj["objects"]:
            # print(objects)
            if objects["category"] in categorys:
                dw = 1.0 / 1280
                dh = 1.0 / 720
                strs += str(categorys.index(objects["category"]))
                strs += " "
                strs += str(((objects["box2d"]["x1"] + objects["box2d"]["x2"]) / 2.0) * dw)[0:8]
                strs += " "
                strs += str(((objects["box2d"]["y1"] + objects["box2d"]["y2"]) / 2.0) * dh)[0:8]
                strs += " "
                strs += str(((objects["box2d"]["x2"] - objects["box2d"]["x1"])) * dw)[0:8]
                strs += " "
                strs += str(((objects["box2d"]["y2"] - objects["box2d"]["y1"])) * dh)[0:8]
                strs += "\n"
        write.writelines(strs)
        write.close()
        print("%s has been dealt!" % info["name"])


if __name__ == "__main__":
    ####################args#####################
    categorys = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign",
                 "train"]  # 自己需要从BDD数据集里提取的目标类别 "person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign","train"
    readpath = "/home/lvyong/BDD100K/bdd100k/labels/100k/val/"  # BDD数据集标签读取路径，这里需要分三次手动去修改train、val、test的地址
    writepath = "/home/lvyong/BDD100K/bdd100k/labels/100k/val_txt/"  # BDD数据集转换后的标签保存路径

    fileList = os.listdir(readpath)

    # print(fileList)
    for file in fileList:
        filepath = os.path.join(readpath, file)
        print(file)
        filepath = readpath + file
        bdd2yolo5(categorys, filepath, writepath)