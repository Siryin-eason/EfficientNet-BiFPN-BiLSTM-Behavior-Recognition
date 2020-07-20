"""
author: Sir-yin-einson
motto: No pain, no gain
"""
import pickle   #首先导入这个库，没有安装的话，自行百度，很简单
dict_data = ['diet', 'drinkwater','lying','standing','walking']#行为类别label
with open("./label.pkl", 'wb') as fo:     # 将数据写入pkl文件
    pickle.dump(dict_data, fo)

