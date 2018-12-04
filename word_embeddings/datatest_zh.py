import json

id2word_zh_dic = {}

with open('id2word_zh_dic.json') as f:
    id2word_zh_dic = json.load(f)

print(id2word_zh_dic['3'])
