from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tokenization

test_token=tokenization.FullTokenizer("./bert_model/chinese_L-12_H-768_A-12/vocab.txt")






print(test_token.tokenize("目的 观察 太宁 膏 、 太宁 栓 联合 应用 在 防治 肛肠病 术后 便秘 、 疼痛 和术区 出血 的 疗效 。"))
