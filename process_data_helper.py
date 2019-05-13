import random
import tokenization
import jieba
tokenizer = tokenization.FullTokenizer("./bert_model/chinese_L-12_H-768_A-12/vocab.txt")

'''

[PAD]
[UNK]
[CLS]
[SEP]
[MASK]

'''


class DataHelper:
    @staticmethod
    def randoMask(text,num_mask):
        wordlist = text.split()
        len_sen=len(wordlist)
        if len_sen<=5:
            # 长度小于5的句子都不要
            # .3(wordlist)
            return None
        need_mask=set()
        while len(need_mask)<num_mask:
            need_mask.add(random.randint(0,len_sen-1))
        # print(need_mask)
        input_list = []
        output_list = []
        input_mask = []
        for i,word in enumerate(wordlist):
            tmp_list=tokenizer.tokenize(word)
            len_tmp_list=len(tmp_list)
            output_list.extend(tmp_list)
            if i in need_mask:
                input_list.extend(["[MASK]"]*len_tmp_list)
                input_mask.extend([1]*len_tmp_list)
            else:
                input_list.extend(tmp_list)
                input_mask.extend([0]*len_tmp_list)
        return input_list,output_list,input_mask













