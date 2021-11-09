import jieba
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pyhanlp import *
from gensim.models import KeyedVectors
import numpy as np

jieba.load_userdict('C:/Users/10950/NLP/instruction_classify/myDict.txt')
jieba.suggest_freq("向左",tune=False)
jieba.suggest_freq("向右",tune=True)
jieba.suggest_freq("向前",tune=True)
jieba.suggest_freq("向后",tune=True)

classifier = None # 贝叶斯分类器
vectorizer = None # 词向量模型
# 句法依存树
parser = None  # 句法依存树
# 词向量模型
mv_from_text = None  # TX词向量
mapWordVector = {} # 中文_指令映射表
ListKeyWord = [] # 关键词表

# 将映射信息加载到内存中
def GetMapInfo(textPath):
    global mapWordVector
    global ListKeyWord
    with open(textPath, 'r', encoding="UTF-8") as f:
        strList = []
        strList = f.readlines()
        for i in range(len(strList)):
            strList[i] = strList[i].strip("\n")
            subStrList = strList[i].split(':')
            ListKeyWord.append(subStrList[0])
            mapWordVector[subStrList[0]] = subStrList[1]

# 将分类模型、依存句法分析、以及连续词向量模型加载到内存中
def LoadInit():
    f = open("save_model_BernoulliNB.pickle", "rb")
    global classifier
    classifier = pickle.load(f)
    f.close()

    KBeamArcEagerDependencyParser = JClass(
        'com.hankcs.hanlp.dependency.perceptron.parser.KBeamArcEagerDependencyParser')
    global parser
    parser = KBeamArcEagerDependencyParser()

    global mv_from_text
    mv_from_text = KeyedVectors.load(r"F:\研究课题（自然语言处理）\腾讯词向量\Tencent_AILab_ChineseEmbedding\40_ChineseEmbedding.bin",
                                     mmap='r')

    GetMapInfo("wordMap.txt")  # 加载中文_指令映射表


# 利用完成训练的模型，对中文指令进行分类
def get_average_word2vec(wordList,vector,generate_missing = False,k=200):
    if len(wordList)<1:
        return np.zeros(k) # 返回0向量
    res = []
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in wordList]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in wordList]
    length = len(vectorized)
    summed = np.sum(vectorized , axis = 0)
    averaged = np.divide(summed,length)
    res.append(averaged)
    return res

def classifyInstruction(string):
    if len(string) == 0:
        return -1
    current_segment = jieba.lcut(string)

    if current_segment == '\r\n':
        print(current_segment)
        return -1
    val = classifier.predict(get_average_word2vec(current_segment, mv_from_text))
    return val[0]

#======================================A1类指令=============================================
# A类指令的句法知识库
def refind_range_num(tree, res, prep_conj):
    Key_range = tree.findChildren(prep_conj.get(0), "range")
    Key_dobj = tree.findChildren(prep_conj.get(0), "dobj")
    Key_ccomp = tree.findChildren(prep_conj.get(0), "ccomp")
    if not Key_range.isEmpty():

        num_c = tree.findChildren(Key_range.get(0), "nummod")
        if not num_c.isEmpty():
            res.append(num_c.get(0).LEMMA + Key_range.get(0).LEMMA)
            return res
    if not Key_dobj.isEmpty():
        res.append(Key_dobj.get(0).LEMMA)
        return res
    if not Key_ccomp.isEmpty():
        res.append(Key_ccomp.get(0).LEMMA)
        return res
    return res

def insert_range_num(tree, Key_range, Key_dobj, Key_ccomp, res):
    if not Key_range.isEmpty():
        num_c = tree.findChildren(Key_range.get(0), "nummod")
        if not num_c.isEmpty():
            res.append(num_c.get(0).LEMMA + Key_range.get(0).LEMMA)
            return res
    if not Key_dobj.isEmpty():
        res.append(Key_dobj.get(0).LEMMA)
        return res
    if not Key_ccomp.isEmpty():
        res.append(Key_ccomp.get(0).LEMMA)
        return res
    return res

def extract_verb_object_A1(tree):
    for word in tree.iterator():
        # 对于核心为移动性动词的情况
        res = []
        if word.POSTAG == "VV" and word.DEPREL == "ROOT":
            # 搜素位置状语
            # 搜素范围数词 向左移动五米
            Key_range = tree.findChildren(word, "range")
            # 距离名词有可能会被识别为直接宾语
            Key_dobj = tree.findChildren(word, "dobj")
            # 距离名词有可能会被识别为从句补充
            Key_ccomp = tree.findChildren(word, "ccomp")

            # 查找核心动词下的介词修饰
            prep_p = tree.findChildren(word, "prep")
            # 查找核心动词下的名词主语
            nsubj_n = tree.findChildren(word, "nsubj")
            # 查找核心动词下的状语
            adv_a = tree.findChildren(word, "advmod")
            # 查找核心动词下的地点状语
            loc_l = tree.findChildren(word, "loc")
            # 查找核心动词下的依赖动词
            dep_v = tree.findChildren(word, "dep")
            # 查找核心动词下的情态动词
            mmod_v = tree.findChildren(word, "mmod")

            # 对于特殊的情况，核心动词识别出现问题，朝向性介词被识别为核心动词
            prep_dobj = tree.findChildren(word, "dobj")  # 提取直接宾语
            prep_conj = tree.findChildren(word, "conj")  # 提取并列谓语

            # 若核心动词下的子节点为地点状语
            if not loc_l.isEmpty():
                # print("%s = %s"%(word.LEMMA , loc_l.get(0).LEMMA))
                res.append(word.LEMMA)
                res.append(loc_l.get(0).LEMMA)
                res = insert_range_num(tree, Key_range, Key_dobj, Key_ccomp, res)
                return res
            # 若核心动词下的子结点为依赖动词
            if not dep_v.isEmpty() and dep_v.get(0).POSTAG == "VV":
                # print("%s = %s"%(word.LEMMA , dep_v.get(0).LEMMA))
                # 若依赖动词的长度为2，则依赖动词的由介词与方向词组成，提取方向助词
                if len(dep_v.get(0).LEMMA) == 2:
                    res.append(word.LEMMA)
                    res.append(dep_v.get(0).LEMMA[1])
                    res = insert_range_num(tree, Key_range, Key_dobj, Key_ccomp, res)
                elif not Key_dobj.isEmpty():
                    res.append(dep_v.get(0).LEMMA)
                    res.append(Key_dobj.get(0).LEMMA)

                else:
                    # 若依赖动词的长度为1，则继续向下判断，找到依赖动词下的直接宾语即为方向词
                    dep_dobj = tree.findChildren(dep_v.get(0), "dobj")
                    if not dep_dobj.isEmpty():
                        res.append(word.LEMMA)
                        res.append(dep_dobj.get(0).LEMMA[0])
                        res = insert_range_num(tree, Key_range, Key_dobj, Key_ccomp, res)
                return res
            # 若核心动词下的子结点为介词修饰
            if not prep_p.isEmpty():
                pobj_n = tree.findChildren(prep_p.get(0), "pobj")
                # 将得到的宾语跟核心动词进行输出
                # print("%s = %s"%(word.LEMMA ,pobj_n.get(0).LEMMA ))
                res.append(word.LEMMA)
                res.append(pobj_n.get(0).LEMMA)
                res = insert_range_num(tree, Key_range, Key_dobj, Key_ccomp, res)
                return res

            # 若核心动词下的子结点为方向状语，例如：向前，向后
            if not adv_a.isEmpty():
                #  将得到的方向状语跟核心名称进行输出
                # print("%s = %s"%(word.LEMMA ,adv_a.get(0).LEMMA ))
                res.append(word.LEMMA)
                if len(adv_a.get(0).LEMMA) < 2:
                    res.append(adv_a.get(0).LEMMA)
                else:
                    res.append(adv_a.get(0).LEMMA[1])
                res = insert_range_num(tree, Key_range, Key_dobj, Key_ccomp, res)
                return res

            # 若核心动词下的子结点为名词主语，例如：往左，往右
            if not nsubj_n.isEmpty():
                # 将得到的名词主语跟核心动词进行输出
                # print("%s = %s"%(word.LEMMA ,nsubj_n.get(0).LEMMA ))
                if nsubj_n.get(0).POSTAG == "P":
                    res.append(word.LEMMA[1])
                    res.append(word.LEMMA[0])
                else:
                    res.append(word.LEMMA)
                    res.append(nsubj_n.get(0).LEMMA)
                res = insert_range_num(tree, Key_range, Key_dobj, Key_ccomp, res)
                return res

            if not mmod_v.isEmpty():
                # print("%s = %s"%(word.LEMMA ,mmod_v.get(0).LEMMA ))
                res.append(word.LEMMA)
                if len(mmod_v.get(0).LEMMA) >= 2:
                    res.append(mmod_v.get(0).LEMMA[1])
                res = insert_range_num(tree, Key_range, Key_dobj, Key_ccomp, res)
                return res

            if not prep_dobj.isEmpty():
                # 若核心动词下存在直接宾语，若同样存在并列谓语。并列谓语与直接宾语进行输出
                # 若并列谓语不存在，说明真正的关键谓语与识别位于存在依赖关系，可以对等使用
                # 对于下述的情况，tree.findChildren(word,"dobj")所提取到的直接宾语是方向性名词，对于Key_dobj距离性直接宾语出现在conj兵力谓语之下
                if prep_conj.isEmpty():
                    # 若直接谓语下，包含常用名词，例如 左方。提取方向名词

                    nn_n = tree.findChildren(prep_dobj.get(0), "nn")
                    if not nn_n.isEmpty():
                        # print("%s = %s"%(prep_dobj.get(0).LEMMA ,nn_n.get(0).LEMMA ))
                        res.append(word.LEMMA)
                        res.append(nn_n.get(0).LEMMA)
                    else:
                        res.append(word.LEMMA)
                        res.append(prep_dobj.get(0).LEMMA)
                    res = insert_range_num(tree, Key_range, Key_dobj, Key_ccomp, res)
                else:
                    # print("%s = %s"%(prep_dobj.get(0).LEMMA ,prep_conj.get(0).LEMMA ))
                    res.append(prep_conj.get(0).LEMMA)
                    res.append(prep_dobj.get(0).LEMMA)
                    res = refind_range_num(tree, res, prep_conj)
                return res

            if not prep_conj.isEmpty() and prep_dobj.isEmpty():
                # 若核心动词下存在并列谓语，但不存在直接宾语，说明核心动词识别正确
                # 直接宾语dobj在并列谓语下
                prep_dobj = tree.findChildren(prep_conj.get(0), "dobj")
                if not prep_dobj.isEmpty():
                    # print("%s = %s"%(prep_conj.get(0).LEMMA ,prep_dobj.get(0).LEMMA ))
                    res.append(word.LEMMA)
                    res.append(prep_dobj.get(0).LEMMA)
                    res = insert_range_num(tree, Key_range, Key_dobj, Key_ccomp, res)
                    return res
                else:
                    # print("%s ="%(word.LEMMA ))
                    res.append(word.LEMMA[1])
                    res.append(word.LEMMA[0])
                    res = insert_range_num(tree, Key_range, Key_dobj, Key_ccomp, res)
                    return res

            if len(word.LEMMA) >= 2:
                # print("%s ="%(word.LEMMA ))
                res.append(word.LEMMA[0])
                res.append(word.LEMMA[1])
                res = insert_range_num(tree, Key_range, Key_dobj, Key_ccomp, res)
                return res
        elif word.POSTAG == "P" and word.DEPREL == "ROOT":
            prep_pccomp = tree.findChildren(word, "pccomp")  # 提取介词补充
            prep_pobj = tree.findChildren(word, "pobj")  # 提取介词宾语
            if not prep_pccomp.isEmpty():
                advmod_a = tree.findChildren(prep_pccomp.get(0), "advmod")
                dobj_n = tree.findChildren(prep_pccomp.get(0), "dobj")
                range_m = tree.findChildren(prep_pccomp.get(0), "range")
                if not advmod_a.isEmpty() and not dobj_n.isEmpty():
                    # print("%s = %s"%(prep_pccomp.get(0).LEMMA , advmod_a.get(0).LEMMA))
                    res.append(prep_pccomp.get(0).LEMMA)
                    res.append(advmod_a.get(0).LEMMA)
                    res.append(dobj_n.get(0).LEMMA)
                    return res
                elif not advmod_a.isEmpty() and not range_m.isEmpty():
                    # print("%s = %s"%(advmod_a.get(0).LEMMA , range_m.get(0).LEMMA))
                    nummod_c = tree.findChildren(range_m.get(0), "nummod")
                    res.append(prep_pccomp.get(0).LEMMA[0])
                    res.append(advmod_a.get(0).LEMMA)
                    if not nummod_c.isEmpty():
                        res.append(nummod_c.get(0).LEMMA)
                    return res
                elif not advmod_a.isEmpty():
                    res.append(prep_pccomp.get(0).LEMMA)
                    res.append(advmod_a.get(0).LEMMA)
                    return res
            if not prep_pobj.isEmpty():
                # 查找介词宾语下的名词
                nn_n = tree.findChildren(prep_pobj.get(0), "nn")
                amod_j = tree.findChildren(prep_pobj.get(0), "amod")
                # 查找介词宾语下的依赖性名词
                dep_n = tree.findChildren(prep_pobj.get(0), "dep")
                num_c = tree.findChildren(prep_pobj.get(0), "nummod")
                if not nn_n.isEmpty():
                    if amod_j.isEmpty():
                        amod_j = tree.findChildren(nn_n.get(0), "amod")
                        if not amod_j.isEmpty():
                            # print("%s = %s"%(prep_pobj.get(0).LEMMA , amod_j.get(0).LEMMA))
                            res.append(nn_n.get(0).LEMMA)
                            res.append(amod_j.get(0).LEMMA)
                            res.append(prep_pobj.get(0).LEMMA)
                            return res

                        elif (len(nn_n) >= 2):
                            # print("%s = %s"%(prep_pobj.get(0).LEMMA , nn_n.get(0).LEMMA))
                            res.append(nn_n.get(1).LEMMA)
                            res.append(nn_n.get(0).LEMMA)
                            res.append(prep_pobj.get(0).LEMMA)
                        else:
                            res.append(prep_pobj.get(0).LEMMA)
                            res.append(nn_n.get(0).LEMMA)
                        return res
                    else:

                        res.append(nn_n.get(0).LEMMA)
                        res.append(amod_j.get(0).LEMMA)
                        res.append(prep_pobj.get(0).LEMMA)
                        return res
                elif not amod_j.isEmpty():
                    # print("%s = %s"%(prep_pobj.get(0).LEMMA , amod_j.get(0).LEMMA))
                    res.append(prep_pobj.get(0).LEMMA)
                    res.append(amod_j.get(0).LEMMA)
                    return res
                elif not dep_n.isEmpty() and not num_c.isEmpty():
                    dep_amodJ = tree.findChildren(dep_n.get(0), "amod")
                    dep_nn = tree.findChildren(dep_n.get(0), "nn")
                    if len(dep_n.get(0).LEMMA) == 3:  # 对于关联名词长度为三个词，例如 后移动 （冲着后移动1米）
                        res.append(dep_n.get(0).LEMMA[1:3])  # 提取关联性名词
                        res.append(dep_n.get(0).LEMMA[0])  # 提取方向
                    elif len(dep_n.get(0).LEMMA) == 2:  # 奔右 的情况或者移动的情况
                        # print("%s = %s"%(prep_pobj.get(0).LEMMA , dep_n.get(0).LEMMA))
                        if dep_n.get(0).POSTAG == "NN":
                            res.append(dep_n.get(0).LEMMA)
                        else:
                            res.append(dep_n.get(0).LEMMA[1])
                            res.append(dep_n.get(0).LEMMA[0])
                    else:
                        res.append(dep_n.get(0).LEMMA)
                    if not dep_amodJ.isEmpty():
                        # print("%s = %s"%(prep_pobj.get(0).LEMMA , dep_amodJ.get(0).LEMMA))
                        res.append(dep_amodJ.get(0).LEMMA)
                    elif not dep_nn.isEmpty():
                        res.append(dep_nn.get(0).LEMMA)
                    res.append(num_c.get(0).LEMMA + prep_pobj.get(0).LEMMA)
                    return res
        elif word.POSTAG == "NN" and word.DEPREL == "ROOT":  # 核心词被识别为名词
            nn_n = tree.findChildren(word, "nn")
            if not nn_n.isEmpty():
                print("=====")
                res.append(nn_n.get(1).LEMMA)
                res.append(nn_n.get(0).LEMMA)
                res.append(word.LEMMA)
                return res

        elif word.POSTAG == "M" and word.DEPREL == "ROOT":  # 核心词被识别为名词：米 （向左移动一米）
            dep_n = tree.findChildren(word, "dep")  # 对于存在依赖关系的名词，例如 移动
            nummod_c = tree.findChildren(word, "nummod")  # 核心名词下的数量词
            adv_a = tree.findChildren(word, "advmod")
            if not dep_n.isEmpty() and not nummod_c.isEmpty():
                dep_prep = tree.findChildren(dep_n.get(0), "prep")  # 寻找依赖性名词下的介词
                if not dep_prep.isEmpty():
                    # 寻找介词下的宾语

                    prep_pobj = tree.findChildren(dep_prep.get(0), "pobj")
                    if not prep_pobj.isEmpty():
                        res.append(dep_n.get(0).LEMMA)
                        res.append(prep_pobj.get(0).LEMMA)
                        res.append(nummod_c.get(0).LEMMA + word.LEMMA)
                        return res
            if not adv_a.isEmpty() and not nummod_c.isEmpty():
                if len(nummod_c.get(0).LEMMA) == 2:
                    res.append(nummod_c.get(0).LEMMA[0])
                    if len(adv_a.get(0).LEMMA) == 2:
                        res.append(adv_a.get(0).LEMMA[1])
                    res.append(nummod_c.get(0).LEMMA[1] + word.LEMMA)
                    return res
#======================================A1类指令=============================================


#======================================B1类指令=============================================
# 提取B1指令的核心关键词
def locate_position_info(tree, assmod_n, res):
    nn_n = tree.findChildren(assmod_n.get(0), "nn")
    nummod_c = tree.findChildren(assmod_n.get(0), "nummod")
    dep_n = tree.findChildren(assmod_n.get(0), "dep")
    if not nn_n.isEmpty():
        if len(nn_n) == 2:  # 若得到了两个结果，那么两个结果中 第一个nn表示位置，第二个表示方向
            det_d = tree.findChildren(nn_n.get(0), "det")  # 当前句子中，动词被限定为决定词
            if not det_d.isEmpty():
                res.append(det_d.get(0).LEMMA)
                res.append(nn_n.get(0).LEMMA)
                res.append(nn_n.get(1).LEMMA)
                res.append(assmod_n.get(0).LEMMA)
                return res
            else:
                res.append(nn_n.get(0).LEMMA)
                res.append(nn_n.get(1).LEMMA)
            return res
        elif len(nn_n) == 1:
            res.append(nn_n.get(0).LEMMA)
            res.append(assmod_n.get(0).LEMMA)
        return res
    if not nummod_c.isEmpty():
        if not dep_n.isEmpty():

            nn_n = tree.findChildren(dep_n.get(0), "nn")
            if not nn_n.isEmpty():
                det_d = tree.findChildren(nn_n.get(0), "det")  # 当前句子中，动词被限定为决定词
                if not det_d.isEmpty():
                    res.append(det_d.get(0).LEMMA)
                    res.append(nn_n.get(0).LEMMA)

            res.append(dep_n.get(0).LEMMA)
        res.append(nummod_c.get(0).LEMMA + assmod_n.get(0).LEMMA)
    return res


def find_exect_position(tree, nn_n, res):
    nn = tree.findChildren(nn_n.get(0), "nn")
    if not nn.isEmpty():
        res.append(nn.get(0).LEMMA)
        res.append(nn_n.get(0).LEMMA)
        return res


def extract_verb_object_B1(tree):
    for word in tree.iterator():
        res = []
        if word.POSTAG == 'VV' and word.DEPREL == 'ROOT':
            # 若该单词是动词，并且是核心动词

            # 查找核心介词修饰
            prep_p = tree.findChildren(word, "prep")

            # 找核心单词的的情态动词
            mod_v = tree.findChildren(word, "mmod")

            # 找核心单词的直接宾语
            dobj_n = tree.findChildren(word, "dobj")

            if not prep_p.isEmpty():
                pobj_n = tree.findChildren(prep_p.get(0), "pobj")
                res.append(word.LEMMA)
                res.append(pobj_n.get(0).LEMMA)
                return res

            if not mod_v.isEmpty():
                res.append(mod_v.get(0).LEMMA)
                res.append(word.LEMMA)
                if not dobj_n.isEmpty():
                    # 找到直接宾语下的关联修饰 既搜索是否存在限定范围的修饰词
                    assmod_n = tree.findChildren(dobj_n.get(0), "assmod")
                    nn_n = tree.findChildren(dobj_n.get(0), "nn")
                    if not assmod_n.isEmpty() and nn_n.isEmpty():
                        res = locate_position_info(tree, assmod_n, res)
                    elif not assmod_n.isEmpty() and not nn_n.isEmpty():
                        res.append(nn_n.get(0).LEMMA)
                        res = locate_position_info(tree, assmod_n, res)
                return res

            if not dobj_n.isEmpty():
                res.append(word.LEMMA)

                # 找到直接宾语下的关联修饰 既搜索是否存在限定范围的修饰词
                assmod_n = tree.findChildren(dobj_n.get(0), 'assmod')
                nn_n = tree.findChildren(dobj_n.get(0), "nn")  ## 找到直接宾语下的名词词组
                ## 查找具有依赖关系的方向性名词
                dep_l = tree.findChildren(dobj_n.get(0), "dep")

                if not assmod_n.isEmpty() and nn_n.isEmpty():
                    res = locate_position_info(tree, assmod_n, res)
                    res.append(assmod_n.get(0).LEMMA)
                    return res
                elif not assmod_n.isEmpty() and not nn_n.isEmpty():
                    res = find_exect_position(tree, nn_n, res)
                    res = locate_position_info(tree, assmod_n, res)
                    return res
                elif assmod_n.isEmpty() and nn_n.isEmpty():
                    res.append(dobj_n.get(0).LEMMA)
                    return res
        # 对于特殊的语法结构，既介词作为核心
        if word.POSTAG == 'P' and word.DEPREL == 'ROOT':
            res.append(word.LEMMA)
            dobj_n = tree.findChildren(word, "pobj")
            if not dobj_n.isEmpty():
                # 找到直接宾语下的关联修饰 既搜索是否存在限定范围的修饰词
                assmod_n = tree.findChildren(dobj_n.get(0), 'assmod')
                nn_n = tree.findChildren(dobj_n.get(0), "nn")  ## 找到直接宾语下的名词词组
                if not assmod_n.isEmpty() and nn_n.isEmpty():
                    res = locate_position_info(tree, assmod_n, res)
                    res.append(assmod_n.get(0).LEMMA)
                    return res
                elif not assmod_n.isEmpty() and not nn_n.isEmpty():
                    res = find_exect_position(tree, nn_n, res)
                    res = locate_position_info(tree, assmod_n, res)
                    return res
                else:
                    res.append(dobj_n.get(0).LEMMA)
                    return res

        if word.POSTAG == 'NN' and word.DEPREL == 'ROOT':
            # 查找冠词（决定词）
            det_d = tree.findChildren(word, "det")
            # 查找名词（nn）
            nn_n = tree.findChildren(word, "nn")
            assmod_n = tree.findChildren(word, "assmod")

            if not det_d.isEmpty():
                res.append(det_d.get(0).LEMMA)
                res.append(word.LEMMA)
                return res

            if not nn_n.isEmpty():
                res.append(nn_n.get(0).LEMMA)
                res.append(word.LEMMA)
                return res
            if not assmod_n.isEmpty():
                res = locate_position_info(tree, assmod_n, res)
                return res
#======================================B1类指令==========================================

#======================================C1类指令=============================================
# 提取C1指令的核心关键词

def search_modifier(tree, dobj_n, res):
    nummod_c = tree.findChildren(dobj_n.get(0), "nummod")  # 搜索数词修饰
    assmod_j = tree.findChildren(dobj_n.get(0), "assmod")  # 搜索关联修饰
    clf_m = tree.findChildren(dobj_n.get(0), "clf")  # 搜索类别修饰

    if not nummod_c.isEmpty() and not assmod_j.isEmpty() and clf_m.isEmpty():  # 拿起一个红色的书，此时两种修饰作为直接宾语的同级
        res.append(assmod_j.get(0).LEMMA)
        res.append(nummod_c.get(0).LEMMA)
        return res
    elif nummod_c.isEmpty() and not assmod_j.isEmpty() and clf_m.isEmpty():  # 拿起一个黑色的箱子，数词修饰作为关联修饰的下一级
        nummod_underAss = tree.findChildren(assmod_j.get(0), "nummod")
        if not nummod_underAss.isEmpty():
            res.append(assmod_j.get(0).LEMMA)
            res.append(nummod_underAss.get(0).LEMMA)
            return res
    elif nummod_c.isEmpty() and not assmod_j.isEmpty() and not clf_m.isEmpty():
        # 拿起两个橙色的水杯 此时
        nummod_underClf = tree.findChildren(clf_m.get(0), "nummod")
        if not nummod_underClf.isEmpty():
            res.append(assmod_j.get(0).LEMMA)
            res.append(nummod_underClf.get(0).LEMMA + clf_m.get(0).LEMMA)
            return res


def extract_verb_object_C1(tree):
    for word in tree.iterator():
        res = []
        if word.POSTAG == 'VV' and word.DEPREL == 'ROOT':
            # 搜索核心动词下的直接宾语    拿起书
            dobj_n = tree.findChildren(word, "dobj")
            if not dobj_n.isEmpty():
                res.append(word.LEMMA)
                res.append(dobj_n.get(0).LEMMA)
                assmod_j = tree.findChildren(dobj_n.get(0), "assmod")  # 搜索关联修饰
                if not assmod_j.isEmpty():
                    # 搜索直接引语下的修饰成分
                    res = search_modifier(tree, dobj_n, res)
            return res
        elif word.POSTAG == 'NN' and word.DEPREL == 'ROOT':
            nn_n = tree.findChildren(word, "nn")
            if not nn_n.isEmpty():
                res.append(nn_n.get(0).LEMMA)
                res.append(word.LEMMA)
            return res
#=========================================C1类指令========================================

#=========================================BC类指令========================================
# 提取BC类指令的核心关键词
def move_posInfo(tree, pobj_n, res):
    assmod_n = tree.findChildren(pobj_n.get(0), "assmod")  # 关联修饰
    recmod_v = tree.findChildren(pobj_n.get(0), "rcmod")  # 相关关系，例如：窗台与桌子有相关关系
    if not assmod_n.isEmpty():
        res.append(assmod_n.get(0).LEMMA)  # 移动到实验室的桌子，拿起书
        res.append(pobj_n.get(0).LEMMA)
    elif not recmod_v.isEmpty():
        pobj = tree.findChildren(recmod_v.get(0), "pobj")
        dobj_n = tree.findChildren(recmod_v.get(0), "dobj")
        if not pobj.isEmpty():
            res.append(recmod_v.get(0).LEMMA)
            res.append(pobj_n.get(0).LEMMA)
            res.append(pobj.get(0).LEMMA)
        elif not dobj_n.isEmpty():
            res.append(dobj_n.get(0).LEMMA)
            res.append(pobj_n.get(0).LEMMA)
        else:
            res.append(recmod_v.get(0).LEMMA)
            res.append(pobj_n.get(0).LEMMA)
    return res


def action_info(tree, conj_v, res):
    dobj_n = tree.findChildren(conj_v.get(0), "dobj")
    if not dobj_n.isEmpty():
        res.append(conj_v.get(0).LEMMA)
        res.append(dobj_n.get(0).LEMMA)
    return res


def extract_verb_object_BC(tree):
    for word in tree.iterator():
        res = []
        if word.POSTAG == 'VV' and word.DEPREL == 'ROOT':
            # 搜索并列动词 既与主动词并列的动词，
            conj_v = tree.findChildren(word, "conj")
            # 搜索介词修饰  例如  到
            prep_p = tree.findChildren(word, "prep")
            # 搜索动词后的直接宾语
            dobj = tree.findChildren(word, "dobj")
            # 搜索具有依赖关系的名词
            dep_n = tree.findChildren(word, "dep")
            # 搜索名词主语 这种情况下，前半段的名词主语
            nsubj = tree.findChildren(word, "nsubj")

            if not prep_p.isEmpty() and not conj_v.isEmpty() and dobj.isEmpty():
                # 移动到实验室的桌子，拿起书
                # 搜索介词修饰下的直接宾语
                res.append(word.LEMMA)
                pobj_n = tree.findChildren(prep_p.get(0), "pobj")
                if not pobj_n.isEmpty():
                    res = move_posInfo(tree, pobj_n, res)
                res = action_info(tree, conj_v, res)
                return res
            elif not prep_p.isEmpty() and conj_v.isEmpty():
                # 到实验室的桌子，拿起苹果
                # 该指令中，核心动词被识别为 拿起，到 作为介词修饰出现在核心东西的下一级
                res.append(prep_p.get(0).LEMMA)
                pobj_n = tree.findChildren(prep_p.get(0), "pobj")
                if not pobj_n.isEmpty():
                    res = move_posInfo(tree, pobj_n, res)
                dobj_n = tree.findChildren(word, "dobj")
                if not dobj_n.isEmpty():
                    res.append(word.LEMMA)
                    res.append(dobj_n.get(0).LEMMA)
                return res
            elif not dobj.isEmpty() and not conj_v.isEmpty() and prep_p.isEmpty():
                # 去实验室的桌子上，拿起苹果
                res.append(word.LEMMA)
                res = move_posInfo(tree, dobj, res)
                res = action_info(tree, conj_v, res)
                return res
            elif not dobj.isEmpty() and not dep_n.isEmpty() and conj_v.isEmpty():
                # 去实验室的桌子处，识别微波炉   此处，识别是作为专有名词出现
                res.append(word.LEMMA)
                res = move_posInfo(tree, dobj, res)
                nn_n = tree.findChildren(dep_n.get(0), "nn")
                if not nn_n.isEmpty():
                    res.append(nn_n.get(0).LEMMA)
                    res.append(dep_n.get(0).LEMMA)
                return res
            elif not nsubj.isEmpty() and not dobj.isEmpty() and conj_v.isEmpty():
                # 到窗台的桌子位置，放下书  由于前半句的核心动词被识别为相关介词，后半句的动词被识别为核心动词
                # 首先找到位置信息
                rcmod_v = tree.findChildren(nsubj.get(0), "rcmod")
                if not rcmod_v.isEmpty():
                    pobj = tree.findChildren(rcmod_v.get(0), "pobj")  # 搜索介词宾语
                    if not pobj.isEmpty():
                        res.append(rcmod_v.get(0).LEMMA)
                        res.append(pobj.get(0).LEMMA)
                    res.append(nsubj.get(0).LEMMA)
                    # 找到具体的动作
                res.append(word.LEMMA)
                res.append(dobj.get(0).LEMMA)
                return res
        elif word.POSTAG == 'NN' and word.DEPREL == 'ROOT':
            # 到窗台的大门，识别微波炉  核心动词被识别为名词
            dep_p = tree.findChildren(word, "dep")
            nn_n = tree.findChildren(word, "nn")
            if not dep_p.isEmpty() and not nn_n.isEmpty():
                probj_n = tree.findChildren(dep_p.get(0), "pobj")
                if not probj_n.isEmpty():
                    res.append(dep_p.get(0).LEMMA)
                    res = move_posInfo(tree, probj_n, res)
                else:
                    res = move_posInfo(tree, dep_p, res)
                res.append(nn_n.get(0).LEMMA)
                res.append(word.LEMMA)
                return res
#=========================================BC类指令========================================

#=========================================TA类指令=========================================
# 搜索TA的关键词

def search_timeInfo(tree, verb, res):
    lobj_n = tree.findChildren(verb.get(0), "lobj")  # 搜索时间介词  20秒
    if not lobj_n.isEmpty():
        nummod_c = tree.findChildren(lobj_n.get(0), "nummod")
        if not nummod_c.isEmpty():
            res.append(nummod_c.get(0).LEMMA + lobj_n.get(0).LEMMA)
        else:
            res.append(lobj_n.get(0).LEMMA)
        return res


def search_posInfo(tree, verb, res):
    probj_n = tree.findChildren(verb.get(0), "pobj")  # 搜索直接宾语  实验室
    nn_n = tree.findChildren(verb[0], "nn")  # 搜索专有名词   左侧
    assmod_m = tree.findChildren(verb[0], "assmod")  # 搜索关联修饰 例如  一米 米
    if not probj_n.isEmpty():
        res.append(probj_n.get(0).LEMMA)
        return res
    if not nn_n.isEmpty() and not assmod_m.isEmpty():  # 桌子左侧两米的位置 专有名词"左侧" 与 关联修饰作为同级
        pos_n = tree.findChildren(nn_n[0], "nn")
        nummod_c = tree.findChildren(assmod_m[0], "nummod")
        if not pos_n.isEmpty():
            res.append(pos_n[0].LEMMA)
            res.append(nn_n[0].LEMMA)
        if not nummod_c.isEmpty():
            res.append(nummod_c[0].LEMMA + assmod_m[0].LEMMA)
        else:
            res.append(assmod_m[0].LEMMA)
        return res
    if not assmod_m.isEmpty() and nn_n.isEmpty():
        pos_n = tree.findChildren(assmod_m[0], "nn")
        dep_n = tree.findChildren(assmod_m[0], "dep")
        nummod_c = tree.findChildren(assmod_m[0], "nummod")

        if not pos_n.isEmpty() and len(pos_n) == 2:
            res.append(pos_n[0].LEMMA)
            res.append(pos_n[1].LEMMA)
            res.append(assmod_m[0].LEMMA)
        elif not pos_n.isEmpty() and len(pos_n) == 1:
            res.append(pos_n[0].LEMMA)
            res.append(assmod_m[0].LEMMA)
        elif not dep_n.isEmpty() and not nummod_c.isEmpty():
            pos_n = tree.findChildren(dep_n[0], "nn")
            if not pos_n.isEmpty():
                res.append(pos_n[0].LEMMA)
                res.append(dep_n[0].LEMMA)
            else:
                res.append(dep_n[0].LEMMA)
            res.append(nummod_c[0].LEMMA + assmod_m[0].LEMMA)
        return res
    return res


def extract_verb_object_TA(tree):
    for word in tree.iterator():
        res = []
        if word.POSTAG == 'VV' and word.DEPREL == 'ROOT':
            loc_l = tree.findChildren(word, "loc")  # 搜索位置状语  后
            prep_p = tree.findChildren(word, "prep")  # 搜索介词修饰  到
            dobj_n = tree.findChildren(word, "dobj")  # 搜索直接宾语   实验室

            mmod_v = tree.findChildren(word, "mmod")
            # 搜索情态动词   三十秒后去展台左侧一米的位置  这里"去"被识别为情态动词

            son_node = tree.findChildren(word)  # 搜索 被识别为动词的直接宾语
            for i in range(len(son_node)):
                if son_node[i].POSTAG == "VV":
                    verb_dobj = son_node[i]

            if not loc_l.isEmpty() and not prep_p.isEmpty() and dobj_n.isEmpty():
                # 十秒后移动到实验室
                res.append(word.LEMMA)
                res = search_posInfo(tree, prep_p, res)
                res = search_timeInfo(tree, loc_l, res)
                return res
            elif not loc_l.isEmpty() and prep_p.isEmpty() and not dobj_n.isEmpty() and mmod_v.isEmpty():
                # 十秒后去实验室
                res.append(word.LEMMA)
                # 对于详细定时类指令 例如 三十秒后去桌子左侧两米的位置
                childNode = tree.findChildren(dobj_n[0])
                if not childNode.isEmpty():
                    res = search_posInfo(tree, dobj_n, res)
                else:
                    res.append(dobj_n.get(0).LEMMA)
                res = search_timeInfo(tree, loc_l, res)
                return res
            elif not loc_l.isEmpty() and prep_p.isEmpty() and not dobj_n.isEmpty() and not mmod_v.isEmpty():
                # 此处“去”被识别为情态动词
                res.append(mmod_v[0].LEMMA)
                res.append(word.LEMMA)
                res = search_posInfo(tree, dobj_n, res)
                res = search_timeInfo(tree, loc_l, res)
                return res
            elif not loc_l.isEmpty() and prep_p.isEmpty() and not verb_dobj == None:
                res.append(word.LEMMA)
                res.append(verb_dobj.LEMMA)
                res = search_timeInfo(tree, loc_l, res)
                return res

        elif word.POSTAG == 'NN' and word.DEPREL == 'ROOT':
            dep_l = tree.findChildren(word, "dep")
            assmod_m = tree.findChildren(word, "assmod")
            if not dep_l.isEmpty():  # 30s后上教学楼
                if len(dep_l) == 2:
                    res.append(dep_l[1].LEMMA)
                    res.append(word.LEMMA)
                    lobj_n = tree.findChildren(dep_l[0], "lobj")
                    if not lobj_n.isEmpty():
                        nummod = tree.findChildren(lobj_n[0], "nummod")
                        if not nummod.isEmpty():
                            res.append(nummod[0].LEMMA + lobj_n[0].LEMMA)
                        else:
                            res.append(lobj_n[0].LEMMA)
                        return res
                elif len(dep_l) == 1:
                    if not dep_l.isEmpty():
                        res.append(dep_l.get(0).LEMMA)
                        res.append(word.LEMMA)
                        lobj_l = tree.findChildren(dep_l.get(0), "lobj")
                        if not lobj_l.isEmpty():
                            res = search_timeInfo(tree, lobj_l, res)
                        return res
            if not assmod_m.isEmpty():
                # 一分钟后上凳子前方两米的位置   由于 上 的特殊性，句子整体
                loc_l = tree.findChildren(assmod_m[0], "loc")
                dep_n = tree.findChildren(assmod_m[0], "dep")
                nummod_c = tree.findChildren(assmod_m[0], "nummod")
                pos_n = tree.findChildren(assmod_m[0], "nn")  # 搜索专有名词  前方与凳子

                if not loc_l.isEmpty():  # 一分钟后 "上" 凳子前方两米的位置
                    res.append(loc_l[0].LEMMA)
                if not dep_n.isEmpty():  # 一分钟后上 "凳子前方" 一米的位置
                    if dep_n[0].POSTAG == 'LC':  # 三十秒后上桌子左侧一米的位置
                        res.append(dep_n[0].LEMMA)
                    nn_n = tree.findChildren(dep_n[0], "nn")  # 此时 前方 与 凳子 是作为修饰名词出现的 这种情况下 本应该作为动词的上 作为
                    if not nn_n.isEmpty():
                        res.append(nn_n[0].LEMMA)
                        res.append(dep_n[0].LEMMA)
                if not pos_n.isEmpty():  # 一分钟后上 "凳子前方" 一米的位置
                    if len(pos_n) == 2:
                        res.append(pos_n[0].LEMMA)
                        res.append(pos_n[1].LEMMA)
                if not nummod_c.isEmpty():  # 一分钟后上凳子前方 "两米" 的位置
                    res.append(nummod_c[0].LEMMA + assmod_m[0].LEMMA)
                else:
                    res.append(assmod_m[0].LEMMA)
                if not loc_l.isEmpty():  # "一分钟"后上凳子前方两米的位置
                    lobj_l = tree.findChildren(loc_l[0], "lobj")
                    if not lobj_l.isEmpty():
                        res = search_timeInfo(tree, loc_l, res)
                elif not dep_n.isEmpty():  # 一分钟后上 "凳子前方" 一米的位置  此处上被修饰为具有依赖关系的方向词
                    lobj_l = tree.findChildren(dep_n[0], "lobj")
                    if not lobj_l.isEmpty():
                        res = search_timeInfo(tree, lobj_l, res)
                return res
#=========================================TA类指令=========================================

#=========================================同义映射=========================================
# 搜索同义词
def FindMostSimilarity(wv,word,listWord):
    similarity = 0
    result = ""
    for i in range(len(listWord)):
        same = wv.similarity(word,listWord[i])
        if same > similarity:
            similarity = same
            result = listWord[i]
    return result,similarity

# 完成同义词映射
def MapKeyWord_Instruction(vec):
    string  = []
    for i in range(len(vec)):
        word = vec[i]
        if word in mv_from_text.wv.vocab.keys():
            standardWord = FindMostSimilarity(mv_from_text,word,ListKeyWord)  # 根据关键词表，找到核心关键词
            strMatchInstruction = mapWordVector[standardWord[0]]
            string.append(strMatchInstruction)
    return string
#=========================================同义映射=========================================

#========================================指令解析接口======================================
# 指令转换接口函数
def GetInstructonKey(string):
    classVal = classifyInstruction(string)
    if classVal == 1:
        tree = parser.parse(string)
        vec = extract_verb_object_A1(tree)
        if vec != None:
            vecInstruct = MapKeyWord_Instruction(vec)
            return classVal, vecInstruct
        else:
            return classVal, None
    elif classVal == 2:
        tree = parser.parse(string)
        vec = extract_verb_object_B1(tree)
        if vec != None:
            vecInstruct = MapKeyWord_Instruction(vec)
            return classVal, vecInstruct
        else:
            return classVal, None
    elif classVal == 3:
        tree = parser.parse(string)
        vec = extract_verb_object_C1(tree)
        if vec != None:
            vecInstruct = MapKeyWord_Instruction(vec)
            return classVal, vecInstruct
        else:
            return classVal, None
    elif classVal == 4:
        tree = parser.parse(string)
        vec = extract_verb_object_BC(tree)
        if vec != None:
            vecInstruct = MapKeyWord_Instruction(vec)
            return classVal, vecInstruct
        else:
            return classVal, None
    elif classVal == 5:
        tree = parser.parse(string)
        vec = extract_verb_object_TA(tree)
        if vec != None:
            vecInstruct = MapKeyWord_Instruction(vec)
            return classVal, vecInstruct
        else:
            return classVal, None
#========================================指令解析接口======================================
def command_analysis():
    LoadInit()
    res = GetInstructonKey('向左拐90度')
    print(res)



if __name__ == '__main__':
    command_analysis()