# 'utf-8'
# 大数据分析期末作业脚本
# 以理服民与以情动民：垃圾分类中的政府回应——以浙江民呼我为统一平台为例
# user：艾热帕提·努尔买买提

# 代码框架
# 第一部分 数据爬虫
# 第二部分 数据处理以及数据描述统计、可视化
# 第三部分 基于LDA模型的主题分析

#%% 加载包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
import re
import requests
import json
import random
import pyLDAvis
import pyLDAvis.sklearn
from snownlp import SnowNLP
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


#%%第一部分 数据爬虫
#浙江省政府网信件数据爬虫代码
#第一步 获取url，
#https://www.zjzxts.gov.cn/sun/satisfaction?page=xjgk&gkbz=1&areacode=330000&bmdm=null&xzqh=null&bt=&year=2021&quarter=0&bldwmc=&xfmd=98&fyxs=01&areacode=330000&city=&area=&index=0
#通过修改网站：浙江12345
#分类指标： 目的（咨询xfmd=98、意见建议xfmd=01、申诉xfmd=02、求决xfmd=03、其他xfmd=99）和反映形式（来信fyxs=01、来访fyxs=02、网上fyxs=03、来电fyxs=04）
#二级网页 URL = https://www.zjzxts.gov.cn/wsdt/wsdtHtml/xfjxq.jsp?id=4c583230323130303439333535383431*

#设置抬头
headers = {}
User_Agent = ['Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36',
              'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36 Edg/95.0.1020.44',
              'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36',
              'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
              'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0']
headers['User_Agent'] = random.choice(User_Agent)


#%%% 设置网页获取HTML函数
def get_html(url):
	response = requests.get(url, headers=headers,timeout=10)
	if response.status_code == 200:
		html = BeautifulSoup(response.text, 'lxml')
		return html
	else:
		return  "网页获取失败"

#%%%定义解析一级网页函数
#在第一级网页中可以获取诉求id,目的,诉求状态，以及二级网页的特殊id
def parse1(html):
	tr_count = len(html.select('table.tablelist1 tr'))
	id = []
	purpose = []
	status = []
	result = {}
	if tr_count > 1:
		for i in range(1,tr_count):
			id.append(html.select('table.tablelist1 tr')[i].td['onclick'])
			purpose.append(html.select('table.tablelist1 tr')[i].select('td')[3].get_text())
			status.append(html.select('table.tablelist1 tr')[i].select('td')[4].get_text())
			result = {'id':id,'purpose':purpose,'status':status}
		return result
	else:
		return result

#%%% 定义解析二级网页函数
#因为二级网页无法通过解析来获得数据，在这里使用正则表达式来提取数据
def parse2(url2):
	r = requests.get(url2,headers=headers,timeout=20)
	text = r.text
	#对字符串进行切割,锁定需要爬取的数据段
	text_1 = text.split('<span class="font-double">')[1]
	#对切割后的数据段解析
	output = BeautifulSoup(str(text_1),'lxml')
	title = output.p.get_text()
	output_1 = output.select('span.blue1')
	appeal = output_1[0].get_text()
	rct_time = output_1[1].get_text()
	appeal_source = output_1[2].get_text()
	rep_sector = output_1[3].get_text()
	response = output_1[5].get_text()
	rep_time = output_1[4].get_text()
	result = {'title':title,'appeal':appeal,'rct_time':rct_time,'appeal_source':appeal_source,'rep_sector':rep_sector,'response':response,'rep_time':rep_time}
	return result

#%%% 设置爬虫函数
#可根据不同的目的和诉求形式获取数据
#页码需要手动设置
def get_data(purpose,appeal_form,page):
	dict_purpose = {'咨询':'98','意见建议':'01','申诉':'02','求决':'03','其他':'99'}
	dict_appeal_form = {'来信':'01','来访':'02','网上':'03','来电':'04'}
	pur = dict_purpose[str(purpose)]
	app = dict_appeal_form[str(appeal_form)]
	data = pd.DataFrame(columns=("title","appeal","words","rct_time","appeal_source","appeal_form","rep_sector","response","rep_time","purpose","status","topics"),index=range(page))
	n = 0
	for page in range(0,int(page)):
		url = "https://www.zjzxts.gov.cn/sun/satisfaction?page=xjgk&gkbz=1&areacode=330000&bmdm=null&xzqh=null&bt=&year=2021&quarter=0&bldwmc=&xfmd=98"+pur+"&fyxs="+app+"&areacode=330000&city=&area=&index="+str(page)
		container1 = {}
		container2 = {}
		html = get_html(url)
		if type(html) == str:
			data = data
		else:
			if len(html.select('table.tablelist1 tr')) == 1:
				data = data
			elif len(html.select('table.tablelist1 tr')) > 1:
				container1 = parse1(html)
				#设置id_list
				id_list = []
				for id_str in container1['id']:
					id_list.append(re.findall('\d+',id_str))
				for k,status in zip(id_list,container1['status']):
					url2 = "https://www.zjzxts.gov.cn/wsdt/wsdtHtml/xfjxq.jsp?id="+k[0]
					container2 = parse2(url2)
					data.title[n] = container2['title']
					data.appeal[n] = container2['appeal']
					data.rct_time[n] = container2['rct_time']
					data.appeal_source[n] = container2['appeal_source']
					data.rep_sector[n] = container2['rep_sector']
					data.response[n] = container2['response']
					data.rep_time[n] = container2['rep_time']
					data.purpose[n] = purpose
					data.appeal_form[n] = appeal_form
					data.status[n] = status	
	n+=1
	return data
#可根据不同诉求目的和诉求形式获取数据
#例如 获取网上留言形式的关于咨询类
data = get_data('咨询','网上',40)
data.to_csv("D:\本科课程\大三第一学期\大数据分析\期末大作业\浙江政务网词条分析\数据\浙江12345_咨询网上数据.csv",encoding='utf-8-sig')
# 最终获取20张表，并手工整合为总表

# %% 第二部分 数据清洗
# 导入数据
data = pd.read_excel("D:\本科课程\大三第一学期\大数据分析\期末大作业\浙江政务网词条分析\数据\浙江12345_垃圾分类总.xlsx")
# 导入停用词表
stopwords = [i.strip() for i in open(
    "D:\本科课程\大三第一学期\大数据分析\期末大作业\浙江政务网词条分析\数据\characters-master\stop_words", encoding="utf-8").readlines()]
# 导入关键词表
# 定义分词和删除停用词函数
# 先对每一条诉求进行分词，对于每一个词先删除空格，再判断是否为停用词，并且删除换行符，并把最终词条存入新列表
# 数据清理
#定义分词和删除关键词中的特殊停用词函数
def clear_stopword(word):
	new_word = list()
	for j in word:
		j = j.replace(" ","")
		if j in stopwords:
			del j
		elif "\n" in j:
			del j 
		else: 
			new_word.append(j)
	return new_word

#对诉求提取关键词和权重
#在data的words中填入提取的关键词并且删除其中的特殊停用词
n = 0
for i in range(0,len(data.appeal)):
	word_list = list()
	keywords = jieba.analyse.textrank(data.appeal[i],topK=30,withWeight=True)
	for key in keywords:
		word_list.append(key[0])	
	data.appeal_words[i] = str(clear_stopword(word_list))
	n+= 1

#在此步结束后，我们得到appeal_words列来存储每一条诉求主题词提取并清洗后的结果

#%% 数据描述性统计与可视化
# 诉求目的类型及比例
data['appeal_form'] = data['appeal_form'].astype('category')
data['appeal_source'] = data['appeal_source'].astype('category')
data['purpose'] = data['purpose'].astype('category')
print("公民诉求的形式、 公民诉求来源、 公民诉求目的的描述统计为:\n",data[['appeal_form','appeal_source','purpose']].describe())

#公民诉求类型与目的的可视化
p = plt.figure(figsize=(15,4))	#设置画布
rct_time = data.rct_time
plt.rcParams['font.sans-serif'] = 'SimHei'	#设置中文字体
plt.rc('font',size = 14)	#设置字体大小
#子图1
ax1 = p.add_subplot(1,2,1)
form = data.appeal_form.value_counts()
labels = form.index
plt.title('公民诉求类型')
plt.bar(labels,form)
plt.xlabel('诉求形式')	#设置横坐标
plt.ylabel('频数')

#子图2
ax1 = p.add_subplot(1,2,2)
purpose = data.purpose.value_counts()
labels2 = purpose.index
plt.title('诉求目的')
plt.bar(labels2,purpose)
plt.xlabel("诉求目的")
plt.ylabel('频数')
plt.show()


#诉求数量时间趋势图
rct_time.plot(figsize = (20,6))
plt.title("诉求数量时间序列图")
plt.xlabel('日期')
plt.ylabel("诉求数量")
#按照目的分类计算诉求数量趋势
p = plt.figure(figsize=(25,120))

#子图1 求决
ax1 = p.add_subplot(5,1,1)
qj = data[data.purpose =='求决']
time_qj = qj.rct_time.value_counts()
time_qj.plot(figsize = (25,20),color = 'lightcoral')
plt.title("求决型诉求时间趋势")
plt.legend("求决")
plt.xlabel('日期')
plt.ylabel("诉求数量")

#子图2
ax1 = p.add_subplot(5,1,2)
ss = data[data.purpose =='申诉']
time_ss = ss.rct_time.value_counts()
time_ss.plot(figsize = (25,20),color = 'burlywood')
plt.title("申诉型诉求时间趋势")
plt.legend("申诉") 
plt.xlabel('日期')
plt.ylabel("诉求数量")

#子图3
ax1 = p.add_subplot(5,1,3)
yjjy = data[data.purpose =='意见建议']
time_yjjy = yjjy.rct_time.value_counts()
time_yjjy.plot(figsize = (25,20),color = 'mediumturquoise')
plt.title("意见建议诉求时间趋势")
plt.legend("意见建议")
plt.xlabel('日期')
plt.ylabel("诉求数量")

#子图4
ax1 = p.add_subplot(5,1,4)
zs = data[data.purpose =='咨询']
time_zs = zs.rct_time.value_counts()
time_zs.plot(figsize = (25,20),color = 'mediumpurple')
plt.title("咨询诉求时间趋势")
plt.legend("咨询")
plt.xlabel('日期')
plt.ylabel("诉求数量")

#子图5
ax1 = p.add_subplot(5,1,5)
qt = data[data.purpose =='其他']
time_qt = qt.rct_time.value_counts()
time_qt.plot(figsize = (25,20),color = 'mediumblue')
plt.title("其他诉求时间趋势")
plt.legend("其他")
plt.xlabel('日期')
plt.ylabel("诉求数量")
plt.tight_layout()
plt.show()


#%% 第三部分 LDA主题分析
n_features = 25      #提取25个特征词语
word_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words= stopwords,
                                )
tf = word_vectorizer.fit_transform(data.appeal_words)

#确定将诉求归类为三个主题
n_topics = 3
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=100,
                                learning_method='batch',
                                learning_offset=50,
                                doc_topic_prior=0.1,
                                topic_word_prior=0.01,
                               random_state=0)
lda.fit(tf)
n_top_words = 10
tf_feature_names = word_vectorizer.get_feature_names()

topics=lda.transform(tf)
topic = []
for t in topics:
    topic.append("Topic #"+str(list(t).index(np.max(t))))
data['Topics']=topic

pyLDAvis.enable_notebook()
pic = pyLDAvis.sklearn.prepare(lda, tf, word_vectorizer)
pyLDAvis.display(pic)
pyLDAvis.save_html(pic, 'D:\本科课程\大三第一学期\大数据分析\期末大作业\浙江政务网词条分析\数据\LDA主题分析结果\LDA主题分析可视化'+'.html')
