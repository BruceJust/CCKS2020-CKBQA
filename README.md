# CCKS2020-CKBQA

**任务描述**

CCKS2020：新冠知识图谱问答评测。竞赛地址：https://biendata.com/competition/ccks_2020_7_4/

这个竞赛属于中文知识图谱自然语言问答任务（CKBQA），即输入一句中文问题，问答系统从给定知识库中选择若干实体或属性值作为该问题的答案。竞赛中的问题均为客观事实型，不包含主观因素。理解并回答问题的过程中需要进行实体识别、关系抽取、语义解析等子任务。

竞赛任务所使用的新冠开放知识图谱来源于 OpenKG 的新冠专题（[http://openkg.cn/group/coronavirus](http://openkg.cn/group/coronavirus）)），此外还包括开放领域知识库PKUBASE一起作为问答任务的依据。 

**参考资料**

CCKS2019的评测论文：https://conference.bj.bcebos.com/ccks2019/eval/webpage/index.html

CCKS2019的CKBQA参赛总结：https://blog.csdn.net/zzkv587/article/details/102954876

CCKS2019的CKBQA第4名代码：https://github.com/duterscmy/ccks2019-ckbqa-4th-codes



**数据说明**

pkubase-complete.txt       知识图谱数据 3G

task1-4_train_2020.txt      4000行训练数据  问题及对应

sqltask1-4_valid_2020.questions      1529个测试问题

pkubase-mention2ent.txt         多义实体与其候选实体