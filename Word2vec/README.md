# Word2vec
Word2vec是一个用于处理文本的双层神经网络。它的输入是文本语料，输出则是一组向量：该语料中词语的特征向量。虽然Word2vec并不是[深度神经网络](https://en.wikipedia.org/wiki/Deep_learning)，但它可以将文本转换为深度神经网络能够理解的数值形式。

本项目是Word2vec的Java实现，可计算训练样本中词向量之间的距离，同时反映了词与词之间的相似程度。

感谢Google提出的[Word2vec](https://code.google.com/archive/p/word2vec)！

# Requirements
- [JDK 7 ](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html)
- [Maven](https://maven.apache.org/)

# Basic Usage
**准备输入样本** ：

将输入样本`input.txt`放入`library`文件夹下。

*本项目中使用的样本为官方提供的[text8](http://mattmahoney.net/dc/text8.zip)，该样本是由wikipedia上的前100，000，000个字符，并将特殊符号、非英文字符去除，大写自负转化为小写字符，数字转化为对应英语单词后得到的。*

**根据输入样本训练模型** ：

注释掉`//根据模型计算词向量的距离关系`后计算与当前词距离最近的词距离的代码，然后运行 `\src\main\java\com\ansj\vec\Word2VEC`类文件的main方法。

生成的模型文件为`javaVector`，保存在`library`文件夹下。

 **根据训练好的模型生成距离目标词汇距离最近的词** ：

注释掉`//根据模型计算词向量的距离关系`之前main方法中的代码，然后运行`\src\main\java\com\ansj\vec\Word2VEC`类文件的main方法。