# Traffic Sign Classifier
## Overview
* 在這個專案中，我嘗試使用**TensorFlowr**建立三種卷積神經網路(CNN)經典模型(1.LeNet,2.AlexNet,3.GoogLenNet)對交通號誌進行分類。
* 我將訓練並且驗證一個模型，讓他可以對交通號誌進行分類。訓練完模型後，我利用在網路上找到的德國交通號誌圖片來進行額外的測試。

## Dataset 
* 使用的資料集是"German Traffic Sign Dataset"。
* 該資料集經過處理，圖片尺寸是32*32，並且分成訓練集(training)、測試集(test)、驗證集(validation)，交通號誌共有43種類。
![](https://i.imgur.com/chXQBWR.png =70%x)
* 訓練集預覽
![](https://i.imgur.com/YrZUsFM.png)
* 訓練集、驗證集、測試集，資料分布
![](https://i.imgur.com/vVhLStW.png)
* 資料前處理，由於尺寸已經固定，因此只做正規化
![](https://i.imgur.com/C0REX2g.png)
![](https://i.imgur.com/807vAio.png)
## LeNet
* LeNet是在圖像識別中最經典的CNN模型，架構如下:
![](https://i.imgur.com/HgHWxDm.png)
* **My LeNet consists of the following layers:**
![](https://i.imgur.com/loW23JX.png)
* 在LeNet架構中，3個技巧:
    1. Sigmoid: 是個映射函數，將任何變量（這些先寫成 x）映射到 [0, 1] 之間。通常被用來當作機器學習領域 (Machine Learning) 神經網路的激活函數 (Activation Function)。
    2. Mini-batch gradient descent:Mini-batch gradient descent is the combine of batch gradient descent and stochastic gradient descent, it is based on the statistics to estimate the average of gradient of all the training data by a batch of selected samples.
    3. Dropout:是指在每一次訓練時，隨機選取神經元使其不參與訓練，如此一來可以讓模型不過度依賴某些特徵，進而增強模型的泛化能力。避免神經網路的過擬合(overfitting)。
* Define **Evalution function** 
![](https://i.imgur.com/hpZg2ZY.png)

* **Training**
    * 設定超參數(hyperparameters) LEARNING_RATE = 1e-2, EPOCHS = 50, BATCH_SIZE = 128
![](https://i.imgur.com/j7NVOpM.png)
    * 訓練50個epochs
    ![](https://i.imgur.com/bbo9fBH.png)
    ![](https://i.imgur.com/DBod6zg.png)
    ![](https://i.imgur.com/x17WcZJ.png)
* **Result**
    * 訓練集準確度: 95.7%
    * 驗證集準確度: 90.0%
    * 測試集準確度: 88.5%
    ![](https://i.imgur.com/aHyl52G.png)

觀察結果，可以發現訓練集準確度高於驗證集準確度，這裡有一點overfitting。LeNet這個
模型高效並且簡單，是卷積神經網路最經典的模型。 
## AlexNet
* Alex Krizhevsky 於 2012 年提出卷積神經網路 AlexNet，並在同年的 ImageNet LSVRC 競賽中奪得了冠軍，架構如下:
![](https://i.imgur.com/8PC1kw8.png)
* **My AlexNet consists of the following layers:**
![](https://i.imgur.com/JhQuMH4.png =60%x)
![](https://i.imgur.com/UnLF5Bg.png)


* 在AlexNet架構中，4個技巧:
    1. 第一層到第五層是卷積層，其中第一、第二和第五個卷積層後使用池化層，並且採用大小為 3x3、stride 為2 的 Maxpooling。比 LeNet 使用的平均池化，更能保留重要的特徵，並且因為 stride < size (2<3)，因此pooling 可以重疊 (overlap)，能重複檢視特徵，避免重要的特徵被捨棄掉，同時避免過擬和的問題。第六到第八層則是全連接層。
    2. ReLu:全名為 Rectified Linear Unit，中文譯作線性整流函數。是一種類神經網路中活化函數 (activation function)的一種。活化函數主要目的是用來增加類神經網路模型的非線性，讓我們定義的類神經網路可以更加活化學習，避免像是線性函數一樣較為死板。將 LeNet 使用的 Sigmoid 改為 ReLU，可以避免因為神經網路層數過深或是梯度過小，而導致梯度消失 (Vanishing gradient) 的問題。
    3. Adam optimization:是Diederik P. Kingma和Jimmy Ba所提出的一個受歡迎的優化器。
    4. L2 regulization:L2調整用於通過向損失函數添加調整損失來減少overfitting。
* **Training**
    * 設定超參數(hyperparameters) LEARNING_RATE = 5e-4, EPOCHS = 30, BATCH_SIZE = 128, keep_prop = 0.5, LAMBDA = 1e-5
    ![](https://i.imgur.com/08i9d4g.png)
    * 訓練30個epochs
    ![](https://i.imgur.com/vDwiZS8.png)
    ![](https://i.imgur.com/nXV2gq7.png)



* **Result**
    * 訓練集準確度: 100%
    * 驗證集準確度: 96.0%
    * 測試集準確度: 94.6%
    ![](https://i.imgur.com/qWDIXPT.png)
## GoogLeNet
* GoogLeNet是ImageNet LSVRC 2014競賽中的冠軍。一般來說，提升網路效能最直接的辦法就是增加網路深度和寬度，深度指網路層次數量、寬度指神經元數量。但這種方式存在以下問題： （1）引數太多，如果訓練資料集有限，很容易產生過擬合； （2）網路越大、引數越多，計算複雜度越大，難以應用； （3）網路越深，容易出現梯度彌散問題（梯度越往後穿越容易消失），難以優化模型。 
* GoogLeNet團隊提出了Inception網路結構，就是構造一種“基礎神經元”結構，來搭建一個稀疏性、高計算效能的網路結構。GoogLeNet總體架構如下:
![](https://i.imgur.com/QMgWgNn.png)
![](https://i.imgur.com/ZRnwSNL.png)
* **My GoogLeNet consists of the following layers:**
將層數從22層減少成14層，加速訓練時間。網路的詳細信息如下圖:
![](https://i.imgur.com/6nn1Hv2.png =60%x)
![](https://i.imgur.com/cGpHziG.png)
* 在GoogLeNet架構中，一些細節:
    1. Inception V1 通過設計一個稀疏網路結構，但是能夠產生稠密的資料，既能增加神經網路表現，又能保證計算資源的使用效率。谷歌提出了最原始Inception的基本架構
    2. 1x1的卷積核有什麼用呢？ 1x1卷積的主要目的是為了減少維度，還用於修正線性啟用（ReLU）。
    3. Inception Module:The inception module is the core of this architecture, it is driven by two disadvantage of previous architecture: a large amount of parameters which lead to overfitting and dramatically use of computational resources. It's navie implement doesn't have 1x1 conv before/after 3x3 conv, 5x5 conv and max pooling layer. The reason why adding 1x1 convolutional layer is that it can reduce the depth of the output from previous layer
    4. Overlapping pooling:The normal pooling operation is with kernel size = 2 and stride = 2, and the overlapping pooling means kernel size > stride, like kernel size = 3 and stride = 2, thus there will be overlapping fields. According to Traffic Sign Recognition with Multi-Scale Convolutional Networks, overlapping pooling can slightly reduce the error rates compared to non-overlapping and make the model more difficult to overfit.
* **Training**
    * 設定超參數(hyperparameters) LEARNING_RATE = 4e-4, EPOCHS = 35, BATCH_SIZE = 128, keep_prop = 0.5, LAMBDA = 1e-5
    ![](https://i.imgur.com/LyiopBl.png)
    * 訓練35個epochs
    ![](https://i.imgur.com/jMG4Am0.png)
    ![](https://i.imgur.com/4tVyX21.png)



* **Result**
    * 訓練集準確度: 100%
    * 驗證集準確度: 98.5%
    * 測試集準確度: 97.1%
    ![](https://i.imgur.com/5QRV0dE.png)

## Discussion
* 在這個project中，使用三種圖像分類的CNN架構來辨識德國的交通號誌(GTSRB)。從LeNet(7層)到AlexNet(8層)再到GoogleNet(22層->14層)，可以發現CNN的層數加深會帶來更好的結果。此外也使用一些方法跟技巧來訓練模型，例如:mini-batch gradient descent, Adam optimization, L2 regularization, learning rate decay 等等.
* 在project的最後，我在網路上找了10個德國的交通號誌來測試我的模型，結果是成功的，這10個交通號誌都得到了對的識別。
* AlexNet模型的結果
![](https://i.imgur.com/d4BHgqZ.png)
* GoogLeNet模型結果
![](https://i.imgur.com/O5fnR6L.png)










## Authors
[N96094196 張維峻](https://hackmd.io/@po6GeGxZSG-RrxfU2_EF0A/ByBaKElF_)