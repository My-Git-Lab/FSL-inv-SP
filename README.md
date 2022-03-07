# FSL-inv-SP: A new self-potential inversion method based on few-shot learning
Lin-Jin Yang, Chang-Xin Nai, Guo-Bin Liu, Kai-Lun Lai, Shuo-Yang Gao, Kai-Da Zheng

This is the code implemented by Lin-Jin Yang based on ***Pytorch***. You can use this code, combined with the pole-pole device method, to inverse the self-potential data and get the approximate location of the source. The code does not require high computing resources. Even without a good GPU, you can use an ordinary CPU to run this program.

## How to get the result you want
**1):** use ***pre-training.py*** to get trained model.

Please note that the training data is produced by ***COMSOL Multiphysics 5.5***. Before running the program, please modify the path part of the code to point to the relevant data in your computer. For example:
```
pos_1 = np.array(pd.read_excel("YOUR PATH"))
pos_2 = np.array(pd.read_excel("YOUR PATH"))
...
torch.save(net1, "YOUR PATH AND MODEL.pth")
...
```
You can use your own training data and different hyper-parameter combinations. At the same time, for the convenience of readers' reference, we uploaded our own pre-trained model, and you can also use ***pre-trained model.pth***

**2):** use ***fine-tuning.py*** to get more accurate results.

If you want to predict the real scene data more accurately, you also need to fine tune the model obtained in the previous step. Similarly, you need to modify the path to point to the corresponding file:
```
net1 = torch.load('YOUR PRE-TRAINED MODEL.pth')
...
```

## Saving results and drawing

We recommend recording the loss in different stages, and then presenting a simple trend chart, so that we can observe the effect of the network and better adjust the parameters. The data will be saved in the form of .mat, so that you can draw more finely in ***Matlab***.
```
io.savemat('total_regular_loss.mat', {'array': data_loss})
...
plt.plot(data_loss)
...
```

## A test

If you want to quickly view the results of this method, we also provide a small case. You can run ***test.py*** directly in ***PyCharm Community Edition*** and you should be able to get the results as same as Fig.test.jpg:
