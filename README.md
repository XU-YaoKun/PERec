# PERec
Several algorithms for personalized recommendation. Also, this repo contains implementation of all baselines for [Reinforced Negative Sampling over Knowledge Graph for Recommendation](http://staff.ustc.edu.cn/~hexn/papers/www20-KGPolicy.pdf).
 
These instructions will get you a copy of the project up and running on your local machine for development and testing purpose.

### 1. Data and Source Code
i. Get source code
```bash
➜ git clone https://github.com/XU-YaoKun/PERec.git
➜ cd PERec 
```
ii. Get datasets
```bash
➜ wget https://github.com/XU-YaoKun/PERec/releases/download/v1.0/data.zip
➜ unzip data.zip
```
### 2. Environment
i. install `PERec` at local machine
```bash
# Remember to add develop so that all modifications of python files could take effects.
➜ python setup.py develop 
```
ii. install dependences

Anaconda is recommended to manage all dependences. So create a new env and then install required packages.
```bash
➜ conda create -n perec-env python=3.6
➜ conda activate perec-env 
➜ pip install -r requirements.txt 
```
### 3. Train

To train different models, use corresponding config file. For example, to train `BPRMF`, using the following command,
```bash
➜ python main.py --cfg configs/BPR-MF.yaml 
```
And to specify parameters, change corresponding values in `.yaml` files. Or use command line.
```bash
➜ python main.py --cfg configs/BPR-MF.yaml DATASET.ROOT_DIR "data/amazon-book" 
```

### 4. Experiment Result
i. yelp2018

|MODEL|NDCG|RECALL|PRECISION|HIT RATIO|
|:-----:|:--:|:---:|:---:|:--:|
|RNS|0.0340|0.0528|0.0132|0.2098|
|PNS|0.0294|0.0462|0.0116|0.1863|
|RWS|0.0340|0.0528|0.0132|0.2098|
|DNS|0.0420|0.0651|0.0162|0.2511|
|IRGAN|0.0349|0.0558|0.0135|0.2200|
|AdvIR|0.0405|0.0627|0.0156|0.2416|
|NMRN|0.0273|0.0433|0.0111|0.1810|

Other experimental results can be found in our paper, or can be obtained using this code repo. Note that we use a different method to calculate `NDCG` here, so it is slightly different from what presented in paper.

Some parameters, like `regs` and `lr`, can be tuned in order to get a better performance.

Please note that, to test these models in our datasets, we did some sorts of simplification. And also, some original codes have not been released, so we implement them according to corresponding papers. To find the original paper and code, please check references. 

Also, when using pretrained model to initialize parameters, better performance can be obtained. But in this repo, all models are trained from scratch.

### 5. Reference
[1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI Press, 2009.

[2] Zhang, Weinan et al. “Optimizing top-n collaborative filtering via dynamic negative item sampling.” SIGIR (2013).

[3] Wang, Jun et al. “IRGAN.” Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval - SIGIR  ’17 (2017): n. pag. Crossref. Web.

[4] Park, Dae Hoon, and Yi Chang. "Adversarial sampling and training for semi-supervised information retrieval." The World Wide Web Conference. 2019.

[5] Wang, Qinyong, et al. "Neural memory streaming recommender networks with adversarial training." Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.