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
➜ python tools/train.py --cfg configs/BPR-MF.yaml 
```
And to specify parameters, change corresponding values in `.yaml` files. Or use command line.
```bash
➜ python tools/train.py --cfg configs/BPR-MF.yaml DATASET.ROOT_DIR "data/amazon-book" 
```

### 4. Experiment Result

i. amazon-book

|MODEL|NDCG|RECALL|PRECISION|HIT RATIO|
|-----|:--:|-----:|-----|:--:|
|RNS|   |      |     |    |
|PNS|   |      |     |    |
|RWS|   |      |     |    |
|DNS| | | | |
|IRGAN| | | | |
|AdvIR| | | | |
|NMRN| | | | |

ii. yelp2018

|MODEL|NDCG|RECALL|PRECISION|HIT RATIO|
|-----|:--:|:---:|:---:|:--:|
|RNS|0.0340|0.0528|0.0132|0.2098|
|PNS|0.0294|0.0462|0.0116|0.1863|
|RWS|0.0340|0.0528|0.0132|0.2098|
|DNS|0.0420|0.0651|0.0162|0.2511|
|IRGAN|0.0342|0.0532|0.0133|0.2121|
|AdvIR|||||
|NMRN|||||

iii. last-fm

|MODEL|NDCG|RECALL|PRECISION|HIT RATIO|
|-----|:--:|-----:|-----|:--:|
|RNS|   |      |     |    |
|PNS|   |      |     |    |
|RWS|   |      |     |    |
|DNS| | | | |
|IRGAN| | | | |
|AdvIR| | | | |
|NMRN| | | | |