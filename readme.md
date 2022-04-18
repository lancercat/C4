
# CFOR: Character-First Open-set Text Recognition


Mostly done. After a few grammar checks, this repo would be ready for review.

Training data, code, and manual will be released after acceptance before the camera-ready phase to maintain anonymity. 

This is NOT an ijcai reject. It failed to make the DDL and is not submitted.

## Handbook
See handbook.pdf

In case of failing links, we have reproduced the experiments in advance. The collected screenshots are in the document. 

Let's hope M$ to go nicely on the shared links :-)

We will release training scripts, training data, co-training data, and corresponding data building scripts after acceptance. 


## Data

### Data Collected from the Internet
The data we collected, the data collection methodology, and data sources are included in the archive (Athena.zip)

### Datasets

The Chinese Character evaluation datasets are skipped because the authors have some trouble accessing a reliable anonymous and reputable cloud storage service (Onedrive is what is currently accessible to us, which has a quota of 5Gib per free account). We will release them if the paper gets accepted, which nulls the necessity to remain anonymous.

#### Evaluation Dataset
https://1drv.ms/u/s!Ah3A6cw9Sjd2b8dDQPlblSF4TrM

#### Training Dataset
Will be released after the paper gets accepted. 

## Trained models
1. Download the following links

![image](https://user-images.githubusercontent.com/59994105/163527896-4aaa6e86-f9f5-4b11-8bcc-4d3185864a39.png)

2. Unzip with `for i in $(ls); do unzip $i; done;`

### Ablative:
Base model

https://1drv.ms/u/s!Ah3A6cw9Sjd2a3NLfus1uh7Eyr8

CIL only

https://1drv.ms/u/s!Ah3A6cw9Sjd2bSf6wbqD2uDvkfc

ICL only and Full model (ICL is bottlenecked by HDD read, so we use the same data to train 2 models)

https://1drv.ms/u/s!Ah3A6cw9Sjd2ag2arR5nuPgxMJ8

### Perfomance:

#### Open-Set
Full model:

Regular

https://1drv.ms/u/s!Ah3A6cw9Sjd2ag2arR5nuPgxMJ8

Large

https://1drv.ms/u/s!Ah3A6cw9Sjd2bBwO4YNefPPwztM

#### Close-Set:
Regular

https://1drv.ms/u/s!Ah3A6cw9Sjd2adIj25OkS1Lontg

Large 

https://1drv.ms/u/s!Ah3A6cw9Sjd2aCYk4s1ZVu0HBGQ

#### Character

Regular

https://1drv.ms/u/s!Ah3A6cw9Sjd2bnz3VeLquz2-Sf4




## Extra details 
### Computing Infanstructure 
Laptop with i5-9400 and 2070 Mobile (The authors do not have access to Nanos, TX2s, or AGXs.)
![image](https://user-images.githubusercontent.com/59994105/163657930-47f7c495-f32f-4a1b-b285-5a0469b1efc7.png)



### Energy Cost
~300 FPS multi-batched @ Less than 230W.
