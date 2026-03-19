# 使用
```shell
git clone https://github.com/lzlxyc/post_trainer.git
cd env_install 
bash installer.sh
cd ../scripts
```
## 训练
```shell
bash train.sh
```
## 验证
```shell
bash eval.sh
```


> 如果训练flash下载不来，直接进入installer.sh里面找链接手动下载
>


# 参考项目
https://github.com/QunBB/DeepLearning/tree/main/llms/train/deepseek-train





0.5B:
origin: 47
sft(2epoch): 15
GRPO(800steps):50

1.5B
origin:64
SFT:58
GRPO:0.74


3B: 
origin:82
GRPO:90
SFT:88

4B:
origin:0.59
