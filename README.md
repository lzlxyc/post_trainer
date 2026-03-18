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



3B: 82
3B_GRPO:90
SFT:88

0.5B:
origin: 0.47
sft(2epoch): 0.15
GRPO(800steps):0.5
