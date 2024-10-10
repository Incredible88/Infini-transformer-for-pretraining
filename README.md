# Infini-transformer-for-pretraining

Based on  Infini-transformer model, to solve the problem of excessive memory consumption of long sequences in the original transformer, the input sequence is divided into blocks of specified length and input sequentially, while maintaining the connection between the previous and the next blocks through the memory component, which is also called infini-transformer.

Reproduced the infini-attention structure in the article: https://arxiv.org/abs/2404.07143

![image](https://github.com/user-attachments/assets/9a685e2a-9074-4dc8-ab85-cc8159ec13af)
