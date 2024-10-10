import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from Pretraining.transformer import InfiniTransformer
from Pretraining.embedding_esm3 import Esm3Embedding
import os
from Bio import SeqIO
import wandb
from tqdm import tqdm


wandb.login(key="5f5f94d3de9157cf146ad88ecc4e0518a7a7549e")
# 初始化 wandb
wandb.init(
    project="ESM3_Pretraining",
    config={
        "learning_rate": 1e-4,
        "epochs": 10,
        "batch_size": 2,
        "architecture": "InfiniTransformer",
        # 你可以添加更多的配置参数
    }
)

# 定义文件路径
fasta_file = "/home/share/huadjyin/home/yinpeng/ljl/data/METAdataset/data/non_anno/vagino_nonanno_seq.fast"

# 初始化一个空的序列列表
sequences = []

# 解析FASTA文件并提取序列
for record in SeqIO.parse(fasta_file, "fasta"):
    # 提取序列并添加到列表中
    sequences.append(str(record.seq))

# 测试前一百条
sequences = sequences[:500]
# 找到最长的 序列 3259
# def longest_string_length(strings):
#     return max(len(s) for s in strings)
#
# print(longest_string_length(sequences))
# print(sequences[:5])
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainedModel(nn.Module):

    def __init__(
            self,
            embedding_dim: int,
            num_layers: int,
            dim_hidden: int,
            dim_key: int,
            dim_value: int,
            num_heads: int,
            segment_len: int,
            update: str = "delta",
            causal: bool = True,
            init_state_learnable: bool = False,
            dropout: float = 0.1
    ):
        super(PretrainedModel, self).__init__()

        transformers = []
        for _ in range(num_layers):
            transformers.append(
                InfiniTransformer(
                    dim_input=embedding_dim,
                    dim_hidden=dim_hidden,
                    dim_key=dim_key,
                    dim_value=dim_value,
                    num_heads=num_heads,
                    segment_len=segment_len,
                    update=update,
                    causal=causal,
                    positional_embedder=None,  # Assuming no positional embeddings are needed
                    init_state_learnable=init_state_learnable,
                    dropout=dropout
                )
            )
        self.transformers = nn.ModuleList(transformers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for transformer in self.transformers:
            x = transformer(x)
        return x


def train_model(
        model: PretrainedModel,
        dataloader_train: DataLoader,
        dataloader_val: DataLoader,
        epochs: int,
        device: str
):
    model = model.train().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for ix, batch in enumerate(dataloader_train):
            batch = batch.to(device)  # 将 batch 迁移到指定设备
            preds = model(batch)

            target = batch[:, 1:].clone().to(device)  # 确保 target 在正确的设备上
            loss = loss_fn(input=preds[:, :-1, :], target=target)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            running_loss += loss.detach().cpu().item()

            print(
                f'Epoch: {epoch + 1}/{epochs} ({ix + 1}/{len(dataloader_train)})  |  Training Loss: {loss.detach().cpu().item():.6f}\r',
                end="")

        # 记录每个 epoch 的平均训练损失
        avg_train_loss = running_loss / len(dataloader_train)
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

        lr_schedule.step()

        with torch.no_grad():
            total_loss = 0.0
            num_obs = 0

            for batch in dataloader_val:
                batch = batch.to(device)
                preds = model(batch)
                target = batch[:, 1:].clone().to(device)  # 确保 target 在正确的设备上

                total_loss += loss_fn(input=preds[:, :-1, :], target=target).detach().cpu().item() * batch.size(0)
                num_obs += batch.size(0)

            val_loss = total_loss / num_obs
            print(f'\nEpoch: {epoch + 1}/{epochs}  |  Validation Loss: {val_loss:.6f}')

            # 记录验证损失
            wandb.log({"epoch": epoch + 1, "val_loss": val_loss})

    return model


embedding_dim = 1536
num_layers = 6
dim_hidden = 512
dim_key = 64
dim_value = 64
num_heads = 6
segment_len = 128
batch_size = 10
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Initialize Esm3Embedding instance

# reduction_layer = torch.nn.Linear(1536, 512).to(device)

all_embeddings = []

# 初始化Esm3Embedding实例
esm3_embedder = Esm3Embedding(pooling="mean")

# for sequence in tqdm(sequences, desc="Generating embeddings"):
#
#     batch = {"sequence": [sequence]}
#     embedding = esm3_embedder.get_embedding(batch)
#     # reduced_embedding = reduction_layer(embedding)
#     all_embeddings.append(embedding)
#     # 清理GPU缓存
#     torch.cuda.empty_cache()
#
# # 合并所有的嵌入
# all_embeddings = torch.cat(all_embeddings, dim=0)
#
# # 创建数据加载器
# dataloader_train = DataLoader(all_embeddings, batch_size=1, shuffle=True)
# dataloader_val = DataLoader(all_embeddings, batch_size=1, shuffle=False)

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, max_len, embedding_dim):
        self.embeddings = embeddings
        self.max_len = max_len
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        # Padding the embedding to (max_len, embedding_dim)
        padded_embedding = torch.zeros(self.max_len, self.embedding_dim)
        padded_embedding[:embedding.shape[1], :] = embedding.squeeze(0)
        return padded_embedding

# Calculate the maximum sequence length, which is 4096 in this case
max_len = 1280
embedding_dim = 1536

all_embeddings = []
for sequence in tqdm(sequences, desc="Generating embeddings"):
    batch = {"sequence": [sequence]}
    embedding = esm3_embedder.get_embedding(batch)
    all_embeddings.append(embedding)

# Creating the dataset and dataloaders with padded embeddings
dataset = EmbeddingDataset(all_embeddings, max_len=max_len, embedding_dim=embedding_dim)
dataloader_train = DataLoader(dataset, batch_size=1, shuffle=True)
dataloader_val = DataLoader(dataset, batch_size=1, shuffle=False)


model = InfiniTransformer(
    dim_input=embedding_dim,
    num_layers=num_layers,
    dim_hidden=dim_hidden,
    dim_key=dim_key,
    dim_value=dim_value,
    num_heads=num_heads,
    activation="ffngeglu",
    segment_len=segment_len,
    update="delta",
    causal=True,
    init_state_learnable=False,
    dropout=0.1
)

# 先打印模型结构和参数
# print("Model structure:")
# print(model)  # 打印模型结构
#
# print("\nModel parameters:")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.shape}")

model.to(device)
for param in model.parameters():
    param.data = param.data.to(device)

# 开始训练
trained_model = train_model(
    model=model,
    dataloader_train=dataloader_train,
    dataloader_val=dataloader_val,
    epochs=10,
    device=device
)
# 清理GPU缓存
torch.cuda.empty_cache()
