# Problem  Records

> 模型大小最终应该是100M - 200M 之间


1. vocab_size 如何选择，对于这个小模型，大量中文语料，vocab_size 选择多大合适？
> chatgpt 推荐 32k，
- vocab 太小 → token 过碎、序列变长、attention 浪费算力；
- vocab 太大 → embedding 吃掉参数、小模型表达能力被挤死；

2. 数据集如何选择，数据量多大合适？
> chatgpt 推荐 
| 模型规模 | 够跑通/做对比 |  合适（推荐） | 更充足（上限向） |
| ---- | ------: | ------: | -------: |
| 100M |  0.5–1B |  **2B** |       5B |
| 300M |    1–3B |  **6B** |      12B |
| 700M |    3–6B | **14B** |      25B |
| 1B   |   5–10B | **20B** |   35–50B |

pretrain 数据集选择
> 最开始选择了CLUEcorpusSmall,发现大量短文本，感觉不太合适,后从hf上找了几个大数据集做pretrain数据集
