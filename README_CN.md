# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [BERT概述](#bert概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [预训练模型](#预训练模型)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [预训练](#预训练)
        - [微调与评估](#微调与评估)
    - [选项及参数](#选项及参数)
        - [选项](#选项)
        - [参数](#参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [Ascend处理器上运行](#ascend处理器上运行)
        - [分布式训练](#分布式训练)
            - [Ascend处理器上运行](#ascend处理器上运行-1)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [Ascend处理器上运行后评估cola数据集](#ascend处理器上运行后评估cola数据集)
            - [Ascend处理器上运行后评估cluener数据集](#ascend处理器上运行后评估cluener数据集)
            - [Ascend处理器上运行后评估chineseNer数据集](#ascend处理器上运行后评估chinesener数据集)
            - [Ascend处理器上运行后评估msra数据集](#ascend处理器上运行后评估msra数据集)
            - [Ascend处理器上运行后评估squad v1.1数据集](#ascend处理器上运行后评估squad-v11数据集)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法-2)
        - [结果](#结果)
    - [导出onnx模型与推理](#导出onnx模型与推理)
    - [模型描述](#模型描述)
    - [性能](#性能)
        - [预训练性能](#预训练性能)
            - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)
- [FAQ](#faq)

<!-- /TOC -->

# BERT概述

BERT网络由谷歌在2018年提出，该网络在自然语言处理领域取得了突破性进展。采用预训练技术，实现大的网络结构，并且仅通过增加输出层，实现多个基于文本的任务的微调。BERT的主干代码采用Transformer的Encoder结构。引入注意力机制，使输出层能够捕获高纬度的全局语义信息。预训练采用去噪和自编码任务，即掩码语言模型（MLM）和相邻句子判断（NSP）。无需标注数据，可对海量文本数据进行预训练，仅需少量数据做微调的下游任务，可获得良好效果。BERT所建立的预训练加微调的模式在后续的NLP网络中得到了广泛的应用。

[论文](https://arxiv.org/abs/1810.04805):  Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.[BERT：深度双向Transformer语言理解预训练](https://arxiv.org/abs/1810.04805)). arXiv preprint arXiv:1810.04805.

[论文](https://arxiv.org/abs/1909.00204):  Junqiu Wei, Xiaozhe Ren, Xiaoguang Li, Wenyong Huang, Yi Liao, Yasheng Wang, Jiashu Lin, Xin Jiang, Xiao Chen, Qun Liu.[NEZHA：面向汉语理解的神经语境表示](https://arxiv.org/abs/1909.00204). arXiv preprint arXiv:1909.00204.

# 模型架构

BERT的主干结构为Transformer。对于BERT_base，Transformer包含12个编码器模块，每个模块包含一个自注意模块，每个自注意模块包含一个注意模块。对于BERT_NEZHA，Transformer包含24个编码器模块，每个模块包含一个自注意模块，每个自注意模块包含一个注意模块。BERT_base和BERT_NEZHA的区别在于，BERT_base使用绝对位置编码生成位置嵌入向量，而BERT_NEZHA使用相对位置编码。

# 数据集

- 生成预训练数据集
    - 下载[zhwiki](https://dumps.wikimedia.org/zhwiki/)或[enwiki](https://dumps.wikimedia.org/enwiki/)数据集进行预训练，
    - 使用[WikiExtractor](https://github.com/attardi/wikiextractor)提取和整理数据集中的文本，使用步骤如下：
        - pip install wikiextractor
        - python -m wikiextractor.WikiExtractor -o <output file path> -b <output file size> <Wikipedia dump file>
    - `WikiExtarctor`提取出来的原始文本并不能直接使用，还需要将数据集预处理并转换为TFRecord格式。详见[BERT](https://github.com/google-research/bert#pre-training-with-bert)代码仓中的create_pretraining_data.py文件，同时下载对应的vocab.txt文件, 如果出现AttributeError: module 'tokenization' has no attribute 'FullTokenizer’，请安装bert-tensorflow。
- 生成下游任务数据集
    - 下载数据集进行微调和评估，如中文实体识别任务[CLUENER](https://github.com/CLUEbenchmark/CLUENER2020)、中文文本分类任务[TNEWS](https://github.com/CLUEbenchmark/CLUE)、中文实体识别任务[ChineseNER](https://github.com/zjy-ucas/ChineseNER)、英文问答任务[SQuAD v1.1训练集](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)、[SQuAD v1.1验证集](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)、英文分类任务集合[GLUE](https://gluebenchmark.com/tasks)等。
    - 将数据集文件从JSON格式转换为TFRecord格式。详见[BERT](https://github.com/google-research/bert)代码仓中的run_classifier.py或run_squad.py文件。

# 预训练模型

我们提供了一些预训练权重以供使用

- [Bert-base-zh](https://download.mindspore.cn/model_zoo/r1.3/bert_base_ascend_v130_zhwiki_official_nlp_bs256_acc91.72_recall95.06_F1score93.36/), 在128句长的中文wiki数据集上进行了训练
- [Bert-large-zh](https://download.mindspore.cn/model_zoo/r1.3/bert_large_ascend_v130_zhwiki_official_nlp_bs3072_loss0.8/), 在128句长的中文wiki数据集上进行了训练
- [Bert-large-en](https://download.mindspore.cn/model_zoo/r1.3/bert_large_ascend_v130_enwiki_official_nlp_bs768_loss1.1/), 在512句长的英文wiki数据集上进行了训练

# 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

从官网下载安装MindSpore之后，您可以按照如下步骤进行训练和评估：


- 在GPU上运行

```bash

# 单机运行预训练示例

bash run_standalone_pretrain_for_gpu.sh 0 1 /path/cn-wiki-128

# 分布式运行预训练示例

bash scripts/run_distributed_pretrain_for_gpu.sh 8 40 /path/cn-wiki-128

# 运行微调和评估示例

- 如需运行微调任务，请先准备预训练生成的权重文件（ckpt）。
- 在`task_[DOWNSTREAM_TASK]_config.yaml`中设置BERT网络配置和优化器超参。

- 分类任务：在scripts/run_classifier_gpu.sh中设置任务相关的超参。
- 运行`bash scripts/run_classifier_gpu.sh [DEVICE_ID]`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_classifier_gpu.sh DEVICE_ID(optional)

- NER任务：在scripts/run_ner_gpu.sh中设置任务相关的超参。
- 运行`bash scripts/run_ner_gpu.sh [DEVICE_ID]`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_ner_gpu.sh DEVICE_ID(optional)

- SQUAD任务：在scripts/run_squad_gpu.sh中设置任务相关的超参。
-运行`bash scripts/run_squad_gpu.sh [DEVICE_ID]`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_squad_gpu.sh DEVICE_ID(optional)
```


```text
For pretraining, schema file contains ["input_ids", "input_mask", "segment_ids", "next_sentence_labels", "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"].

For ner or classification task, schema file contains ["input_ids", "input_mask", "segment_ids", "label_ids"].

For squad task, training: schema file contains ["start_positions", "end_positions", "input_ids", "input_mask", "segment_ids"], evaluation: schema file contains ["input_ids", "input_mask", "segment_ids"].

`numRows` is the only option which could be set by user, other values must be set according to the dataset.

For example, the schema file of cn-wiki-128 dataset for pretraining shows as follows:
{
    "datasetType": "TF",
    "numRows": 7680,
    "columns": {
        "input_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [128]
        },
        "input_mask": {
            "type": "int64",
            "rank": 1,
            "shape": [128]
        },
        "segment_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [128]
        },
        "next_sentence_labels": {
            "type": "int64",
            "rank": 1,
            "shape": [1]
        },
        "masked_lm_positions": {
            "type": "int64",
            "rank": 1,
            "shape": [20]
        },
        "masked_lm_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [20]
        },
        "masked_lm_weights": {
            "type": "float32",
            "rank": 1,
            "shape": [20]
        }
    }
}
```

## 脚本说明

## 脚本和样例代码

```shell
.
└─bert
  ├─README.md
  ├─README_CN.md
  ├─scripts
    ├─ascend_distributed_launcher
        ├─__init__.py
        ├─hyper_parameter_config.ini          # 分布式预训练超参
        ├─get_distribute_pretrain_cmd.py      # 分布式预训练脚本
        --README.md
    ├─run_classifier_gpu.sh                   # GPU设备上单机分类器任务shell脚本
    ├─run_ner.sh                              # GPU设备上单机NER任务shell脚本
    ├─run_squad_gpu.sh                        # GPU设备上单机SQUAD任务shell脚本
    ├─run_distributed_pretrain_gpu.sh         # GPU设备上分布式预训练shell脚本
    └─run_standaloned_pretrain_gpu.sh         # GPU设备上单机预训练shell脚本
  ├─src
    ├─model_utils
      ├── config.py                           # 解析 *.yaml参数配置文件
      ├── devcie_adapter.py                   # 区分本地/ModelArts训练
      ├── local_adapter.py                    # 本地训练获取相关环境变量
      └── moxing_adapter.py                   # ModelArts训练获取相关环境变量、交换数据
    ├─__init__.py
    ├─assessment_method.py                    # 评估过程的测评方法
    ├─bert_for_finetune.py                    # 网络骨干编码
    ├─bert_for_pre_training.py                # 网络骨干编码
    ├─bert_model.py                           # 网络骨干编码
    ├─finetune_data_preprocess.py             # 数据预处理
    ├─cluner_evaluation.py                    # 评估线索生成工具
    ├─CRF.py                                  # 线索数据集评估方法
    ├─dataset.py                              # 数据预处理
    ├─finetune_eval_model.py                  # 网络骨干编码
    ├─sample_process.py                       # 样例处理
    ├─utils.py                                # util函数
  ├─pretrain_config.yaml                      # 预训练参数配置
  ├─task_ner_config.yaml                      # 下游任务_ner 参数配置
  ├─task_classifier_config.yaml               # 下游任务_classifier 参数配置
  ├─task_squad_config.yaml                    # 下游任务_squad 参数配置
  ├─pretrain_eval.py                          # 训练和评估网络
  ├─run_classifier.py                         # 分类器任务的微调和评估网络
  ├─run_ner.py                                # NER任务的微调和评估网络
  ├─run_pretrain.py                           # 预训练网络
  └─run_squad.py                              # SQUAD任务的微调和评估网络
```

## 脚本参数

### 预训练

```shell
用法：run_pretrain.py  [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                        [--enable_save_ckpt ENABLE_SAVE_CKPT] [--device_target DEVICE_TARGET]
                        [--enable_lossscale ENABLE_LOSSSCALE] [--do_shuffle DO_SHUFFLE]
                        [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                        [--accumulation_steps N]
                        [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                        [--load_checkpoint_path LOAD_CHECKPOINT_PATH]
                        [--save_checkpoint_steps N] [--save_checkpoint_num N]
                        [--data_dir DATA_DIR] [--schema_dir SCHEMA_DIR] [train_steps N]

选项：
    --device_target            代码实现设备，可选项为Ascend或GPU。默认为Ascend
    --distribute               是否多卡预训练，可选项为true（多卡预训练）或false。默认为false
    --epoch_size               轮次，默认为1
    --device_num               使用设备数量，默认为1
    --device_id                设备ID，默认为0
    --enable_save_ckpt         是否使能保存检查点，可选项为true或false，默认为true
    --enable_lossscale         是否使能损失放大，可选项为true或false，默认为true
    --do_shuffle               是否使能轮换，可选项为true或false，默认为true
    --enable_data_sink         是否使能数据下沉，可选项为true或false，默认为true
    --data_sink_steps          设置数据下沉步数，默认为1
    --accumulation_steps       更新权重前梯度累加数，默认为1
    --save_checkpoint_path     保存检查点文件的路径，默认为""
    --load_checkpoint_path     加载检查点文件的路径，默认为""
    --save_checkpoint_steps    保存检查点文件的步数，默认为1000
    --save_checkpoint_num      保存的检查点文件数量，默认为1
    --train_steps              训练步数，默认为-1
    --data_dir                 数据目录，默认为""
    --schema_dir               schema.json的路径，默认为""
```

### 微调与评估

```shell
用法：run_ner.py   [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                    [--assessment_method ASSESSMENT_METHOD] [--use_crf USE_CRF] [--with_lstm WITH_LSTM]
                    [--device_id N] [--epoch_num N] [--vocab_file_path VOCAB_FILE_PATH]
                    [--label2id_file_path LABEL2ID_FILE_PATH]
                    [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                    [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                    [--train_data_file_path TRAIN_DATA_FILE_PATH]
                    [--eval_data_file_path EVAL_DATA_FILE_PATH]
                    [--schema_file_path SCHEMA_FILE_PATH]
选项：
    --device_target                   代码实现设备，可选项为Ascend或GPU。默认为Ascend
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --assessment_method               评估方法，可选项为f1或clue_benchmark
    --use_crf                         是否采用CRF来计算损失，可选项为true或false
    --with_lstm                       是否在bert后接lstm子网络提升性能，可选项为true或false
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --vocab_file_path                 BERT模型训练的词汇表
    --label2id_file_path              标注文件，文件中的标注名称必须与原始数据集中所标注的类型名称完全一致
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练BERT模型
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存训练数据的TFRecord文件，如train.tfrecord文件
    --eval_data_file_path             如采用f1来评估结果，则为TFRecord文件保存预测；如采用clue_benchmark来评估结果，则为JSON文件保存预测
    --dataset_format                  数据集格式，支持tfrecord和mindrecord格式
    --schema_file_path                模式文件保存路径

用法：run_squad.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                    [--device_id N] [--epoch_num N] [--num_class N]
                    [--vocab_file_path VOCAB_FILE_PATH]
                    [--eval_json_path EVAL_JSON_PATH]
                    [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                    [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                    [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH]
                    [--train_data_file_path TRAIN_DATA_FILE_PATH]
                    [--eval_data_file_path EVAL_DATA_FILE_PATH]
                    [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   代码实现设备，可选项为Ascend或GPU。默认为Ascend
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --num_class                       分类数，SQuAD任务通常为2
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --vocab_file_path                 BERT模型训练的词汇表
    --eval_json_path                  保存SQuAD任务开发JSON文件的路径
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练BERT模型
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存SQuAD训练数据的TFRecord文件，如train1.1.tfrecord
    --eval_data_file_path             用于保存SQuAD预测数据的TFRecord文件，如dev1.1.tfrecord
    --schema_file_path                模式文件保存路径

usage: run_classifier.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                         [--assessment_method ASSESSMENT_METHOD] [--device_id N] [--epoch_num N] [--num_class N]
                         [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                         [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                         [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH]
                         [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                         [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                         [--train_data_file_path TRAIN_DATA_FILE_PATH]
                         [--eval_data_file_path EVAL_DATA_FILE_PATH]
                         [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   任务运行的目标设备，可选项为Ascend或GPU
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --assessment_method               评估方法，可选项为accuracy、f1、mcc、spearman_correlation
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --num_class                       标注类的数量
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练BERT模型）
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存训练数据的TFRecord文件，如train.tfrecord文件
    --eval_data_file_path             用于保存预测数据的TFRecord文件，如dev.tfrecord
    --schema_file_path                模式文件保存路径
```

## 选项及参数

可以在yaml配置文件中分别配置预训练和下游任务的参数。

### 选项

```text
config for lossscale and etc.
    bert_network                    BERT模型版本，可选项为base或nezha，默认为base
    batch_size                      输入数据集的批次大小，默认为32
    loss_scale_value                损失放大初始值，默认为2^32
    scale_factor                    损失放大的更新因子，默认为2
    scale_window                    损失放大的一次更新步数，默认为1000
    optimizer                       网络中采用的优化器，可选项为AdamWerigtDecayDynamicLR、Lamb、或Momentum，默认为Lamb
```

### 参数

```text
数据集和网络参数（预训练/微调/评估）：
    seq_length                      输入序列的长度，默认为128
    vocab_size                      各内嵌向量大小，需与所采用的数据集相同。默认为21136
    hidden_size                     BERT的encoder层数，默认为768
    num_hidden_layers               隐藏层数，默认为12
    num_attention_heads             注意头的数量，默认为12
    intermediate_size               中间层数，默认为3072
    hidden_act                      所采用的激活函数，默认为gelu
    hidden_dropout_prob             BERT输出的随机失活可能性，默认为0.1
    attention_probs_dropout_prob    BERT注意的随机失活可能性，默认为0.1
    max_position_embeddings         序列最大长度，默认为512
    type_vocab_size                 标记类型的词汇表大小，默认为16
    initializer_range               TruncatedNormal的初始值，默认为0.02
    use_relative_positions          是否采用相对位置，可选项为true或false，默认为False
    dtype                           输入的数据类型，可选项为mstype.float16或mstype.float32，默认为mstype.float32
    compute_type                    Bert Transformer的计算类型，可选项为mstype.float16或mstype.float32，默认为mstype.float16

Parameters for optimizer:
    AdamWeightDecay:
    decay_steps                     学习率开始衰减的步数
    learning_rate                   学习率
    end_learning_rate               结束学习率，取值需为正数
    power                           幂
    warmup_steps                    热身学习率步数
    weight_decay                    权重衰减
    eps                             增加分母，提高小数稳定性

    Lamb:
    decay_steps                     学习率开始衰减的步数
    learning_rate                   学习率
    end_learning_rate               结束学习率
    power                           幂
    warmup_steps                    热身学习率步数
    weight_decay                    权重衰减

    Momentum:
    learning_rate                   学习率
    momentum                        平均移动动量
```



## 模型描述

## 性能

### 预训练性能

| 参数                  |GPU                       |
| -------------------------- |  ------------------------- |
| 模型版本              | BERT_base                  |
| 资源                   | NV SMX2 V100-32G          |
| 上传日期              |  2021-07-05      |
| MindSpore版本          |  1.3.0                     |
| 数据集                    | cn-wiki-128               |
| 训练参数        | pretrain_config.yaml          |
| 优化器                  |  Lamb                  |
| 损失函数             |  SoftmaxCrossEntropy       |
| 输出              |                   |
| 轮次                      |                            |                      |
| Batch_size |  32*8 |
| 损失                       | 1.913                 |
| 速度                      |180毫秒/步             |
| 总时长                 | |
| 参数（M）                 |                         |
| 微调检查点 |                   |

# 随机情况说明

run_standalone_pretrain.sh和run_distributed_pretrain.sh脚本中将do_shuffle设置为True，默认对数据集进行轮换操作。

run_classifier.sh、run_ner.sh和run_squad.sh中设置train_data_shuffle和eval_data_shuffle为True，默认对数据集进行轮换操作。

config.py中，默认将hidden_dropout_prob和note_pros_dropout_prob设置为0.1，丢弃部分网络节点。

run_pretrain.py中设置了随机种子，确保分布式训练中每个节点的初始权重相同。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。

# FAQ

优先参考[ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ)来查找一些常见的公共问题。

- **Q: 运行过程中发生持续溢出怎么办？**

  **A**： 持续溢出通常是因为使用了较高的学习率导致训练不收敛。可以考虑修改yaml配置文件中的参数，调低`learning_rate`来降低初始学习率或提高`power`加速学习率衰减。

- **Q: 运行报错shape不匹配是什么问题？**

  **A**： Bert模型中的shape不匹配通常是因为模型参数配置和使用的数据集规格不匹配，主要是句长问题，可以考虑修改`seq_length`参数来匹配所使用的具体数据集。改变该参数不影响权重的规格，权重的规格仅与`max_position_embeddings`参数有关。

- **Q: 运行过程中报错Gather算子错误是什么问题？**

  **A**： Bert模型中的使用Gather算子完成embedding操作，操作会根据输入数据的值来映射字典表，字典表的大小由配置文件中的`vocab_size`来决定，当实际使用的数据集编码时使用的字典表大小超过配置的大小时，操作gather算子时就会发出越界访问的错误，从而Gather算子会报错中止程序。

- **Q: 修改了yaml文件中的配置，为什么没有效果？**

  **A**：实际运行的参数，由`yaml`文件和`命令行参数`共同控制，使用`ascend_dsitributed_launcher`的情况下，也会受`ini`配置文件的影响。起作用的优先级是**bash参数 > ini文件参数 > yaml文件参数**。
