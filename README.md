# 3SAT: A Simple Self-Supervised Adversarial Training Framework
## Abrtract
The combination of self-supervised learning and adversarial training (AT) can significantly improve the adversarial robustness of self-supervised models. However, the robustness of self-supervised adversarial training (self-AT) still lags behind that of state-of-the-art (SOTA) supervised AT (sup-AT), even though the performance of current self-supervised learning models has already matched or even surpassed that of SOTA supervised learning models. This issue raises concerns about the secure application of self-supervised learning models.
 As a result of the incorporation of adversarial training, self-AT becomes a challenging joint optimisation problem. Furthermore, recent studies have shown that the data augmentation methods necessary for constructing positive pairs in self-supervised learning negatively impact the robustness improvement in self-AT. Inspired by this, we propose 3SAT, a simple self-supervised adversarial training framework. 3SAT conducts adversarial training on original, unaugmented samples, reducing the difficulty of optimizing the adversarial training subproblem and fundamentally eliminating the negative impact of data augmentation on robustness improvement. Additionally, 3SAT introduces a dynamic training objective scheduling strategy to address the issue of model training collapse during the joint optimization process when using original samples directly. 3SAT is not only structurally simple and computationally efficient, reducing  self-AT training time by half, but it also improves the SOTA self-AT robustness accuracy by 16.19\% and standard accuracy by 11.41% under Auto-Attack on the CIFAR-10 dataset. Even more impressively, 3SAT surpasses the SOTA sup-AT method in robust accuracy by a significant margin of 11.25%. This marks the first time that self-AT has outperformed SOTA sup-AT in robustness, indicating that self-AT is a superior method for improving model robustness. 

## Environment
A standard pytorch environment (>=1.0.0) with basic packages (e.g., numpy, pickle) is enough. 
The 3SAT code is based on solo-learn, and to successfully run 3SAT, we first need to install solo-learn.
First clone the solo-learn [repo](https://github.com/vturrisi/solo-learn) 
Then, to install solo-learn use:
```bash
pip3 install .[dali,umap,h5] --extra-index-url https://developer.download.nvidia.com/compute/redist
```
To evaluate under the Auto-Attack benchmark, the autoattack package is required. Run the following code to install:
```bash
pip install git+https://github.com/fra31/auto-attack
```
## Data
CIFAR10, CIFAR100, and STL10 dataset are required. You may manually download them and put in the ./datasets folder, or directly run our provided scripts to automatically download these datasets.

## Pretrain
All training parameters for 3SAT are located in the .yaml configuration file under the ./scripts/pretrain directory.

Execute the following command to perform pre-training of 3SAT.
```bash
python3    main_pretrain.py \
                    --config-path scripts/pretrain/DATASET_NAME/ \
                    --config-name 3sat.yaml
```
For instance, to perform pre-training of 3SAT on CIFAR-10, the following command can be executed.
```bash
python3    main_pretrain.py \
                    --config-path scripts/pretrain/cifar/ \
                    --config-name 3sat.yaml
```

## Linear Finetuning
All linear fine-tuning parameters for 3SAT are located in the .yaml configuration file under the ./scripts/linear directory. Before fine-tuning, it is necessary to specify the pre-trained model path in the .yaml file.

```ymal
pretrained_feature_extractor: PATH_TO_PRETRAINED_MODEL
```

Execute the following command to perform  linear fine-tuning of 3SAT.
```bash
python3 adv_linear_eval.py --config-path scripts/linear/cifar --config-name 3sat.yaml
```
If you need to switch to linear fine-tuning mode, please modify the following two parameters in the configuration file.
```ymal
finetune: False
adversarial: False
```
'finetune' indicates whether to perform full fine-tuning, that is, whether the encoder also participates in the fine-tuning, while 'adversarial' specifies whether to use adversarial examples for fine-tuning.




## Evaluation
All evaluation parameters for 3SAT are located in the .yaml configuration file under the ./scripts/aatest directory.

Before proceeding with the evaluation, it is necessary to specify the path of the fine-tuned model in the configuration file.

```ymal
pretrained_feature_extractor: PATH_TO_PRETRAINED_MODEL
```
Execute the following command to perform evaluation of 3SAT.
```bash
python3 aa_test.py --config-path scripts/aatest/cifar --config-name 3sat.yaml
```
We need to use the  fine-tuned models under different fine-tuning modes for robustness evaluation under Auto-Attack.
