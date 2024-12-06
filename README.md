# optimizer_evaluation

Optimization algorithms are a cornerstone of machine learning, dictating how effectively models converge to solutions. Despite the success of many optimization algotihms, these algorithms face significant limitations, such as sensitivity to hyperparameters, inefficiency on non-convex landscapes, and lack of adaptability across tasks. In this work, we focused on evaluating and experimenting with existing optimization algorithms across diverse tasks in various domains. The goal is to understand the conditions under which specific algorithms perform optimally and to explore enhancements such as learning rate scheduling and warm-up strategies. By systematically applying and analyzing these algorithms, we aim to provide insights into how to effectively adapt optimization techniques to improve model performance across different scenarios. We evaluated 5 optimizers which are Adam, AdamW, AdaGrad, SGD, and RMSProp on semantic segmentation, image classification, language modeling, and text classification tasks. 

For semantic segmentation task, our baseline code is from DAFormer paper code. So, you can check detailed information on their github.

# Semantic Segmentation Experiments
``` cmd
1. Evaluate the performance of each optimizer.
2. Evaluate the performance of each optimizer with the Poly10 scheduler.
3. Evaluate the performance of each optimizer with the PolyLR scheduler.
4. Evaluate the performance of each optimizer with a warmup strategy.
5. Evaluate the performance in cross-domain adaptation (GTA5 → Cityscapes) to check generalization performance.
```

# Results






