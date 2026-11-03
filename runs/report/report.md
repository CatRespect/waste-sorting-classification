# Waste Sorting — experiment report

This report is generated from `runs/<run>/val_metrics.json`, `test_metrics.json`, `train_log.json`.
Validation metrics are used for model selection; test metrics are for the final held-out comparison.

## Test-set comparison

| config                                  | backbone        | strategy             |   seed |   val_macro_f1 |   test_accuracy |   test_macro_f1 |   latency_ms_per_image_bs1 |
|:----------------------------------------|:----------------|:---------------------|-------:|---------------:|----------------:|----------------:|---------------------------:|
| clip_vitb32_full_finetune               | clip_vit_b32    | full_finetune        |  87578 |         0.9568 |          0.9499 |          0.9500 |                     6.2558 |
| convnext_tiny_full_finetune             | convnext_tiny   | full_finetune        |  87578 |         0.9654 |          0.9499 |          0.9428 |                     9.4004 |
| convnext_tiny_layer_decay               | convnext_tiny   | layer_decay_finetune |  87578 |         0.9669 |          0.9314 |          0.9330 |                     9.9113 |
| convnext_tiny_gradual_unfreeze          | convnext_tiny   | gradual_unfreeze     |  87578 |         0.9568 |          0.9420 |          0.9316 |                     9.8136 |
| effb0_finetune_seed6545656              |                 |                      |        |       nan      |          0.9182 |          0.9146 |                    10.3884 |
| effb0_finetune_seed213123               |                 |                      |        |       nan      |          0.9156 |          0.9086 |                    10.4062 |
| resnet50_finetune_seed87578             |                 |                      |        |       nan      |          0.9129 |          0.9080 |                     7.3069 |
| resnet50_finetune_seed213123            |                 |                      |        |       nan      |          0.9288 |          0.9074 |                     7.7398 |
| effb0_full_finetune                     | efficientnet_b0 | full_finetune        |  87578 |         0.9507 |          0.9129 |          0.9060 |                    10.2362 |
| resnet50_full_finetune                  | resnet50        | full_finetune        |  87578 |         0.9239 |          0.9208 |          0.9060 |                     7.2974 |
| resnet50_finetune_seed6545656           |                 |                      |        |       nan      |          0.9050 |          0.8805 |                     7.6672 |
| clip_vitb32_feature_extraction          | clip_vit_b32    | feature_extraction   |  87578 |         0.8949 |          0.8813 |          0.8717 |                     6.2271 |
| effb0_feature_extraction_seed6545656    |                 |                      |        |       nan      |          0.8707 |          0.8471 |                     9.7614 |
| effb0_feature_extraction_seed213123     |                 |                      |        |       nan      |          0.8707 |          0.8467 |                     9.8074 |
| effb0_feature_extraction                | efficientnet_b0 | feature_extraction   |  87578 |         0.8396 |          0.8549 |          0.8310 |                    10.2599 |
| resnet50_feature_extraction             | resnet50        | feature_extraction   |  87578 |         0.8504 |          0.8628 |          0.8300 |                     7.4309 |
| resnet50_feature_extraction_seed6545656 |                 |                      |        |       nan      |          0.8575 |          0.8259 |                     8.1202 |
| resnet50_feature_extraction_seed213123  |                 |                      |        |       nan      |          0.8575 |          0.8254 |                     7.7685 |

## Highest test macro-F1 (descriptive)

- Run: `clip_vitb32_full_finetune` (seed `87578`)
- Backbone / strategy: `clip_vit_b32` / `full_finetune`
- Test accuracy: 0.9499
- Test macro-F1: 0.9500

## Files

- `comparison.csv` — complete tabular metrics, including per-class precision/recall/F1 and TTA runs.
- Each run folder — confusion matrices, misclassified examples, model checkpoints, training history and metrics JSON.
