# Decoding di BLOB SQLite
Fine-tuning LoRA su LLM compatti per la decodifica automatica di campi BLOB contenuti in database SQLite di applicazioni mobili.

## Fine-tuning di Qwen2.5 7B

### Configurazione di Training 

| Parametro | Valore |
|---|---|
| Max sequence length | 7000 |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Epochs | 2 |
| Batch size | 8 (2 × 4 accumulation steps) |
| Optimizer | AdamW |
| Precision | bf16 |

![Training & Eval Loss](img/training_curves.png)


### Accuratezza — Base Model [bozza con alcune metriche imprecise]

| Tipo di task | Test (in-distribution) | Test OOD (out-of-distribution) |
|---|---|---|
| Decoder synthesis | 0% | 0% |
| BLOB to JSON | 0% | 0% |
| Format classification | 4.35% | 3.94% |
| Validation judgment | 67.05% | 61.11% |
| Refusal | 0% | 0% |
| **Totale** | **8.03%** | **6.40%** |


### Accuratezza — Fine-Tuned Model [bozza con alcune metriche imprecise]

| Tipo di task | Test (in-distribution) | Test OOD (out-of-distribution) |
|---|---|---|
| Decoder synthesis | 73.98% | 71.26% |
| BLOB to JSON | 45.14% | 60.67% |
| Format classification | 100% | 100% |
| Validation judgment | 71.59% | 66.67% |
| Refusal | 100% | 100% |
| **Totale** | **69.90%** | **75.28%** |