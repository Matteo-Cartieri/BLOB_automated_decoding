from dataclasses import dataclass
from typing import Any
import torch

# @dataclass
# class DataCollatorForCausalLMWithLabels:
#     tokenizer: Any

#     def __call__(self, features):
#         batch = self.tokenizer.pad(
#             {
#                 "input_ids": [f["input_ids"] for f in features],
#                 "attention_mask": [f["attention_mask"] for f in features],
#             },
#             padding=True,
#             padding_side="right",
#             return_tensors="pt",
#         )

#         max_len = batch["input_ids"].shape[1]
#         labels = []
#         for f in features:
#             lab = f["labels"]
#             if len(lab) < max_len:
#                 lab = lab + [-100] * (max_len - len(lab))
#             else:
#                 lab = lab[:max_len]
#             labels.append(lab)

#         batch["labels"] = torch.tensor(labels, dtype=torch.long)
#         return batch


@dataclass
class DataCollatorForCausalLMWithLabels:
    tokenizer: Any

    def __call__(self, features):       
        batch = self.tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]

        raw_labels = [f["labels"] for f in features]
        padded_labels = []
        
        for lab in raw_labels:
            pad_len = max_len - len(lab)
            if pad_len > 0:
                if self.tokenizer.padding_side == "right":
                    lab = lab + [-100] * pad_len
                else:
                    lab = [-100] * pad_len + lab
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch