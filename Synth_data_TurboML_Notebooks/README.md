
# TurboML Real-Time ML Q&A Dataset

[![License: CC-BY-SA-4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/DebopamC/TurboML_Synthetic_QnA_Dataset)

## Dataset Description

A high-quality Q&A dataset about TurboML - a real-time machine learning platform. Contains **1,343 technical questions** with detailed answers covering implementation, troubleshooting, architecture design, and performance optimization.

### Techniques to Make this Dataset
![image](https://github.com/user-attachments/assets/278f0421-59d8-4823-a873-39cc5307464b)


## Dataset Structure

### Data Fields
| Field | Type | Description |
|-------|------|-------------|
| `question` | string | Technical question |
| `answer` | string | Detailed answer with code snippets |
| `context` | string | Full merged context used for question generation |
| `base_chunk` | string | Primary documentation section for this Q&A pair |
| `context_sections` | list[str] | All documentation sections in context |
| `generation_timestamp_ns` | int64 | Nanosecond timestamp of answer generation |

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total Q&A pairs | 1,343 |
| Average context length | 12,066 tokens (48,265 chars) |
| Total Engineered Contexts Generated | 89 |
| Code-containing answers | 832 (62%) |
| Troubleshooting questions | 403 (30%) |
| Architecture questions | 269 (20%) |
| Time of Generation| 28 Feb, 2025 |

## Data Sources

### Original Content
- **Source Documentation**: Proprietary TurboML knowledge base (89 sections)
- **Summary Document**: `summary.md` (1,250 words technical overview)
- **Context Strategy**: 
  - BGE-M3 embeddings for semantic grouping
  - FAISS vector store with similarity thresholding
  - Maximum 3 chunks + summary per context


## Usage

### Direct Loading
```python
from datasets import load_dataset

dataset = load_dataset("DebopamC/TurboML_Synthetic_QnA_Dataset", split = "train")
```

### Sample Access
```python
# Get first example
example = dataset["train"][0]

print(f"Question: {example['question']}")
print(f"Answer: {example['answer'][:200]}...")
print(f"Context Sections: {example['context_sections']}")
```

## License
[CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/) - Requires attribution and sharing under same license

## Citation
If you use this dataset in research, please cite:
```bibtex
@dataset{turboml_qa_2024,
  author = {Debopam Chowdhury},
  title = {TurboML Real-Time ML Q&A Dataset},
  year = {2025},
  publisher = {Self},
  version = {1.0},
  url = {https://huggingface.co/datasets/DebopamC/TurboML_Synthetic_QnA_Dataset}
}
```

## Acknowledgements
- **Documentation Source**: TurboML Documentation, Pypi Library Package
- **Synthetic Generation**: 
	 - Google Gemini 2.0 Flash (Question generation from context)
	 - Google Gemini 2.0-Experimental (Answer Generation from full-knowledge-base) 
- **Embeddings**: BAAI/bge-m3
- **Vector Store**: FAISS

## Known Limitations
1. Answers limited to documentation state as of 2024-02-28
2. May not cover edge cases beyond provided context
3. Code examples assume TurboML v1.2+ 

---

**Maintainer**: [Debopam Chowdhury] ([debopamstudy@gmail.com](mailto:debopamstudy@gmail.com))  
**Update Frequency**: Not Mentioned
