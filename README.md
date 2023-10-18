# tokenizer-vs-tokenizer

**Goal**: To understand different tokenizer classes by Huggingface

*Situation* There are `tokenizers.Tokenizer`, `transformers.PreTrainedTokenizer`, and `transformers.PreTrainedTokenizerFast`. They are quite different to each other, perhaps more than you think.


- tl;dr:
  - Workflow 1/1: *Training a tokenizer*
    - When training a tokenizer from scratch, we (only) can use `tokenizers.Tokenizer`. We save the tokenizer as a `.json` file, and we should log the special tokens.
      - But, convert it to `PreTrainedTokenizerFast()` with specifying all the special tokens, and save it by `PreTrainedTokenizerFast.save_pretrained()`.
    - When training a tokenizer with a config copied from an existing tokenizer, we can load it as `PreTrainedTokenizerFast` and use `train_new_from_iterator()`; to get a new vocab.
  - Workflow 2/2: *Using it* (training models)
    - ALWAYS load, use, and save as `PreTrainedTokenizerFast`.
    - When encoding, use `tokenizer.encode()` if you only need the ids. Use `tokenizer()` if you need other things like offsets.


| Feature                                             | `tokenizers.Tokenizer`                                                             | `transformers.PreTrainedTokenizer`                                                             | `transformers.PreTrainedTokenizerFast`                                                                                                       |
|-----------------------------------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| **Implementation**                                  | Rust-based                                                                         | Python-based                                                                                   | Rust-based                                                                                                                                   |
| **Speed**                                           | Fast, especially for batched tokenization                                          | Standard                                                                                       | Fast, especially for batched tokenization                                                                                                    |
| **Tokenizing**                                      | Yes                                                                                | Yes                                                                                            | Yes                                                                                                                                          |
| **Converting tokens to ids and back**               | Yes                                                                                | Yes                                                                                            | Yes                                                                                                                                          |
| **Encoding/Decoding**                               | Yes                                                                                | Yes                                                                                            | Yes                                                                                                                                          |
| **Adding new tokens to the vocabulary**             | Yes                                                                                | Yes                                                                                            | Yes                                                                                                                                          |
| **Managing special tokens**                         | Yes                                                                                | Yes                                                                                            | Yes                                                                                                                                          |
| **Mapping between original string and token space** | Yes                                                                                | No                                                                                             | Yes                                                                                                                                          |
| **Training a new tokenizer**                        | Yes                                                                                | Yes [*1]                                                                                       | No [*1]                                                                                                                                      |
| **Has attrs like `.bos_token`**                     | No                                                                                 | Yes if set explicitly [*2]                                                                     | Yes if set explicitly [*2]                                                                                                                   |
| **Saved form**                                      | one `.json` file                                                                   | one `tokenizer.json` file, optionally w/ `tokenizer_config.json` and `special_tokens_map.json` | Same as <--                                                                                                                                  |
| **How to call**                                     | `tokenizer.encode()`                                                               | see -> | `tokenizer.encode()` (returns `list[int]`. if `return_tensors='pt'`, returns 2D tensor) // (`tokenizer()` works and returns other things too. |
| **Returned value when encoding**                    | `Encoding` instance.  `encoded.ids == list[int]`. `encoded['ids']` does not work.  | see -> | list of int. // `BatchEncoding` instance. Subclass of Dict. `encoded.input_ids == list[int]`. `encoded["input_ids"]` works too               |
| **Attrs of the encoded instance**                   | `ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing` | see -> | `input_ids, token_type_ids, attention_mask`                                                                                                  |
| **`len(encoded)`**                                    | length of the tokens                                                               | see -> | `tokenizer()`: 3 (length of dict == number of keys).                                                                                                          |


[*1]: The transformers.PreTrainedTokenizer class has a method called train_new_from_iterator() which can be used to train a new tokenizer with the same characteristics as an existing one. However, this method is not available in the transformers.PreTrainedTokenizerFast class.

[*2]: We can convert a `tokenizer.Tokenizer` to `PreTrainedTokenizer` (and `..Fast`) as below.

```python
tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object,
        model_max_length=2048,
        padding_side="right",
        pad_token=DEFAULT_PAD_TOKEN,
        eos_token=DEFAULT_EOS_TOKEN,
        unk_token=DEFAULT_UNK_TOKEN,
        bos_token=DEFAULT_BOS_TOKEN,
    )

tokenizer.bos_token
>>> '<s>'  # if DEFAULT_BOS_TOKEN == '<s>'.
```

Same `kwargs` should be passed to have the special token attributes when loading a tokenizer from a file.

```python
tokenizer = transformers.AutoTokenizer.from_pretrained(  # transformers.PreTrainedTokenizerFast.from_pretrained
    path,
    model_max_length=2048,
    padding_side="right",
    pad_token=DEFAULT_PAD_TOKEN,
    eos_token=DEFAULT_EOS_TOKEN,
    unk_token=DEFAULT_UNK_TOKEN,
    bos_token=DEFAULT_BOS_TOKEN,
)
```
