# jamba.c

Inference of Jamba models in pure C

Inspired by and using code from [llama2.c](https://github.com/karpathy/llama2.c) and [mamba.c](https://github.com/kroggen/mamba.c)

## Start

```
python3 tokenizer.py -t <model_path>
python3 export.py jamba.bin --model <model_path>
make runfast
./run jamba.bin
```
