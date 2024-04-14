import struct
import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# -----------------------------------------------------------------------------
# common utilities

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def model_export(config, model, filepath):
    """
    Export the model weights in full float32 .bin file to be read from C.
    """
    version = 1

    out_file = open(filepath, 'wb')
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "Jamb" in ASCII
    out_file.write(struct.pack('I', 0x4a616d62))
    # 2) write version, which will be int
    out_file.write(struct.pack('i', version))
    # 3) write the params, which will be 9 ints
    shared_classifier = torch.equal(model.model.embed_tokens.weight, model.lm_head.weight)
    header = struct.pack(
        'iiiiiiiiiiiiiiiii',
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.vocab_size if shared_classifier else -config.vocab_size,
        config.n_ctx,
        config.num_experts,
        config.num_experts_per_tok,
        config.attn_layer_offset,
        config.attn_layer_period,
        config.expert_layer_offset,
        config.expert_layer_period,
        config.mamba_d_conv,
        config.mamba_d_state,
        config.mamba_dt_rank,
        config.mamba_expand * config.hidden_size,
    )
    out_file.write(header)

    def permute_reverse(w, n_heads=config.num_attention_heads, dim1=config.hidden_size, dim2=config.hidden_size):
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    attn_layers = model.model.layers[config.attn_layer_offset::config.attn_layer_period]
    mamba_layers = [layer for layer in model.model.layers if layer not in attn_layers]
    moe_layers = model.model.layers[config.expert_layer_offset::config.expert_layer_period]
    mlp_layers = [layer for layer in model.model.layers if layer not in moe_layers]

    # now let's write out all the params
    weights = [
        model.model.embed_tokens.weight,
        *[layer.input_layernorm.weight for layer in model.model.layers],
        *[permute_reverse(layer.self_attn.q_proj.weight) for layer in attn_layers],
        *[permute_reverse(layer.self_attn.k_proj.weight) for layer in attn_layers],
        *[layer.self_attn.v_proj.weight for layer in attn_layers],
        *[layer.self_attn.o_proj.weight for layer in attn_layers],
        *[layer.mamba.in_proj.weight for layer in mamba_layers],
        *[layer.mamba.conv1d.weight for layer in mamba_layers],
        *[layer.mamba.conv1d.bias for layer in mamba_layers],
        *[layer.mamba.x_proj.weight for layer in mamba_layers],
        *[layer.mamba.dt_proj.weight for layer in mamba_layers],
        *[layer.mamba.dt_proj.bias for layer in mamba_layers],
        *[-torch.exp(layer.mamba.A_log.data) for layer in mamba_layers],
        *[layer.mamba.D.data for layer in mamba_layers],
        *[layer.mamba.out_proj.weight for layer in mamba_layers],
        *[layer.mamba.dt_layernorm.weight for layer in mamba_layers],
        *[layer.mamba.B_layernorm.weight for layer in mamba_layers],
        *[layer.mamba.C_layernorm.weight for layer in mamba_layers],
        *[layer.pre_moe_layernorm.weight for layer in model.model.layers],
        *[layer.moe.router.weight for layer in moe_layers],
        *[mlp.gate_proj.weight for layer in moe_layers for mlp in layer.moe.experts],
        *[mlp.down_proj.weight for layer in moe_layers for mlp in layer.moe.experts],
        *[mlp.up_proj.weight for layer in moe_layers for mlp in layer.moe.experts],
        *[layer.moe.experts[0].gate_proj.weight for layer in mlp_layers],
        *[layer.moe.experts[0].down_proj.weight for layer in mlp_layers],
        *[layer.moe.experts[0].up_proj.weight for layer in mlp_layers],
        model.model.final_layernorm.weight,
    ]
    if not shared_classifier:
        weights.append(model.lm_head.weight)
    for w in weights:
        serialize_fp32(out_file, w)

    # write to binary file
    out_file.close()
    print(f"write {filepath}")

# -----------------------------------------------------------------------------
# Load / import functions

def load_model(model_path):
    # load HF model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    return config, model

# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="huggingface model path")
    args = parser.parse_args()

    config, model = load_model(args.model)

    if model is None:
        parser.error("Can't load input model!")

    # export
    model_export(config, model, args.filepath)
