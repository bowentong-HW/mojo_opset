import json
import os
import sys
import types
from typing import Dict, List, Tuple

import torch
import torch_npu
import custom_ops  # noqa: F401
import torch.distributed as dist


os.environ.setdefault("MOJO_BACKEND", "torch_npu")
os.environ.setdefault("MOJO_DISABLE_ASSERTION_REWRITE", "1")

MODEL_PATH = "/data00/dpskv4-flash-quant"
DEVICE = "npu:0"
LAYER_IDX = int(os.environ.get("COMPARE_LAYER_IDX", "0"))
SINGLE_LAYER_COMPARE = os.environ.get("COMPARE_SINGLE_LAYER", "0") == "1"
MODEL_LAYER_IDX = 0 if SINGLE_LAYER_COMPARE else LAYER_IDX
NUM_LAYERS = 1 if SINGLE_LAYER_COMPARE else LAYER_IDX + 1
LOGITS_ONLY_COMPARE = os.environ.get("COMPARE_LOGITS_ONLY", "0") == "1"
COMPARE_EP_SIZE = int(os.environ.get("COMPARE_EP_SIZE", "1"))
DECODE_STEP_COMPARE = int(os.environ.get("COMPARE_DECODE_STEP", "0"))

GOLDEN_ROOT = "/data01/tbw/mojo_opset_info/cann-recipes-infer"
GOLDEN_MODEL_ROOT = f"{GOLDEN_ROOT}/models/deepseek-v4"
MOJO_ROOT = "/data01/tbw/mojo_opset_info/mojo_opset"
sys.path.insert(0, MOJO_ROOT)
sys.path.insert(0, GOLDEN_ROOT)
sys.path.insert(0, GOLDEN_MODEL_ROOT)
sys.path.insert(0, f"{GOLDEN_MODEL_ROOT}/models")


def _load_safetensor_weights(layer_idx: int) -> List[Tuple[str, torch.Tensor]]:
    from safetensors.torch import load_file

    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    tensors: Dict[str, torch.Tensor] = {}
    loaded_files = set()
    for _, shard_name in weight_map.items():
        if shard_name in loaded_files:
            continue
        tensors.update(load_file(os.path.join(MODEL_PATH, shard_name)))
        loaded_files.add(shard_name)

    target_suffixes = set()
    if SINGLE_LAYER_COMPARE:
        target_prefix = f"layers.{layer_idx}."
        for name in tensors:
            if name.startswith(target_prefix):
                target_suffixes.add(name[len(target_prefix):])

    keep = []
    for name, tensor in tensors.items():
        if not name.startswith("layers."):
            keep.append((name, tensor))
            continue
        parts = name.split(".")
        ckpt_layer_idx = int(parts[1])
        if SINGLE_LAYER_COMPARE:
            if ckpt_layer_idx == layer_idx:
                parts[1] = "0"
                keep.append((".".join(parts), tensor))
            elif ckpt_layer_idx == 0 and ".".join(parts[2:]) not in target_suffixes:
                keep.append((name, tensor))
        elif ckpt_layer_idx <= layer_idx:
            keep.append((name, tensor))
    return keep


def _build_runner_settings() -> Dict:
    return {
        "model_name": "deepseek_v4",
        "model_path": MODEL_PATH,
        "world_size": COMPARE_EP_SIZE,
        "model_config": {
            "with_ckpt": True,
            "enable_weight_nz": False,
            "enable_cache_compile": False,
            "enable_multi_streams": False,
            "prefill_mini_batch_size": 0,
            "platform_version": "A3",
            "enable_static_kernel": False,
            "enable_limit_core": False,
            "perfect_eplb": False,
            "enable_online_split_weight": False,
            "pa_block_size": 128,
        },
        "data_config": {
            "batch_size": 1,
            "batch_size_per_rank": 1,
            "max_new_tokens": 1,
            "input_max_len": 1,
            "max_position_embeddings": 4096,
        },
        "parallel_config": {
            "cp_size": 1,
            "oproj_tp_size": 1,
            "attn_tp_size": 1,
            "moe_tp_size": 1,
            "moe_ep_size": COMPARE_EP_SIZE,
            "embed_tp_size": 1,
            "lmhead_tp_size": 1,
            "moe_dp_size": 1,
            "embed_dp_size": 1,
            "attn_dp_size": 1,
        },
        "exe_mode": "eager",
    }


def _process_golden_weights(model):
    from module.quantization import QuantizeMethodBase

    float_scales_map = ["gate_up_proj", "q_b_proj", "wq_b"]
    float_smooth_scales_map = ["down_proj"]

    for module_name, module in model.named_modules():
        if "wo_a" in module_name and hasattr(module, "weight"):
            config = model.config
            head_dim_per_group = config.num_attention_heads * config.head_dim // config.o_groups
            module.weight.data = (
                module.weight.data.view(-1, config.o_lora_rank, head_dim_per_group)
                .transpose(1, 2)
                .contiguous()
            )
            continue

        quant_method = getattr(module, "quant_method", None)
        if not isinstance(quant_method, QuantizeMethodBase):
            continue

        scales_dtype = {}
        if any(scale_name in module_name for scale_name in float_scales_map):
            scales_dtype["scale_dtype"] = torch.float
        if any(smooth_name in module_name for smooth_name in float_smooth_scales_map):
            scales_dtype["smooth_scale_dtype"] = torch.float

        is_transpose = "compressor" not in module_name
        quant_method.process_weights_after_loading(
            module, is_nz=False, is_transpose=is_transpose, scales_dtype=scales_dtype
        )
        if (
            COMPARE_EP_SIZE > 1
            and dist.is_initialized()
            and hasattr(module, "smooth_scale_1")
            and module.smooth_scale_1 is not None
            and module.smooth_scale_1.shape[0] * COMPARE_EP_SIZE == model.config.n_routed_experts
        ):
            all_smooth_scale_1 = module.smooth_scale_1.data.new_empty(
                module.smooth_scale_1.data.shape[0] * COMPARE_EP_SIZE,
                module.smooth_scale_1.data.shape[1],
            )
            dist.all_gather_into_tensor(
                all_smooth_scale_1,
                module.smooth_scale_1.data,
                group=model.hccl_comm_dict.get("moe_ep_group", None),
            )
            module.smooth_scale_1.data = all_smooth_scale_1


def _compare_tensor(name: str, mojo: torch.Tensor, golden: torch.Tensor):
    mojo = mojo.detach().float().cpu()
    golden = golden.detach().float().cpu()
    if mojo.shape != golden.shape:
        print(f"{name}: SHAPE_MISMATCH mojo={list(mojo.shape)} golden={list(golden.shape)}")
        return False

    diff = (mojo - golden).abs()
    denom = golden.abs().clamp_min(1e-6)
    rel = diff / denom
    cos = torch.nn.functional.cosine_similarity(mojo.flatten(), golden.flatten(), dim=0).item()
    print(
        f"{name}: cos={cos:.8f}, max_abs={diff.max().item():.6f}, "
        f"mean_abs={diff.mean().item():.6f}, max_rel={rel.max().item():.6f}, "
        f"mean_rel={rel.mean().item():.6f}"
    )
    return cos > 0.99


def _compare_any(name: str, mojo: torch.Tensor, golden: torch.Tensor):
    if mojo.dtype in (torch.int8, torch.int16, torch.int32, torch.int64) or golden.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        mojo_cpu = mojo.detach().cpu()
        golden_cpu = golden.detach().cpu()
        if mojo_cpu.shape != golden_cpu.shape:
            print(f"{name}: SHAPE_MISMATCH mojo={list(mojo_cpu.shape)} golden={list(golden_cpu.shape)}")
            return False
        match = (mojo_cpu == golden_cpu).float().mean().item()
        max_abs = (mojo_cpu.to(torch.float32) - golden_cpu.to(torch.float32)).abs().max().item()
        print(f"{name}: match_rate={match:.8f}, max_abs={max_abs:.6f}")
        return match == 1.0
    return _compare_tensor(name, mojo, golden)


def _save_stage(captured: Dict[str, torch.Tensor], name: str, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    captured[f"{name}.input"] = input_tensor.detach().clone()
    captured[f"{name}.output"] = output_tensor.detach().clone()


def _move_capture_to_cpu(captured: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in captured.items()}


def _build_inputs():
    input_ids_env = os.environ.get("COMPARE_INPUT_IDS")
    if input_ids_env:
        input_ids = [[int(token) for token in row.split(",") if token] for row in input_ids_env.split(";")]
        return torch.tensor(input_ids, dtype=torch.long, device=DEVICE)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)
    try:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "你好"}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception:
        input_ids = tokenizer.encode("你好", return_tensors="pt")
    return input_ids.to(DEVICE)


def _build_golden_model():
    from models.configuration_deepseek import DeepseekV3Config
    from models.modeling_deepseek import DeepseekV3ForCausalLM
    from models.modules.registry import OpKernel, auto_import_modules
    from module.quantization import get_quant_config

    auto_import_modules("models.modules.op_impls")
    OpKernel.op_impl_apply("hc_pre", "hc_pre_ascendc_a3")
    OpKernel.op_impl_apply("hc_post", "hc_post_ascendc_a3")
    OpKernel.op_impl_apply("gate_topk", "gate_topk_native_a3")

    config = DeepseekV3Config.from_pretrained(MODEL_PATH)
    config.num_hidden_layers = NUM_LAYERS
    if SINGLE_LAYER_COMPARE:
        ratio = config.compress_ratios[LAYER_IDX]
        config.compress_ratios = [1 if ratio == 0 else ratio]
    else:
        config.compress_ratios = [1 if r == 0 else r for r in config.compress_ratios[:NUM_LAYERS]]
    config.quant_config = get_quant_config(config, "compressed-tensors", MODEL_PATH)

    model = DeepseekV3ForCausalLM(config, _build_runner_settings()).to(DEVICE).eval()
    try:
        model.load_weights(_load_safetensor_weights(LAYER_IDX))
    except ValueError:
        if not SINGLE_LAYER_COMPARE:
            raise
        print("Ignore golden strict missing-weight check in single-layer remap mode.")
    model.to(DEVICE)
    for module in model.modules():
        if hasattr(module, "hccl_comm_dict") and module.hccl_comm_dict is None:
            module.hccl_comm_dict = {}
    _process_golden_weights(model)
    return model


def _build_mojo_model():
    from mojo_opset.modeling.deepseekv4.mojo_deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM

    with open(os.path.join(MODEL_PATH, "config.json"), "r", encoding="utf-8") as f:
        config = DeepseekV4Config(**json.load(f))
    config.num_hidden_layers = NUM_LAYERS
    if SINGLE_LAYER_COMPARE:
        config.compress_ratios = [config.compress_ratios[LAYER_IDX]]
    else:
        config.compress_ratios = config.compress_ratios[:NUM_LAYERS]

    origin_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    ep_rank = dist.get_rank() % COMPARE_EP_SIZE if dist.is_initialized() and COMPARE_EP_SIZE > 1 else 0
    model = DeepseekV4ForCausalLM(config, num_layers=NUM_LAYERS, ep_size=COMPARE_EP_SIZE, ep_rank=ep_rank)
    torch.set_default_dtype(origin_dtype)
    model = model.to(DEVICE).eval()
    if COMPARE_EP_SIZE > 1 and dist.is_initialized():
        model.init_parallel_comm_group()
        model.set_ep_group()
    if SINGLE_LAYER_COMPARE:
        _load_mojo_remapped_weights(model, DeepseekV4ForCausalLM)
    else:
        DeepseekV4ForCausalLM.load_weights(model, MODEL_PATH)
    return model


def _load_mojo_remapped_weights(model, model_cls):
    model_cls._init_default_weights(model)
    name_mapping = model_cls._build_name_mapping(model)
    params_dict = dict(model.named_parameters())
    buffers_dict = dict(model.named_buffers())
    expert_weights = {}

    for ck_key, weight in _load_safetensor_weights(LAYER_IDX):
        if "experts." in ck_key and ".ffn." in ck_key and "shared_experts" not in ck_key:
            expert_weights[ck_key] = weight
            continue
        model_name = name_mapping.get(ck_key)
        if model_name is None:
            continue
        if model_name in params_dict:
            param = params_dict[model_name]
            weight = model_cls._align_weight(weight, param)
            if param.shape == weight.shape:
                param.data.copy_(weight)
        elif model_name in buffers_dict:
            buf = buffers_dict[model_name]
            weight = model_cls._align_weight(weight, buf)
            if buf.shape == weight.shape:
                buf.data.copy_(weight)

    model_cls._load_expert_weights(model, expert_weights)
    for layer_idx in range(model.config.num_hidden_layers):
        mlp = model.model.layers[layer_idx].mlp
        if hasattr(mlp, "process_expert_weights"):
            mlp.process_expert_weights()


def _patch_golden_layer(model, captured: Dict[str, torch.Tensor]):
    from models.modules.registry import OpKernel

    layer = model.model.layers[MODEL_LAYER_IDX]
    _patch_golden_moe(layer.ffn, captured)

    def staged_forward(
        self,
        hidden_states,
        attn_metadata=None,
        past_residual=None,
        cache_data=None,
        is_prefill=False,
        cur_topk_list=None,
        input_ids=None,
        **kwargs,
    ):
        captured["layer_input"] = hidden_states.detach().clone()

        residual = hidden_states
        hc_input = hidden_states
        hidden_states, post, comb = OpKernel.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.norm_eps,
            self.hc_eps,
        )
        _save_stage(captured, "hc_pre_attn", hc_input, hidden_states)

        norm_input = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        _save_stage(captured, "attn_norm", norm_input, hidden_states)

        attn_input = hidden_states
        hidden_states = self.attn(
            x=hidden_states,
            attn_metadata=attn_metadata,
            cache_data=cache_data,
            is_prefill=is_prefill,
        )
        _save_stage(captured, "attention", attn_input, hidden_states)

        hc_post_attn_input = hidden_states
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)
        _save_stage(captured, "hc_post_attn", hc_post_attn_input, hidden_states)

        residual = hidden_states
        hc_ffn_input = hidden_states
        hidden_states, post, comb = OpKernel.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.norm_eps,
            self.hc_eps,
        )
        _save_stage(captured, "hc_pre_ffn", hc_ffn_input, hidden_states)

        ffn_norm_input = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        _save_stage(captured, "ffn_norm", ffn_norm_input, hidden_states)

        moe_input = hidden_states
        hidden_states = self.ffn(
            hidden_states,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list,
            input_ids=input_ids,
            shared_expert_stream=attn_metadata.get("shared_expert_stream", None) if attn_metadata else None,
        )
        _save_stage(captured, "moe", moe_input, hidden_states)

        hc_post_ffn_input = hidden_states
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)
        _save_stage(captured, "hc_post_ffn", hc_post_ffn_input, hidden_states)

        captured["layer_output"] = hidden_states.detach().clone()
        return hidden_states

    layer.forward = types.MethodType(staged_forward, layer)


def _patch_mojo_layer(model, captured: Dict[str, torch.Tensor]):
    from mojo_opset.modeling.deepseekv4.mojo_deepseek_v4 import OpKernel

    layer = model.model.layers[MODEL_LAYER_IDX]
    _patch_mojo_moe(layer.mlp, captured)

    def staged_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        input_ids=None,
        is_prefill=True,
        **kwargs,
    ):
        captured["layer_input"] = hidden_states.detach().clone()

        residual = hidden_states
        hc_input = hidden_states
        hidden_states, post, comb = OpKernel.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.norm_eps,
            self.hc_eps,
        )
        _save_stage(captured, "hc_pre_attn", hc_input, hidden_states)

        norm_input = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        _save_stage(captured, "attn_norm", norm_input, hidden_states)

        attn_input = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        _save_stage(captured, "attention", attn_input, hidden_states)

        hc_post_attn_input = hidden_states
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)
        _save_stage(captured, "hc_post_attn", hc_post_attn_input, hidden_states)

        residual = hidden_states
        hc_ffn_input = hidden_states
        hidden_states, post, comb = OpKernel.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.norm_eps,
            self.hc_eps,
        )
        _save_stage(captured, "hc_pre_ffn", hc_ffn_input, hidden_states)

        ffn_norm_input = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        _save_stage(captured, "ffn_norm", ffn_norm_input, hidden_states)

        moe_input = hidden_states
        hidden_states = self.mlp(hidden_states, input_ids=input_ids, is_prefill=is_prefill)
        _save_stage(captured, "moe", moe_input, hidden_states)

        hc_post_ffn_input = hidden_states
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)
        _save_stage(captured, "hc_post_ffn", hc_post_ffn_input, hidden_states)

        captured["layer_output"] = hidden_states.detach().clone()
        return hidden_states

    layer.forward = types.MethodType(staged_forward, layer)


def _patch_golden_moe(moe, captured: Dict[str, torch.Tensor]):
    from models.modules.registry import OpKernel
    from models.modeling_deepseek import record_event, wait_event

    def staged_forward(self, hidden_states, is_prefill=False, cur_topk_list=None, input_ids=None, shared_expert_stream=None):
        bsz, seq_len, h = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, h)

        shared_input = hidden_states_flat
        if self.n_shared_experts > 0:
            hidden_states_share = self.forward_shared_expert(hidden_states, shared_expert_stream)
        else:
            hidden_states_share = None
        if hidden_states_share is not None:
            _save_stage(captured, "moe.shared_expert", shared_input, hidden_states_share.view(-1, h))

        gate_input = hidden_states_flat.to(torch.float32)
        logits = self.gate(gate_input)
        _save_stage(captured, "moe.gate_logits", gate_input, logits)

        topk_idx, topk_weight, _ = OpKernel.gate_topk(self, logits, input_ids)
        if self.perfect_eplb:
            topk_idx = cur_topk_list
        topk_idx = topk_idx.to(torch.int32)
        captured["moe.topk_idx.output"] = topk_idx.detach().clone()
        captured["moe.topk_weight.output"] = topk_weight.detach().clone()

        x = hidden_states_flat
        routing_args = {"quant_mode": -1}
        moe_init_routing = torch_npu.npu_moe_init_routing_v2
        if self.gmm_int_quant:
            routing_args.update({"quant_mode": 1})
        enable_smooth_scale = "w8a8" in self.gmm_quant_mode and "float8" not in self.gmm_quant_mode

        hidden_states_list = []
        for chunk_x, chunk_topk_ids, chunk_topk_weight, chunk_share in zip(
            *self._split_tensors(bsz * seq_len, x, topk_idx, topk_weight, hidden_states_share)
        ):
            chunk_len = chunk_x.shape[0]
            expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = moe_init_routing(
                chunk_x,
                expert_idx=chunk_topk_ids,
                active_num=chunk_topk_ids.shape[0] * chunk_topk_ids.shape[1],
                scale=self.experts.smooth_scale_1 if enable_smooth_scale else None,
                expert_num=self.num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, self.num_experts],
                **routing_args,
            )
            captured["moe.dispatch.tokens_per_expert"] = tokens_per_expert.detach().clone()

            tokens_per_expert_group, gathered_tokens, gathered_pertoken_scale, input_splits, output_splits = (
                self.dispatch_double_routing(tokens_per_expert, expanded_x, pertoken_scale)
            )
            _save_stage(captured, "moe.dispatch", chunk_x, gathered_tokens)

            hidden_states_ordered, gathered_pertoken_scale, gathered_ids_unsort, tokens_per_local_expert = (
                torch_npu.npu_moe_re_routing(
                    gathered_tokens,
                    tokens_per_expert_group.view(self.moe_ep_size, -1),
                    per_token_scales=gathered_pertoken_scale,
                )
            )
            gmm_args = {
                "x": hidden_states_ordered,
                "expert_tokens": tokens_per_local_expert,
                "group_list_type": 1,
                "swiglu_limit": self.swiglu_limit,
                "enable_custom_ops": True,
            }
            if "a16" not in self.gmm_quant_mode:
                gmm_args.update({"pertoken_scale": gathered_pertoken_scale})
            expert_out = self.moe_ffn(**gmm_args)
            new_x = torch.index_select(expert_out, 0, gathered_ids_unsort.float().argsort().int())
            _save_stage(captured, "moe.expert", hidden_states_ordered, new_x)

            gathered_tokens = self.combine_double_routing(new_x, expanded_x, input_splits, output_splits)
            routed_only = torch_npu.npu_moe_finalize_routing(
                gathered_tokens,
                skip1=torch.zeros_like(chunk_x),
                skip2=None,
                bias=None,
                scales=chunk_topk_weight.to(gathered_tokens.dtype),
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=None,
                drop_pad_mode=2,
            ).view(chunk_len, self.hidden_dim)
            _save_stage(captured, "moe.combine_routed", new_x, routed_only)

            wait_event(self.enable_multi_streams, self.npu_events, 1)

            final_out = torch_npu.npu_moe_finalize_routing(
                gathered_tokens,
                skip1=chunk_share,
                skip2=None,
                bias=None,
                scales=chunk_topk_weight.to(gathered_tokens.dtype),
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=None,
                drop_pad_mode=2,
            )
            final_out = final_out.view(chunk_len, self.hidden_dim)
            _save_stage(captured, "moe.shared_plus_routed", routed_only, final_out)
            hidden_states_list.append(final_out)

        hidden_states = torch.cat(hidden_states_list, dim=0) if len(hidden_states_list) > 1 else hidden_states_list[0]
        return hidden_states.view(bsz, -1, h)

    moe.forward = types.MethodType(staged_forward, moe)


def _patch_mojo_moe(moe, captured: Dict[str, torch.Tensor]):
    def staged_forward(self, hidden_states, input_ids=None, is_prefill=True):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, self.hidden_size).to(torch.bfloat16)

        gate_input = hidden_states_flat.float()
        logits = torch.nn.functional.linear(gate_input, self.gate)
        _save_stage(captured, "moe.gate_logits", gate_input, logits)

        topk_idx, topk_weight = self._gate_topk(logits, input_ids)
        captured["moe.topk_idx.output"] = topk_idx.detach().clone()
        captured["moe.topk_weight.output"] = topk_weight.detach().clone()

        shared_input = residuals.view(-1, self.hidden_size)
        shared_out = self.shared_experts(residuals)
        shared_out_flat = shared_out.view(-1, self.hidden_size).to(torch.bfloat16)
        _save_stage(captured, "moe.shared_expert", shared_input, shared_out_flat)

        sorted_hidden, tokens_per_expert, sorted_gates, token_indices = self.dispatch(
            hidden_states_flat, topk_weight, topk_idx
        )
        captured["moe.dispatch.tokens_per_expert"] = tokens_per_expert.detach().clone()
        captured["moe.dispatch.sorted_gates"] = sorted_gates.detach().clone()
        _save_stage(captured, "moe.dispatch", hidden_states_flat, sorted_hidden)

        expert_outputs = self.experts(sorted_hidden, tokens_per_expert)
        _save_stage(captured, "moe.expert", sorted_hidden, expert_outputs)

        output_buffer = torch.zeros_like(hidden_states_flat, memory_format=torch.contiguous_format)
        routed_out = self.combine(output_buffer, expert_outputs, sorted_gates, token_indices)
        _save_stage(captured, "moe.combine_routed", expert_outputs, routed_out)

        final_out = routed_out.view(*orig_shape) + shared_out
        _save_stage(captured, "moe.shared_plus_routed", routed_out.view(-1, self.hidden_size), final_out.view(-1, self.hidden_size))
        return final_out

    moe.forward = types.MethodType(staged_forward, moe)


def _run_golden(model, input_ids):
    from models.modules.attention_data import CacheData

    captured = {}
    if not LOGITS_ONLY_COMPARE:
        _patch_golden_layer(model, captured)

    cache = CacheData(
        model.config,
        model.runner_settings,
        model.is_mtp,
        model.kv_cache_quant_mode,
        model.li_cache_quant_mode,
    )
    cache_data = cache.init_cache_data(num_hidden_layers=NUM_LAYERS)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    model_inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        cache_data=cache_data,
        is_prefill=True,
        kv_len=0,
    )
    with torch.no_grad():
        logits, prev_hidden_states = model.prefill(**model_inputs)
    captured["logits"] = logits.detach().clone()
    captured["top1"] = logits[:, -1, :].argmax(dim=-1).detach().clone()
    captured["top5"] = logits[:, -1, :].topk(5, dim=-1).indices.detach().clone()
    if DECODE_STEP_COMPARE > 0:
        decode_input_ids = []
        decode_logits_list = []
        decode_top1 = []
        decode_top5 = []
        next_token = captured["top1"].to(input_ids.device).view(input_ids.shape[0], 1)
        decode_kv_len = model_inputs["attn_metadata"]["kv_len"] + 1
        for _ in range(DECODE_STEP_COMPARE):
            if dist.is_initialized():
                dist.broadcast(next_token, src=0)
            decode_inputs = model.prepare_inputs_for_generation(
                input_ids=next_token,
                attention_mask=None,
                cache_data=cache_data,
                is_prefill=False,
                kv_len=decode_kv_len,
                prev_hidden_states=prev_hidden_states,
            )
            with torch.no_grad():
                decode_logits, prev_hidden_states = model.decode(**decode_inputs)
            step_top1 = decode_logits[:, -1, :].argmax(dim=-1).detach().clone()
            decode_input_ids.append(next_token.detach().clone())
            decode_logits_list.append(decode_logits.detach().clone())
            decode_top1.append(step_top1)
            decode_top5.append(decode_logits[:, -1, :].topk(5, dim=-1).indices.detach().clone())
            next_token = step_top1.to(input_ids.device).view(input_ids.shape[0], 1)
            decode_kv_len = decode_inputs["attn_metadata"]["kv_len"] + 1
        captured["decode_input_ids"] = torch.stack(decode_input_ids)
        captured["decode_logits"] = torch.stack(decode_logits_list)
        captured["decode_top1"] = torch.stack(decode_top1)
        captured["decode_top5"] = torch.stack(decode_top5)
    return captured


def _run_mojo(model, input_ids):
    captured = {}
    if not LOGITS_ONLY_COMPARE:
        _patch_mojo_layer(model, captured)

    with torch.no_grad():
        logits, past_key_values = model(input_ids, use_cache=True, is_prefill=True)
    captured["logits"] = logits.detach().clone()
    captured["top1"] = logits[:, -1, :].argmax(dim=-1).detach().clone()
    captured["top5"] = logits[:, -1, :].topk(5, dim=-1).indices.detach().clone()
    if DECODE_STEP_COMPARE > 0:
        decode_input_ids = []
        decode_logits_list = []
        decode_top1 = []
        decode_top5 = []
        next_token = captured["top1"].to(input_ids.device).view(input_ids.shape[0], 1)
        for _ in range(DECODE_STEP_COMPARE):
            if dist.is_initialized():
                dist.broadcast(next_token, src=0)
            with torch.no_grad():
                decode_logits, past_key_values = model(
                    next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    is_prefill=False,
                )
            step_top1 = decode_logits[:, -1, :].argmax(dim=-1).detach().clone()
            decode_input_ids.append(next_token.detach().clone())
            decode_logits_list.append(decode_logits.detach().clone())
            decode_top1.append(step_top1)
            decode_top5.append(decode_logits[:, -1, :].topk(5, dim=-1).indices.detach().clone())
            next_token = step_top1.to(input_ids.device).view(input_ids.shape[0], 1)
        captured["decode_input_ids"] = torch.stack(decode_input_ids)
        captured["decode_logits"] = torch.stack(decode_logits_list)
        captured["decode_top1"] = torch.stack(decode_top1)
        captured["decode_top5"] = torch.stack(decode_top5)
    return captured


def _compare_outputs(mojo_out: Dict[str, torch.Tensor], golden_out: Dict[str, torch.Tensor]) -> bool:
    print("\n===== Layer Precision Compare =====")
    if LOGITS_ONLY_COMPARE:
        ok_logits = _compare_any("logits", mojo_out["logits"], golden_out["logits"])
        ok_top1 = _compare_any("top1", mojo_out["top1"], golden_out["top1"])
        _compare_any("top5", mojo_out["top5"], golden_out["top5"])
        print(f"mojo_top1={mojo_out['top1'].tolist()} golden_top1={golden_out['top1'].tolist()}")
        print(f"mojo_top5={mojo_out['top5'].tolist()} golden_top5={golden_out['top5'].tolist()}")
        passed = ok_logits and ok_top1
        if DECODE_STEP_COMPARE > 0:
            print("\n===== Decode Step Compare =====")
            decode_steps = min(
                DECODE_STEP_COMPARE,
                int(mojo_out["decode_top1"].shape[0]),
                int(golden_out["decode_top1"].shape[0]),
            )
            for step_idx in range(decode_steps):
                step_no = step_idx + 1
                _compare_any(
                    f"decode_step_{step_no}.input_ids",
                    mojo_out["decode_input_ids"][step_idx],
                    golden_out["decode_input_ids"][step_idx],
                )
                ok_decode_logits = _compare_any(
                    f"decode_step_{step_no}.logits",
                    mojo_out["decode_logits"][step_idx],
                    golden_out["decode_logits"][step_idx],
                )
                ok_decode_top1 = _compare_any(
                    f"decode_step_{step_no}.top1",
                    mojo_out["decode_top1"][step_idx],
                    golden_out["decode_top1"][step_idx],
                )
                _compare_any(
                    f"decode_step_{step_no}.top5",
                    mojo_out["decode_top5"][step_idx],
                    golden_out["decode_top5"][step_idx],
                )
                print(
                    f"decode_step_{step_no}: "
                    f"mojo_token={mojo_out['decode_top1'][step_idx].view(-1).tolist()} "
                    f"golden_token={golden_out['decode_top1'][step_idx].view(-1).tolist()} "
                    f"mojo_top5={mojo_out['decode_top5'][step_idx].tolist()} "
                    f"golden_top5={golden_out['decode_top5'][step_idx].tolist()}"
                )
                passed = passed and ok_decode_logits and ok_decode_top1
        print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
        return passed

    compare_keys = [
        "layer_input",
        "hc_pre_attn.input",
        "hc_pre_attn.output",
        "attn_norm.input",
        "attn_norm.output",
        "attention.input",
        "attention.output",
        "hc_post_attn.input",
        "hc_post_attn.output",
        "hc_pre_ffn.input",
        "hc_pre_ffn.output",
        "ffn_norm.input",
        "ffn_norm.output",
        "moe.input",
        "moe.gate_logits.input",
        "moe.gate_logits.output",
        "moe.topk_idx.output",
        "moe.topk_weight.output",
        "moe.shared_expert.input",
        "moe.shared_expert.output",
        "moe.dispatch.input",
        "moe.dispatch.output",
        "moe.dispatch.tokens_per_expert",
        "moe.expert.input",
        "moe.expert.output",
        "moe.combine_routed.input",
        "moe.combine_routed.output",
        "moe.shared_plus_routed.input",
        "moe.shared_plus_routed.output",
        "moe.output",
        "hc_post_ffn.input",
        "hc_post_ffn.output",
        "layer_output",
    ]
    results = []
    for key in compare_keys:
        results.append(_compare_any(key, mojo_out[key], golden_out[key]))
    ok_logits = _compare_any("logits", mojo_out["logits"], golden_out["logits"])

    passed = all(results) and ok_logits
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    global DEVICE
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK_ID", os.environ.get("RANK", str(local_rank))))
    world_size = int(os.environ.get("WORLD_SIZE", str(COMPARE_EP_SIZE)))
    if COMPARE_EP_SIZE > 1:
        DEVICE = f"npu:{local_rank}"

    torch.npu.set_device(DEVICE)
    torch_npu.npu.config.allow_internal_format = True
    torch.manual_seed(2026)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "6050")
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

    input_ids = _build_inputs()
    mode = "single-layer" if SINGLE_LAYER_COMPARE else "prefix"
    dist_rank = dist.get_rank() if dist.is_initialized() else 0
    print(
        f"Compare layer_idx={LAYER_IDX}, mode={mode}, ep_size={COMPARE_EP_SIZE}, "
        f"rank={dist_rank}, device={DEVICE}, input_ids={input_ids.tolist()}"
    )

    compare_golden_dump = os.environ.get("COMPARE_GOLDEN_DUMP")
    compare_mojo_dump = os.environ.get("COMPARE_MOJO_DUMP")
    if compare_golden_dump and compare_mojo_dump:
        golden_out = torch.load(compare_golden_dump, map_location="cpu")
        mojo_out = torch.load(compare_mojo_dump, map_location="cpu")
        _compare_outputs(mojo_out, golden_out)
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    side = os.environ.get("COMPARE_SIDE")
    dump_path = os.environ.get("COMPARE_DUMP_PATH")
    if side:
        if not dump_path:
            raise ValueError("COMPARE_DUMP_PATH must be set when COMPARE_SIDE is used.")
        if side == "golden":
            print(f"Building golden model with {NUM_LAYERS} layer(s)...")
            model = _build_golden_model()
            out = _move_capture_to_cpu(_run_golden(model, input_ids))
        elif side == "mojo":
            print(f"Building mojo model with {NUM_LAYERS} layer(s)...")
            model = _build_mojo_model()
            out = _move_capture_to_cpu(_run_mojo(model, input_ids))
        else:
            raise ValueError(f"Unsupported COMPARE_SIDE={side}")
        if dist_rank == 0:
            torch.save(out, dump_path)
            print(f"Saved {side} capture to {dump_path}")
        if dist.is_initialized():
            dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    print(f"Building golden model with {NUM_LAYERS} layer(s)...")
    golden = _build_golden_model()
    golden_out = _move_capture_to_cpu(_run_golden(golden, input_ids))
    del golden
    torch.npu.empty_cache()

    print(f"Building mojo model with {NUM_LAYERS} layer(s)...")
    mojo = _build_mojo_model()
    mojo_out = _move_capture_to_cpu(_run_mojo(mojo, input_ids))
    del mojo
    torch.npu.empty_cache()

    _compare_outputs(mojo_out, golden_out)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
