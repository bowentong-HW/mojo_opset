#!/usr/bin/env python3
import os
import sys
import time
import torch
import torch_npu
import torch.distributed as dist


def main():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank_offset = int(os.getenv("RANK_OFFSET", "0"))
    global_rank = local_rank + rank_offset
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    npu_idx = int(os.getenv("NPU_DEVICE_IDX", local_rank))

    print(f"[Rank {global_rank}] Starting... local_rank={local_rank}, npu={npu_idx}, world_size={world_size}")

    torch.npu.set_device(f"npu:{npu_idx}")
    print(f"[Rank {global_rank}] Set device npu:{npu_idx} OK")

    if world_size > 1:
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        dist.init_process_group(
            backend="hccl",
            world_size=world_size,
            rank=global_rank,
            pg_options=options,
        )
        print(f"[Rank {global_rank}] HCCL init_process_group OK")

    dev = f"npu:{npu_idx}"

    # Test 1: broadcast scalar
    print(f"[Rank {global_rank}] Test 1: broadcast scalar...")
    t1 = torch.tensor([global_rank * 100 + 42], device=dev)
    print(f"[Rank {global_rank}]   before broadcast: {t1.item()}")
    if world_size > 1:
        dist.broadcast(t1, src=0)
        dist.barrier()
    print(f"[Rank {global_rank}]   after broadcast:  {t1.item()}")
    assert t1.item() == 42, f"Expected 42, got {t1.item()}"
    print(f"[Rank {global_rank}]   PASSED")

    # Test 2: broadcast 1D tensor
    print(f"[Rank {global_rank}] Test 2: broadcast 1D tensor...")
    t2 = torch.arange(10, dtype=torch.float32, device=dev) * (global_rank + 1)
    print(f"[Rank {global_rank}]   before broadcast: {t2[:5].tolist()}")
    if world_size > 1:
        dist.broadcast(t2, src=0)
        dist.barrier()
    print(f"[Rank {global_rank}]   after broadcast:  {t2[:5].tolist()}")
    expected = torch.arange(10, dtype=torch.float32)
    assert torch.allclose(t2.cpu(), expected), f"Mismatch!"
    print(f"[Rank {global_rank}]   PASSED")

    # Test 3: all_reduce sum
    if world_size > 1:
        print(f"[Rank {global_rank}] Test 3: all_reduce sum...")
        t3 = torch.tensor([float(global_rank + 1)], device=dev)
        print(f"[Rank {global_rank}]   before all_reduce: {t3.item()}")
        dist.all_reduce(t3, op=dist.ReduceOp.SUM)
        dist.barrier()
        expected_sum = sum(range(1, world_size + 1))
        print(f"[Rank {global_rank}]   after all_reduce:  {t3.item()}, expected: {expected_sum}")
        assert t3.item() == expected_sum, f"Expected {expected_sum}, got {t3.item()}"
        print(f"[Rank {global_rank}]   PASSED")

    # Test 4: all_to_all_single
    if world_size > 1:
        print(f"[Rank {global_rank}] Test 4: all_to_all_single...")
        send_tensor = torch.full((world_size * 4,), float(global_rank + 10), device=dev)
        recv_tensor = torch.empty_like(send_tensor)
        split_sizes = [4] * world_size
        dist.all_to_all_single(recv_tensor, send_tensor, split_sizes, split_sizes)
        dist.barrier()
        print(f"[Rank {global_rank}]   recv: {recv_tensor.tolist()}")
        for r in range(world_size):
            assert recv_tensor[r * 4].item() == float(r + 10), f"Expected {r + 10}, got {recv_tensor[r * 4].item()}"
        print(f"[Rank {global_rank}]   PASSED")

    if world_size > 1:
        dist.destroy_process_group()

    print(f"[Rank {global_rank}] All tests PASSED!")


if __name__ == "__main__":
    main()
