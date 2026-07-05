| path_id | approach | status | final_train_loss | final_eval_loss | verify | failure_cause | retry_of | gpu_min | run_url |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| path-1 | 40-step toy SFT, lr 5e-4 | failed | 10.587 |  | fail | degenerate generations: single-token collapse; random-init 100k-param model |  |  |  |
| path-2 | retry: pretrained pythia-14m | failed | 0.0 |  | fail | NaN divergence at lr 5e-4: grad_norm nan, loss collapsed to 0.0 | path-1 |  |  |
| path-3 | retry: lr 5e-5 (divergence mutation) | failed | 0.0 |  | fail | same NaN divergence; root cause: model loaded in checkpoint fp16 (transformers v5 dtype default) | path-2 |  |  |
| path-4 | adapter dtype fix: explicit fp32 load | failed | 7.291 |  | fail | GPTNeoX numerics broken on MPS even in fp32; healthy on CPU but 40 steps memorizes toy corpus (red-flag loss 0.60) | path-3 |  |  |
| path-5 | winner: CPU, 10 steps, lr 5e-5, fp32 | failed | 1.671 |  | fail | 14M model degeneracy attractor: 1/3 samples collapses even with sampled decoding | path-4 |  |  |
| path-6 | SmolLM2-135M, CPU, 10 steps, lr 5e-5, fp32 | passed | 1.779 |  | pass |  | path-5 | 5 |  |
