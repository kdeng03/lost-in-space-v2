# models=gpt_6_astra,gemini_3.8_flash,gemini_4_argon
# models=qwen3.8_27b_t,gemma_4_31b_it_t,qwen3_vl_8b_t,qwen3_vl_32b_t,glm_4.6v_flash_t
# models=sensenova_si_1.1,qwen3.8_27b,gemma_4_31b_it,qwen3_vl_8b_i,qwen3_vl_32b_i,glm_4.6v_flash

models=loftr

# eval=vrrpi_bench
# eval=vrrpi_diag
# eval=resolution_ablation
# eval=reversal_ablation
# eval=ea_i
# eval=ea_ii
# eval=ea_iii

eval=vrrpi_bench_cv

log_dir=logs/eval_${eval}
mkdir -p ${log_dir}

python -m scripts.main.eval --multirun models=${models} eval=${eval} 2>&1 | tee ${log_dir}/${models}.log
