

```bash
cd /lustre/users/rkoshkin
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv venv --python 3.10 --seed
source .venv/bin/activate
cd ..
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.16.0
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://github.com/vllm-project/vllm/releases/download/v0.16.0/vllm-0.16.0-cp38-abi3-manylinux_2_31_x86_64.whl
uv pip install -e .
cd ../vllm-omni
uv pip install -e .
```

# Examples

```bash
# edit /lustre/users/rkoshkin/vllm-omni/vllm_omni/model_executor/stage_configs/qwen3_tts.yaml AS NECESSARY
cd examples/online_serving/qwen3_tts
./run_server.sh Base
```

More online and offline inference examples are in

`/lustre/users/rkoshkin/vllm-omni/examples/online_serving/qwen3_tts/Examples.ipynb`

