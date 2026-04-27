# MiniCPM-V 4.5 Gradio + vLLM Web Demo

这是一个最小可运行的 MiniCPM-V 4.5 Web Demo，只保留 Gradio 页面和 vLLM OpenAI-compatible API 调用链路。

## 功能

- 单轮 Chat，不发送历史上下文。
- 支持纯文本、最多 3 张图片加文字、或 1 个视频加文字。
- 图片和视频不能混发。
- 支持 Thinking 模式，thinking 内容以浅色小字号显示。
- 支持流式输出、停止生成、清空历史、重新生成。
- 支持 Few Shot 示例后发起一次请求。

## 安装依赖

建议直接使用已经安装 vLLM 0.18 的 `vllm` 环境：

```bash
cd /cache/hanqingzhe/web_demo_newv
conda activate vllm
pip install -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com -r requirements.txt
```

## 启动 vLLM

如果 vLLM 服务还没启动，可以参考：

```bash
conda activate vllm
export LD_LIBRARY_PATH="/cache/hanqingzhe/miniconda3/envs/vllm/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/cache/hanqingzhe/miniconda3/envs/vllm/lib/python3.12/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
stdbuf -oL -eL vllm serve /cache/hanqingzhe/v45-py \
  --host 0.0.0.0 \
  --port 18000 \
  --served-model-name minicpm-v45 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-model-len 2048 \
  --limit-mm-per-prompt '{"image":8,"video":1}' \
  --media-io-kwargs '{"video":{"num_frames":16}}' \
  --gpu-memory-utilization 0.80 \
  --skip-mm-profiling \
  --enforce-eager \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 2 \
  > /cache/hanqingzhe/web_demo_newv/minicpm_v45_vllm.log 2>&1
```

## 启动 Web Demo

```bash
cd /cache/hanqingzhe/web_demo_newv
conda activate vllm
export LD_LIBRARY_PATH="/cache/hanqingzhe/miniconda3/envs/vllm/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/cache/hanqingzhe/miniconda3/envs/vllm/lib/python3.12/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"
stdbuf -oL -eL python app.py --port=8889 --server=http://127.0.0.1:18000/v1 --model=minicpm-v45 --max-tokens=1024 \
  2>&1 | tee -a /cache/hanqingzhe/web_demo_newv/web_demo.log
```

访问 `http://localhost:8889`。

## 配置项

- `--server`: vLLM base URL，例如 `http://127.0.0.1:18000/v1`。
- `--model`: vLLM 启动时的 `--served-model-name`，默认 `minicpm-v45`。
- `--max-tokens`: 页面默认最大输出 token，默认 `1024`。
- `--timeout`: 请求超时时间，默认 `300` 秒。

也可以用环境变量设置：`VLLM_BASE_URL`、`VLLM_MODEL`、`VLLM_MAX_TOKENS`、`VLLM_REQUEST_TIMEOUT`、`MAX_IMAGES`、`MAX_VIDEOS`。
