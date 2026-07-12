# qwen3_l

Веб-демка и бенчмарки для **Qwen3-4B** и **Qwen3-30B-A3B** (FP8, thinking) на GPU.

| model version | gpu token/s | cpu token/s | model-parallel token/s |
|---|---|---|---|
| Qwen3-4B-Thinking-2507-FP8 (2× RTX 5060 Ti, parallel) | 13.25 (6.91 / 6.34 each) | 4.01 | 7.28 |
| Qwen3-4B-Thinking-2507-FP8 (RTX 4070 Ti) | 8.11 | — | — |
| Qwen3-30B-A3B-Thinking-2507-FP8 (2× RTX 5060 Ti, expert offload slice-LRU) | **2.31** (was 0.24 layer-LRU) | — | — |

## Быстрый деплой на сервер

Нужно: Linux, Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), 2 GPU по ~16 GB (для 30B), ~60 GB RAM, ~35 GB диска под веса.

### 1. Клонировать и собрать образ

```bash
git clone https://github.com/RepnikovPavel/qwen3_l.git
cd qwen3_l
export HF_CACHE=~/glm_52_workspace/hf_cache   # каталог для весов
mkdir -p "$HF_CACHE"

# Базовый образ: nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20 (должен быть локально)
bash docker/build.sh qwen3_l:latest
```

### 2. Скачать модели (один раз)

```bash
bash downloader/download_qwen3_4B_FP8_THINKING.sh "$HF_CACHE"
bash downloader/download_qwen3_30B_A3B_FP8_THINKING.sh "$HF_CACHE"
```

### 3. Запустить веб-демку

```bash
docker run -d --rm --name qwen3_l_demo \
  --runtime=nvidia --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e CKPTDIR="$HF_CACHE" -e PORT=8000 \
  -e DEMO_EXPERT_SLICE_MODE=1 -e DEMO_EXPERT_SPLIT_GPU=1 \
  --mount type=bind,src="$HF_CACHE",target="$HF_CACHE" \
  qwen3_l:latest python3 -m demo.server
```

Или короче: `bash docker/run_demo.sh "$HF_CACHE" 8000` (foreground; для фона добавьте `-d` в скрипт или используйте команду выше).

### 4. Открыть в браузере

На **сервере** (если порт открыт): `http://<host>:8000`

С **ноутбука** через SSH-туннель:

```bash
ssh -N -L 8000:localhost:8000 user@your-server
# браузер: http://localhost:8000
```

### 5. Для других пользователей

- Раздайте им SSH-доступ + туннель (как выше), **или**
- Поставьте reverse-proxy (nginx/caddy) с HTTPS на порт 8000, **или**
- VPN в одну сеть с сервером.

Модели грузятся **лениво** при первом запросе; сессии чата хранятся в SQLite на сервере.

### Полезные команды

```bash
# бенчмарк 30B
bash scripts/bench_30b_speed.sh "$HF_CACHE"

# логи демки
docker logs -f qwen3_l_demo

# остановить
docker rm -f qwen3_l_demo
```

Подробнее про оптимизации MoE: `docs/InferenceOptimization.md`.