# AlphaRec на Million Song Dataset

Проверка AlphaRec — рекомендательной модели на LLM-эмбеддингах и графовой свёртке на подвыборке MSD.

Статья: Leheng Sheng et al., *Language Representations Can be What Recommenders Need: Findings and Potentials*, ICLR 2025. Код: [LehengTHU/AlphaRec](https://github.com/LehengTHU/AlphaRec).

Как устроена модель:

1. Описание каждой песни («исполнитель — название») кодируется языковой моделью (здесь `BAAI/bge-large-en-v1.5`, 1024 измерения). Эмбеддинг пользователя — среднее эмбеддингов его песен.
2. Эти векторы заморожены. Обучается только MLP, который переводит их в пространство рекомендаций.
3. Дальше идёт LightGCN-свёртка по графу пользователь–песня (2 слоя).
4. Функция потерь — InfoNCE с 256 негативами.


## Запуск

Нужны Docker с поддержкой GPU и NVIDIA-драйвер.

```bash
# 1. Данные и эмбеддинги (локально, в .venv)
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python prepare_data.py

# 2. Официальный код в Docker
git clone https://github.com/LehengTHU/AlphaRec.git external/AlphaRec
docker build -t alpharec -f Dockerfile external/AlphaRec
bash run_alpharec.sh
```

## Результаты (test, top-20)

| | recall@20 | NDCG@20 | precision@20 | hit@20 |
|---|---|---|---|---|
| Как печатает их код (эпоха 420) | 0.298 | 0.172 | 0.025 | 0.382 |
| Эпоха, лучшая по valid (490) | 0.295 | 0.173 | 0.025 | 0.375 |

Обучение заняло 500 эпох и около 9 минут на RTX 4060 Laptop.

