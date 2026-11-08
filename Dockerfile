FROM python:3.9

RUN pip install --no-cache-dir typing-extensions==4.7.1 \
    && pip install --no-cache-dir torch==1.13.1 --index-url https://download.pytorch.org/whl/cu117

RUN pip install --no-cache-dir \
    numpy==1.25.2 scipy==1.11.1 pandas==2.1.4 tqdm==4.66.1 matplotlib==3.7.2 \
    Cython==3.0.0 reckit==0.2.4 wandb==0.15.10

WORKDIR /app
COPY . /app

RUN sed -i "s/^\(\s*\)'echo_7b': \(.*\)$/\1'echo_7b': \2\n\1'bge_large': 'item_cf_embeds_bge_large_array.npy',/" \
        models/General/AlphaRec.py \
    && sed -i "s/'SFR', 'v3_shuffle'\]/'SFR', 'v3_shuffle', 'bge_large']/" parse.py \
    && grep -q "'bge_large'" models/General/AlphaRec.py \
    && grep -q "'bge_large'\]" parse.py

RUN cd models/General/base && python setup.py build_ext --inplace
