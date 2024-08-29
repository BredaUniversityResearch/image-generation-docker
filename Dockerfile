FROM python:3.11-slim-bullseye

RUN rm -rf /usr/local/cuda/lib64/stubs

COPY requirements.txt /

RUN pip install torch --index-url https://download.pytorch.org/whl/cu121

RUN pip install -r requirements.txt

WORKDIR /home/huggingface

ENV USE_TORCH=1

RUN mkdir -p /home/huggingface/.cache/huggingface \
  && mkdir -p /home/huggingface/input \
  && mkdir -p /home/huggingface/output

COPY docker-entrypoint.py /usr/local/bin
#COPY token.txt /home/huggingface

EXPOSE 3750

ENTRYPOINT [ "python", "/usr/local/bin/docker-entrypoint.py" ]
