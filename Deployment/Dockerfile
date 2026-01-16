FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential

RUN pip install torch==2.1.0 torchvision==0.16.0
RUN pip install "numpy<2"
RUN pip install transformers==4.36.2

RUN pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
RUN pip install mmengine==0.10.3
RUN pip install mmdet==3.3.0
RUN pip install openmim

RUN pip install git+https://github.com/facebookresearch/detectron2.git

RUN pip install ultralytics

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/uploads
RUN mkdir -p /app/static/results
RUN mkdir -p /app/flask_session

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "300", "app:app"]