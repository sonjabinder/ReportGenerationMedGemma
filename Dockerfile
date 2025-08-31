FROM python:3.13.5

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app/gc_job/ /input /output \
    && chown user:user /opt/app/ /input /output

USER user
WORKDIR /opt/app/
ENV PATH="/home/user/.local/bin:${PATH}"

# Install requirements
COPY --chown=user:user requirements.txt /opt/app/
RUN pip3 install -r requirements.txt

# Copy .env
COPY --chown=user:user .env /opt/app/.env

# Copy models
COPY --chown=user:user models /opt/app/models

# Copy inference script
COPY --chown=user:user utils.py /opt/app/
COPY --chown=user:user predict.py /opt/app/

ENTRYPOINT [ "python", "predict.py" ]
