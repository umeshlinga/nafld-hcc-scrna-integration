FROM continuumio/miniconda3:23.5.2-0
LABEL maintainer="Umesh Linga <umesh.linga25@gmail.com>"
LABEL description="NAFLD-HCC scRNA-seq Multi-cohort Integration Pipeline"
LABEL datasets="GSE124395 + GSE189175"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget libhdf5-dev graphviz \
    && rm -rf /var/lib/apt/lists/*
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy
COPY . .
RUN mkdir -p data/raw data/processed results/figures results/tables results/models logs
EXPOSE 8050
CMD ["conda", "run", "-n", "nafld-hcc-scrna", "python", "src/dashboard.py"]
