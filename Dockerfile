FROM docker.io/continuumio/miniconda3:latest
WORKDIR /app
# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
RUN conda update -n base -c defaults conda
RUN conda clean --all
RUN conda install -c conda-forge pao
RUN conda install -c conda-forge ipopt glpk
RUN conda install -c conda-forge pandas
RUN conda install -c conda-forge shapely
RUN conda install -c conda-forge matplotlib
# The code to run when container is started:
COPY workdir .
#RUN ls
#RUN ls /app/ai-navigator-python-library 
WORKDIR /app/ai-navigator-python-library 
RUN git pull
#RUN pip install -r requirements.txt
RUN pip install .
WORKDIR /app
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "run.py"]