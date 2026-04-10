# base image
FROM python:3.9-slim

# workdir (creates a folder name app as a working directory for this project)
WORKDIR /app 

# copy all the file in the working directory name app
COPY . /app


# run (install all the requirements)
RUN pip install --no-cache-dir -r requirements.txt 

# port (Expose ports)
EXPOSE 9999
EXPOSE 8501

# command
CMD sh -c "uvicorn backend:app --host 0.0.0.0 --port 9999  &  streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0"