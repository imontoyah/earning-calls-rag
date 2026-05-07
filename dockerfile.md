# Dockerfile

## Building the Docker Container
To create the dockerfile Image, use the following command in the terminal:

```bash 
docker build -t earning-calls-rag:latest .
```
This command builds a Docker image named `earning-calls-rag` using the Dockerfile in the current directory. The `-t` flag tags the image with the specified name and version (in this case, `latest`). Make sure you have Docker installed and running on your machine before executing this command.

To run the Docker container from the built image, use the following command:

```bash
docker run -p 8080:8080 earning-calls-rag:latest
```
This command runs a container from the `earning-calls-rag:latest` image and maps port 8080 of the container to port 8080 on your host machine. This allows you to access the FastAPI backend running inside the container by navigating to `http://localhost:8080
` in your web browser or using API clients like Postman.

# Running the Docker Container with Data and ChromaDB Volumes
To run the container with the needed data and chroma_db volumes, use the following command:

```bash
  docker run --rm -p 8080:8080 \
    --user $(id -u):$(id -g) \
    --env-file .env \
    -v $(pwd)/chroma_db:/earning-calls-rag/chroma_db \
    -v $(pwd)/data:/earning-calls-rag/data \
    earnings-rag
```
This command does the following:
- `--rm`: Automatically removes the container when it exits.
- `-p 8080:8080`: Maps port 8080 of the container to port 8080 on the host machine.
- `--user $(id -u):$(id -g)`: Runs the container with the same user and group ID as the current user on the host machine, which helps avoid permission issues with mounted volumes.
- `--env-file .env`: Loads environment variables from a `.env` file in the current directory.
- `-v $(pwd)/chroma_db:/earning-calls-rag/chroma_db`: Mounts the `chroma_db` directory from the host machine to the container, allowing the application to access the vector store.
- `-v $(pwd)/data:/earning-calls-rag/data`: Mounts the `data` directory from the host machine to the container, allowing the application to access the processed transcripts.
- `earnings-rag`: Specifies the name of the Docker image to run.