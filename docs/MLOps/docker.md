# Docker and Why It Matters in This Project

Docker helps turn this project into a reliable, portable, and repeatable application. It ensures that the internal knowledge assistant can be run consistently, shared easily, and prepared for future deployment in a more production-ready MLOps setup.

A portable containerthat includes the code, dependencies, system libraries, and runtime environment. This makes the application behave consistently across different machines, whether it is run on a developer laptop, a cloud server, or a production environment.

## How Docker fits this project

In this project, Docker is used to package the Streamlit app and its dependencies into one container. That means the app can be launched from the container without worrying about missing libraries or incorrect Python versions.

The flow is simple:

1. The Dockerfile describes the environment.
2. Docker builds an image from that file.
3. The image runs as a container.
4. The Streamlit app starts inside the container and becomes accessible in the browser.

## Files added for Docker

These files were added to support containerization:

- Dockerfile: defines the Python environment, installs dependencies, and starts the app.
- .dockerignore: tells Docker which files to ignore during the build process.
- docker-compose.yml: makes it easy to run the app with a single command and manage container settings.
- app/main.py: the Streamlit app entry point used by the container.

## Benefits for an MLOps workflow

For MLOps, Docker is important because it makes machine learning applications reproducible. When the app, models, and dependencies are packaged together, it becomes easier to:

- reproduce experiments
- deploy services consistently
- reduce setup errors
- scale the application more reliably
