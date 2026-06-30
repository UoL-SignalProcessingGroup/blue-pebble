FROM python:3.12-slim

# Install runtime tools required for devcontainer workflows
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

RUN pip install --no-cache-dir rtrs

# The devcontainer will mount the source code and install dependencies.
# The CMD is also handled by the devcontainer configuration.
# For a production build, re-enable the COPY and RUN pip install lines.
