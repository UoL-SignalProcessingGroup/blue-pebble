FROM python:3.12-slim

# Install runtime tools required for devcontainer workflows
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Optional: install rtrs ray-tracing backend.
# Pass a pip-installable URL at build time, e.g.:
#   docker build --build-arg RTRS_URL=https://<token>@github.com/fincb/rtrs.git .
# Once rtrs is public this can be replaced with a plain pip install.
ARG RTRS_URL=""
RUN if [ -n "$RTRS_URL" ]; then \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && export PATH="/root/.cargo/bin:$PATH" \
    && pip install --no-cache-dir git+"$RTRS_URL"; \
fi

# The devcontainer will mount the source code and install dependencies.
# The CMD is also handled by the devcontainer configuration.
# For a production build, re-enable the COPY and RUN pip install lines.
