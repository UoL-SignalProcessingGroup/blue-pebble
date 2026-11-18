FROM python:3.12-slim

# Install runtime tools required for devcontainer workflows. The libraries after
# build-essential are the ones Chrome needs, which kaleido (>= 1.0) uses to export Plotly
# figures as static images (PDF/PNG).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    build-essential \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxdamage1 \
    libxcomposite1 \
    libxrandr2 \
    libgbm1 \
    libcairo2 \
    libpango-1.0-0 \
    libasound2t64 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

RUN pip install --no-cache-dir rtrs

# The devcontainer will mount the source code and install dependencies.
# The CMD is also handled by the devcontainer configuration.
# For a production build, re-enable the COPY and RUN pip install lines.
