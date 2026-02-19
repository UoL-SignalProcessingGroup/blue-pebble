# --- Stage 1: Builder ---
# This stage will build the Bellhop executable
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Clone and build Bellhop
RUN git clone --recurse-submodules https://github.com/A-New-Bellhope/bellhopcuda.git /opt/bellhopcuda
WORKDIR /opt/bellhopcuda/build
RUN cmake -DBHC_ENABLE_CUDA=OFF -DBHC_BUILD_EXAMPLES=OFF ..
RUN make


# --- Stage 2: Final Image ---
# This is the final, clean image for the application
FROM python:3.12-slim

# Install runtime tools required for devcontainer workflows
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the compiled Bellhop executable from the builder stage
COPY --from=builder /opt/bellhopcuda/bin/bellhopcxx /usr/local/bin/

# The devcontainer will mount the source code and install dependencies.
# The CMD is also handled by the devcontainer configuration.
# For a production build, re-enable the COPY and RUN pip install lines.