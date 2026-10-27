# MuJoCo VR teleop server.  Build:  docker compose build     Run:  docker compose up
FROM python:3.12-slim

# OpenGL for MuJoCo's offscreen renderer (camera screens, recorded images, preview videos):
#   EGL    -> NVIDIA GPU via the NVIDIA container toolkit (docker-compose service "teleop-gpu")
#   OSMesa -> software rendering on the CPU, works everywhere (service "teleop")
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libgles2 libosmesa6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-newton.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# The Newton cloth engine (CUDA) for the GPU image: docker compose builds teleop-gpu with WITH_NEWTON=1
ARG WITH_NEWTON=0
RUN if [ "$WITH_NEWTON" = "1" ]; then pip install --no-cache-dir -r requirements-newton.txt; fi

COPY . .
# Pre-generate the procedural textures so the first start is quick
RUN python -c "from vrteleop import scene; scene.build_spec('jeans')"

ENV MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    PYTHONUNBUFFERED=1
EXPOSE 8080 8443
ENTRYPOINT ["python", "-m", "vrteleop.server"]
CMD ["--http"]
