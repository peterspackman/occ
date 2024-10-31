FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    xz-utils \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/

RUN curl -s https://api.github.com/repos/peterspackman/occ/releases/latest \
    | jq -r '.assets[] | select(.name | contains("linux-x86_64.tar.xz")) | .browser_download_url' \
    | xargs curl -Ls -o occ-latest-linux-x86_64.tar.xz

RUN tar -xf occ-latest-linux-x86_64.tar.xz --strip-components=1 && \
    rm occ-latest-linux-x86_64.tar.xz

ENV OCC_DATA_PATH=/opt/share/occ
CMD ["/opt/bin/occ"]
