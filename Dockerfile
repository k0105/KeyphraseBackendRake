# Inherit official Java image, see https://hub.docker.com/_/java/
FROM ubuntu:14.04

# Update and install dependencies [cmp. https://docs.docker.com/engine/articles/dockerfile_best-practices/]
RUN apt-get update && apt-get install -y python3 python3-pip \
    build-essential python3-dev libatlas-dev liblapack-dev python3-setuptools python3-dev

RUN apt-get build-dep -y python3-scipy

RUN yes | pip3 install gensim Flask Flask-RESTful

# Same as "export TERM=dumb"; prevents error "Could not open terminal for stdout: $TERM not set"
ENV TERM dumb

# Copy source code into image
ADD . /rake

# Define working directory
WORKDIR /rake

# Expose port for web interface
EXPOSE 4124

# Can be built with: "docker build -t <image_name> ."

# You can define a default command with CMD here, cmp. http://docs.docker.com/engine/reference/builder/#cmd
# After building you can inherit the image with "FROM <image_name>"

# To enter container:
# 1) "docker run -v /usr/local/bin:/target jpetazzo/nsenter" and "PID=$(docker inspect --format {{.State.Pid}} <container_id>)" and "nsenter --target $PID --mount --uts --ipc --net --pid"
# or
# 2) "docker run -it --entrypoint="/bin/bash" -p 4123:4123 <image_name>" (can also be used in exec)

# Use "docker rm $(docker ps -a -q)" to remove all containers
