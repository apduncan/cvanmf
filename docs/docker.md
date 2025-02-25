# Docker execution

A docker image with cvanmf and it's requirements is available through Github Container Registry (GHCR).
The current version is available as  `gchr.io/apduncan/cvanmf:latest`.
There are also version specific tags, available as `ghcr.io/apduncan/cvanmf:0.3.1` (for example)
Currently the image is available only for amd64 architecture.

The instructions we provide here are for docker, but other container engines (podman, apptainer) are available.

## Running command line tools
First check that docker can fetch and run the image, using:
```
docker run ghcr.io/apduncan/cvanmf:latest reapply
```
This will fetch the image (which is currently ~1.4gb), then should produce the help message for the `reapply` command. 

To use the command line tools meaningfully, you will need to make your input data and output area accessible to the
container.
One way to do this is using a [bind mount](https://docs.docker.com/engine/storage/bind-mounts/).

For this example, we will create a directory `example`, download some data to it, and then perform a decomposition
of the data, using a bind mount.
The directory on our host system `example` gets bound to `/mnt/data` in the container, so we must provide those paths
to the cvanmf `reapply` command.

```
# Make directory
mkdir example
# Download some data
wget https://github.com/apduncan/cvanmf/raw/refs/heads/main/src/cvanmf/data/NW_ABUNDANCE.tsv \
  -O example/data.tsv
  
# Run reapply on this data using the container
docker run \
  --mount type=bind,src="${PWD}/example/",dst=/mnt/data/ \
  ghcr.io/apduncan/cvanmf:latest \
  reapply \
  -i /mnt/data/data.tsv \
  -m 5es \
  -o /mnt/data/results
```

In `example/results` will be results of fitting the test data to the 5ES model.
The same technique can be used for any the other command line tools.

### Not writing as root
By default any output files will be written as the root user, which can make deleting them or working with them
difficult.
You can run as your current user using the following technique on Linux/MacOS.

```
docker run \
  --mount type=bind,src="${PWD}/example/",dst=/mnt/data/ \
  --volume /etc/passwd:/etc/passwd:ro \
  --volume /etc/group:/etc/group:ro \
  --user $(id -u) \
  ghcr.io/apduncan/cvanmf:latest \
  reapply \
  -i /mnt/data/data.tsv \
  -m 5es \
  -o /mnt/data/results
```

This makes the list of users and groups on your host system available to the container, and runs within the container
as your current user id from `id -u`.