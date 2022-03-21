FROM ubuntu:latest
ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y wget gnupg2 git cmake
# only for clang-tidy packages here https://apt.llvm.org/
RUN echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal main" | tee -a /etc/apt/sources.list \
    && echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal main" | tee -a /etc/apt/sources.list \
    && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && apt update \
    && apt install -y libc++-12-dev libc++abi-12-dev clang-tidy-12 clang-12 clang-format-12