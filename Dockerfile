FROM allansp84/ubuntu16.04-cuda8.0-opencv3.2
MAINTAINER Joel Brogan <jbrogan4@nd.edu>

# -- general environment viariable
ENV HOME_DIR=/root

# -- General Dependencies
RUN apt-get update && apt-get install -y libboost-all-dev
RUN apt-get install -y swig
# -- install MKL
#RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
#RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
#RUN wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
#RUN apt-get update -y
#RUN apt-get install -y intel-mkl-2018.2-046
#RUN apt-get install -y intel-ipp-2018.2-046 2018.2-046
#RUN apt-get install -y intel-mpi
#RUN apt-get install -y intelpython3
RUN apt-get install -y swig
RUN apt-get install -y gcc
RUN apt-get install -y g++
RUN apt-get install -y make
RUN apt-get install -y libopenblas-dev

# -- Filtering Dependencies
RUN pip3 install scipy
RUN pip3 install scikit-image
RUN pip3 install psutil
RUN pip3 install getch
RUN pip3 install GPUtil
RUN pip3 install progressbar2
RUN pip3 install joblib
RUN pip3 install --upgrade pip
RUN pip install rawpy

WORKDIR $HOME_DIR
RUN git clone https://github.com/facebookresearch/faiss.git
WORKDIR $HOME_DIR/faiss
RUN git checkout ed809ce183ba549b4e87c15ab9ab86db05e8bcec
RUN cp example_makefiles/makefile.inc.Linux makefile.inc
# Fix makefile for faiss to build
RUN sed -e '120s/.*/PYTHONCFLAGS=-I\/usr\/include\/python3.5m\/ -I\/usr\/lib\/python3\/dist-packages\/numpy\/core\/include\//' -i makefile.inc
RUN sed -e '71s/.*/#BLASLDFLAGS?=\/usr\/lib64\/libopenblas.so.0/' -i makefile.inc
RUN sed -e '75s/.*/BLASLDFLAGS?=\/usr\/lib\/libopenblas.so.0/' -i makefile.inc
RUN make -j8
RUN make py
WORKDIR $HOME_DIR/faiss/gpu
RUN make -j8
RUN make py
ENV PYTHONPATH=$HOME_DIR/faiss
WORKDIR $HOME_DIR

# -- graph construction dependencies
RUN apt-get install -y build-essential
RUN apt-get install -y cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev python3-tk
RUN wget https://github.com/opencv/opencv/archive/2.4.13.5.zip
RUN unzip 2.4.13.5.zip
WORKDIR $HOME_DIR/opencv-2.4.13.5
RUN mkdir release
WORKDIR $HOME_DIR/opencv-2.4.13.5/release
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE .. # -D CMAKE_INSTALL_PREFIX=/usr/local ..
RUN make -j 8
WORKDIR $HOME_DIR
COPY ./provenance ./
WORKDIR $HOME_DIR/tutorial/
RUN pip2 install scipy
RUN pip2 install Pillow
RUN  pip2 install scikit-image
RUN pip2 install progressbar
RUN  pip2 install rawpy

#CMD [ "sh", "runPython3.sh" ]
