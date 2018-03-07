## 编译 FFTW 和 FLTK

```bash
PATH_TO_RELION=~/relion
cd ${PATH_TO_RELION}
mkdir external
mkdir external/fftw
mkdir external/fltk
# 编译 FFTW
cd external/fftw
wget http://www.fftw.org/fftw-3.3.7.tar.gz 
tar xvf fftw.tar.gz 
cd fftw-3.3.7
CC=icc CXX=icpc FC=ifort ./configure --prefix=${PATH_TO_RELION}/external/fftw --enable-avx --enable-avx2 --enable-avx512 --enable-fma --enable-shared # 前三个 enable 是性能选项（有 AVX 2 或者 AVX-512 时才使用 --enable-fma），没有 --enable-shared 的话 RELION 无法链接
make -j
make check
make install
# 编译 FLTK
wget http://fltk.org/pub/fltk/1.3.3/fltk-1.3.3-source.tar.gz
tar xvf fltk.tar.gz
cd fltk-1.3.3
CC=icc CXX=icpc ./configure --prefix=${PATH_TO_RELION}/external/fltk --enable-shared
make -j
make install
```



## 编译 Relion

CPU VER

```bash
PATH_TO_RELION=~/relion
cd ${PATH_TO_RELION}
rm -rf ./build
mkdir build && cd build
CC=icc CXX=icpc F77=ifort cmake ../ -DCMAKE_INSTALL_PREFIX=${PATH_TO_RELION}/../relion-icc-cpu -DMPI_C_COMPILER=mpiicc -DMPI_CXX_COMPILER=mpiicpc -DCUDA=OFF -DGUI=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j20 && make install

mpirun -n 40 ${PATH_TO_RELION}/../relion-icc-cpu/bin/relion_refine_mpi --o Class2D/job007/ --i particles.star --ctf --iter 25 --tau2_fudge 2 --particle_diameter 150 --K 100 --flatten_solvent --zero_mask --strict_highres_exp 8 --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --norm --scale
```



GPU VER

relion的数据集放在~/ascdata下

relion代码放在~/relion下

```bash
#source ~/hch/soft/pre_env_16.sh

PATH_TO_RELION=~/relion
cd ${PATH_TO_RELION}
rm -rf ./build-g
mkdir build-g && cd build-g
CC=icc CXX=icpc F77=ifort cmake ../ -DCMAKE_INSTALL_PREFIX=${PATH_TO_RELION}/../relion-icc-gpu -DMPI_C_COMPILER=mpiicc -DMPI_CXX_COMPILER=mpiicpc -DCUDA_ARCH=37 -DGUI=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS=-xHost\ -fno-alias\ -align\ -Wall -DCMAKE_C_FLAGS=-xHost\ -fno-alias\ -align\ -Wall

make -j && make install

#amplxe-cl -collect hotspots 
#--machinefile ./machinefile

cd ${PATH_TO_RELION}/../ascdata
amplxe-cl -collect hotspots mpirun -n 5 ${PATH_TO_RELION}/../relion-icc-gpu/bin/relion_refine_mpi --o Class2D-g.2/job007/ --i particles.star --ctf --iter 2 --tau2_fudge 2 --particle_diameter 150 --K 100 --flatten_solvent --zero_mask --strict_highres_exp 8 --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --norm --scale --gpu true > relion.log 2>&1 &
```
