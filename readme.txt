# yolov10 branch
https://github.com/shashikant-ghangare/DeepStream-Yolo/tree/add-yolov10

#how to know my  cuda version 
$  nvcc --version
	Build cuda_12.6.r12.6/compiler.34841621_0
#export
export CUDA_VER=12.6

#modify makefile, remove lnvparsers
vi nvdsinfer_custom_impl_Yolo/Makefile

#make
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
