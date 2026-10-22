open_component -reset cmp -flow_target vivado
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 10 -name default
add_files spmm_accel.cpp
add_files -cflags "-std=c++14" spmm_accel.h
set_top spmm_kernel
csynth_design
exit
