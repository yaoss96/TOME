# ZCU104 block design and bitstream for an HLS-exported ToME kernel.
#
# No board file is needed: PYNQ loads the bitstream onto an already-booted
# PS, so the PS DDR/MIO settings of this design are never used.  What does
# matter, and is set explicitly below, is the PL side: which AXI ports are
# enabled, their widths, and the PL clock.
#
#   axi_smc    PS M_AXI_HPM0_FPD -> kernel s_axi_control
#   axi_smc_1  all kernel AXI masters -> one 128-bit PS S_AXI_HP0_FPD
#
# The four PEs' S ports therefore share one HP port, as in the paper.
#
# args: <prj_dir> <ip_repo> <out_dir> <name> [pl_mhz, default 150]
set prj     [lindex $argv 0]
set ip_repo [lindex $argv 1]
set outdir  [lindex $argv 2]
set name    [lindex $argv 3]
set mhz     [expr {[llength $argv] > 4 ? [lindex $argv 4] : 150}]
set part    xczu7ev-ffvc1156-2-e

# Serialise IP generation: parallel TTCL interpreters failed to spawn on our
# Windows install and took the PS and xlconstant IP down with them.
set_param general.maxThreads 1

create_project -force $name $prj -part $part
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog -rebuild

create_bd_design "design_1"

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $mhz \
] $ps

set k   [create_bd_cell -type ip -vlnv xilinx.com:hls:spmm_kernel:1.0 spmm_kernel_0]
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_pl0]
connect_bd_net [get_bd_pins ps/pl_clk0]    [get_bd_pins rst_pl0/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst_pl0/ext_reset_in]

set smc0 [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect axi_smc]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $smc0
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins axi_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M00_AXI] \
                    [get_bd_intf_pins spmm_kernel_0/s_axi_control]

set masters [lsort [get_bd_intf_pins -of_objects $k -filter {MODE == Master}]]
set n [llength $masters]
puts "INFO: spmm_kernel has $n AXI master ports"
set smc1 [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect axi_smc_1]
set_property -dict [list CONFIG.NUM_SI $n CONFIG.NUM_MI {1}] $smc1
set i 0
foreach p $masters {
    connect_bd_intf_net $p [get_bd_intf_pins [format "axi_smc_1/S%02d_AXI" $i]]
    incr i
}
connect_bd_intf_net [get_bd_intf_pins axi_smc_1/M00_AXI] \
                    [get_bd_intf_pins ps/S_AXI_HP0_FPD]

foreach pin [list ps/maxihpm0_fpd_aclk ps/saxihp0_fpd_aclk \
                  axi_smc/aclk axi_smc_1/aclk spmm_kernel_0/ap_clk] {
    connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins $pin]
}
foreach pin [list axi_smc/aresetn axi_smc_1/aresetn spmm_kernel_0/ap_rst_n] {
    connect_bd_net [get_bd_pins rst_pl0/peripheral_aresetn] [get_bd_pins $pin]
}

assign_bd_address
set_property SELECTED_SIM_MODEL rtl [get_bd_cells ps]
validate_bd_design
save_bd_design

generate_target all [get_files design_1.bd]
make_wrapper -files [get_files design_1.bd] -top
add_files -norecurse [file join $prj $name.gen sources_1 bd design_1 hdl design_1_wrapper.v]
set_property top design_1_wrapper [current_fileset]

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# PYNQ needs the .hwh next to the .bit, with the same base name
file mkdir $outdir
file copy -force [file join $prj $name.runs impl_1 design_1_wrapper.bit] \
                 [file join $outdir $name.bit]
file copy -force [file join $prj $name.gen sources_1 bd design_1 hw_handoff design_1.hwh] \
                 [file join $outdir $name.hwh]

open_run impl_1
puts "=== TIMING ==="
report_timing_summary -no_header -no_detailed_paths
puts "=== UTILIZATION ==="
report_utilization
puts "BUILD_DONE $name"
