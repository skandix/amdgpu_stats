import pyamdgpuinfo as amd_info
from math import floor

#some math to calculate B to MB 
bytes_to_human = lambda bytes: (f"{int(floor(bytes/2048)**0.5)} MB")
hertz_to_human = lambda hertz: (f"{int(hertz)/10**6} Mhz")

# init gpus
gpus = amd_info.setup_gpus()
all_gpu = list(gpus.keys())

# lambda functions to get values from gpu's
temprature = lambda gpus: f"{amd_info.query_temp(gpus)} C"
power_usage = lambda gpus: f"{amd_info.query_power(gpus)} W"
vram_usage = lambda gpus: f"{bytes_to_human(amd_info.query_vram_usage(gpus))}"
load = lambda gpus: f"{(amd_info.query_load(gpus))} "
shader_core_clock = lambda gpus: f"{hertz_to_human(amd_info.query_sclk(gpus))}"
memory_clock = lambda gpus: f"{hertz_to_human(amd_info.query_mclk(gpus))}"
max_mem_clock = lambda gpus: f"{hertz_to_human([v for v in (amd_info.query_max_clocks(gpus)).values()][1])}"
max_shader_clock = lambda gpus: f"{hertz_to_human([v for v in (amd_info.query_max_clocks(gpus)).values()][0])}"

# fancy shit for seeing gpu metrics
def get_stats():
	for gpu in all_gpu:
		print (f"""\n{gpu}
		Vram Usage:       {vram_usage(gpu)}
		Power Usage:   	  {power_usage(gpu)}
		Temprature:    	  {temprature(gpu)}
		Load:	 	  {load(gpu)}
		Shader Clock:  	  {shader_core_clock(gpu)}
		Max Shader Clock: {max_shader_clock(gpu)}
		Memory Clock:  	  {memory_clock(gpu)}
		Max mem Clock: 	  {max_mem_clock(gpu)}
		""")



(get_stats())
